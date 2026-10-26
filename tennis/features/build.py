"""Pre-match feature construction.

Leakage control is structural, not a post-hoc check. The engine walks matches in
chronological order and, for each one, *reads* player state to emit a feature
row and only *then* folds the result into that state. Nothing computed for a
match can contain information from that match or any later one. The unit test in
tests/test_leakage.py asserts this property directly.

Orientation: rows are emitted as (p1, p2) with the side chosen by a hash of
match_id, so 'p1' is independent of who won and the label is ~50/50. Feeding the
model raw winner/loser columns would otherwise be a trivially perfect leak.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import logging
import math
import pickle
from collections import defaultdict, deque
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import islice

import numpy as np
import pandas as pd

from tennis.config import ARTIFACTS
from tennis.features.elo import EloBook
from tennis.ingest.parse import DEFAULT, UNFINISHED, WALKOVER, tiebreaks_won

log = logging.getLogger(__name__)

FORM_WINDOWS = (10, 25, 50)
SURF_WINDOW = 25
STAT_WINDOW = 25
MAXLEN = 60  # longest window we ever read

# Statuses that carry no result worth rating or learning from: a walkover was
# never played, a default is a disqualification, and an unfinished match has no
# winner to speak of. Retirements are NOT in here -- a retirement is a win, and
# the Elo book is tuned with them counted.
NOT_PLAYED = (WALKOVER, DEFAULT, UNFINISHED)

# Pseudo-observations of 0.5 added to a win rate before it is read. A player
# with 1 tiebreak played and 1 won is not a 100% tiebreak player; the prior
# says how many matches of evidence it takes to move off 0.5.
SHRINK_K = {"decider": 10.0, "tb": 10.0, "career": 20.0}
# Surface serve/return rates blend toward the all-surface figure by
# n / (n + K), the way the surface Elo does.
SURF_SERVE_K = 15.0
# Tournament speed is read from this many prior seasons at most.
SPEED_YEARS = 5
SPEED_YEARS_SHORT = 3        # trailing window for the tspeed3_* variants
# Layout of a tournament-season bucket: aces and the serve points they are
# counted over, points won on serve and serve points, games and sets, matches.
_SPEED_FIELDS = ("ace", "ace_pts", "won", "svpt", "games", "sets", "n")


@lru_cache(maxsize=None)
def _date_to_ord(d: int) -> int:
    # A season has ~52 distinct tourney_dates and the loop calls this twice per
    # match, so cache it; datetime.date is also ~10x cheaper than pd.Timestamp.
    y, m, dd = d // 10000, (d // 100) % 100, d % 100
    return _dt.date(y, max(m, 1), max(dd, 1)).toordinal()


# Days from an event's first day to the day each round is played. The archived
# files stamp every match with the Monday the event began, so without this a
# final reads as played on day 0 and every later-round opponent as having had
# zero rest. Medians of (match_date - event start) over the 66,553 ATP matches
# whose true date is known, by event shape; challengers and the WTA, which
# carry no per-match dates at all, borrow the one-week shape.
_ROUND_DAY = {
    "slam": {"R128": 1, "R64": 3, "R32": 5, "R16": 7, "QF": 9, "SF": 11, "F": 13},
    "long": {"R128": 1, "R64": 3, "R32": 5, "R16": 7, "QF": 8, "SF": 10, "F": 11},
    "week": {"R64": 1, "R32": 1, "R16": 3, "QF": 4, "SF": 5, "F": 6,
             "RR": 1, "BR": 6, "3RD/4TH": 6},
}


def _round_day(row) -> int:
    """Estimated days after the event's first day that this match was played."""
    if row.get("level") == "grand_slam":
        shape = "slam"
    else:
        ds = row.get("draw_size")
        shape = "long" if ds is not None and ds == ds and ds >= 96 else "week"
    rnd = row.get("round")
    return _ROUND_DAY[shape].get(rnd if isinstance(rnd, str) else "", 0)


def _play_ord(row) -> int:
    """Day-ordinal a match was played on, as well as the data allows.

    `tourney_date` is the event's start, kept for grouping. Where the source
    recorded the real day (`date_is_exact`) that wins; where it stamped the whole
    event with its first day, the day is placed by round. Both are known before
    the match is played, so this is not a leak. Rows that do not say either way
    -- a live prediction, whose `tourney_date` already is the day -- are taken
    at face value.
    """
    start = int(row["tourney_date"])
    if row.get("date_is_exact") == 0:
        return _date_to_ord(start) + _round_day(row)
    md = row.get("match_date")
    if row.get("date_is_exact") == 1 and md is not None and md == md:
        return _date_to_ord(int(md))
    return _date_to_ord(start)


def _side_is_winner_first(match_id: str) -> bool:
    return hashlib.md5(match_id.encode()).digest()[0] % 2 == 0


def _mean(dq, n=None):
    if not dq:
        return np.nan
    # Walk the tail only; list(dq) copied the whole deque on every call.
    vals = list(islice(reversed(dq), n)) if n else dq
    # Plain sum/len: np.mean's per-call array conversion dominated the build.
    # NaN still propagates, as it did under np.mean.
    return sum(vals) / len(vals) if len(vals) else np.nan


def _shrunk(w, n, k):
    """Win rate pulled toward 0.5 by k pseudo-observations (0.5 when n == 0)."""
    return (w + 0.5 * k) / (n + k)


def _tourney_key(row) -> str:
    """Year-agnostic identity of a tournament, for the speed index."""
    name = row.get("tourney_name")
    if not isinstance(name, str) or not name:
        return ""
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _safe_div(a, b):
    return a / b if (b is not None and b > 0 and a is not None) else np.nan


def _dq() -> deque:
    """Module-level factory so PlayerState stays picklable for incremental runs."""
    return deque(maxlen=MAXLEN)


# Half-life in matches for the exponentially weighted form features. A flat
# 25-match mean treats a match from 14 months ago exactly like last week's and
# then drops it off a cliff; an EWMA decays smoothly and has no cutoff. Two
# horizons because "hot right now" and "good this season" are different things.
EWMA_HALFLIVES = {"fast": 5.0, "slow": 20.0}


class _Ewma:
    """Exponentially weighted mean, bias-corrected while still warming up."""

    __slots__ = ("alpha", "value", "weight")

    def __init__(self, half_life: float):
        self.alpha = 1.0 - 0.5 ** (1.0 / half_life)
        self.value = 0.0
        self.weight = 0.0

    def push(self, x: float) -> None:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return
        self.value = self.alpha * x + (1 - self.alpha) * self.value
        # tracks total applied weight so early values are not shrunk toward 0
        self.weight = self.alpha + (1 - self.alpha) * self.weight

    def get(self) -> float:
        return self.value / self.weight if self.weight > 1e-9 else np.nan


def _ewma_pair() -> dict:
    return {k: _Ewma(h) for k, h in EWMA_HALFLIVES.items()}


@dataclass
class PlayerState:
    """Rolling per-player history. All deques are append-on-result."""

    results: deque = field(default_factory=_dq)
    # plain dict, not defaultdict: a defaultdict's factory would be a closure
    # and would break pickling of the engine state.
    surf_results: dict = field(default_factory=dict)
    # serve/return rates, only appended for matches that carry stats
    spw: deque = field(default_factory=_dq)
    rpw: deque = field(default_factory=_dq)
    first_in: deque = field(default_factory=_dq)
    first_won: deque = field(default_factory=_dq)
    second_won: deque = field(default_factory=_dq)
    ace_rate: deque = field(default_factory=_dq)
    df_rate: deque = field(default_factory=_dq)
    bp_saved: deque = field(default_factory=_dq)
    hold: deque = field(default_factory=_dq)
    brk: deque = field(default_factory=_dq)
    # totals/spread oriented, only for cleanly completed matches
    games_per_set: deque = field(default_factory=_dq)
    total_games: deque = field(default_factory=_dq)
    margin: deque = field(default_factory=_dq)
    # schedule
    dates: deque = field(default_factory=_dq)
    minutes: deque = field(default_factory=_dq)
    last_date: int | None = None
    career_w: int = 0
    career_n: int = 0
    # exponentially weighted form, as an alternative to the flat windows
    ew_result: dict = field(default_factory=_ewma_pair)
    ew_spw: dict = field(default_factory=_ewma_pair)
    ew_rpw: dict = field(default_factory=_ewma_pair)
    ew_margin: dict = field(default_factory=_ewma_pair)
    # strength of recent opposition, so raw serve/return rates can be read in
    # context: 70% of serve points against a weak field is not 70% against
    # the top ten, and the model cannot know that from the rate alone
    opp_elo: deque = field(default_factory=_dq)
    # clutch: deciding sets and tiebreaks (tb_* count individual tiebreaks
    # won and played, not matches that contained one)
    decider_w: int = 0
    decider_n: int = 0
    tb_w: int = 0
    tb_n: int = 0
    # serve/return relative to what each opponent usually concedes
    ew_adj_spw: dict = field(default_factory=_ewma_pair)
    ew_adj_rpw: dict = field(default_factory=_ewma_pair)
    # per-surface serve/return, {surface: _Ewma}, and matches-with-stats counts
    surf_spw: dict = field(default_factory=dict)
    surf_rpw: dict = field(default_factory=dict)
    surf_stat_n: dict = field(default_factory=dict)
    # the same, but relative to what each opponent usually concedes
    surf_adj_spw: dict = field(default_factory=dict)
    surf_adj_rpw: dict = field(default_factory=dict)
    # surface continuity
    last_surface: str = ""
    surf_streak: int = 0


class FeatureEngine:
    """Holds all running state; can be pickled to support incremental updates."""

    # Bumped whenever a change alters what the engine would emit for matches it
    # has already folded in, so a saved state from before it is rebuilt rather
    # than extended. 2: matches dated by round within an event, and defaults /
    # unfinished matches no longer rated.
    VERSION = 2

    def __init__(self) -> None:
        self.version = FeatureEngine.VERSION
        self.elo = EloBook()
        self.players: dict[str, PlayerState] = defaultdict(PlayerState)
        self.h2h: dict[tuple[str, str], int] = defaultdict(int)
        self.h2h_surf: dict[tuple[str, str, str], int] = defaultdict(int)
        self.last_seq: int = -1
        # Overall ratings as they stood on the first day of the current season,
        # {"start": YYYYMMDD, "ratings": {player_id: rating}}. Served so the
        # dashboard need not replay history to show season-to-date movement.
        self.season_snap: dict | None = None
        # Tournament-edition aggregates, {(tourney, surface): {year: [...]}}.
        self.speed: dict[tuple[str, str], dict[int, list]] = {}

    # -- reading -----------------------------------------------------------
    def _player_feats(self, pid: str, surface: str, date_ord: int) -> dict:
        st = self.players.get(pid)
        f: dict[str, float] = {}
        # date_ord drives layoff decay: a rating read months after the player
        # last played is worth less than the number sitting in the book.
        f["elo"] = self.elo.get(pid, "overall", date_ord)
        f["elo_n"] = self.elo.n(pid, "overall")
        f["elo_surf"] = self.elo.get(pid, surface, date_ord) if surface else np.nan
        f["elo_surf_n"] = self.elo.n(pid, surface) if surface else np.nan

        if st is None:
            st = PlayerState()
        for w in FORM_WINDOWS:
            f[f"winrate_{w}"] = _mean(st.results, w)
        f["surf_winrate"] = _mean(st.surf_results.get(surface, deque()), SURF_WINDOW) \
            if surface else np.nan
        f["career_n"] = st.career_n
        f["career_winrate"] = _safe_div(st.career_w, st.career_n)

        f["spw"] = _mean(st.spw, STAT_WINDOW)
        f["rpw"] = _mean(st.rpw, STAT_WINDOW)
        f["first_in"] = _mean(st.first_in, STAT_WINDOW)
        f["first_won"] = _mean(st.first_won, STAT_WINDOW)
        f["second_won"] = _mean(st.second_won, STAT_WINDOW)
        f["ace_rate"] = _mean(st.ace_rate, STAT_WINDOW)
        f["df_rate"] = _mean(st.df_rate, STAT_WINDOW)
        f["bp_saved"] = _mean(st.bp_saved, STAT_WINDOW)
        f["hold_pct"] = _mean(st.hold, STAT_WINDOW)
        f["break_pct"] = _mean(st.brk, STAT_WINDOW)
        # Total points won per point played: serve% and return% combined. Above
        # 1.0 means the player wins more points than they lose overall.
        f["serve_edge"] = _diff(f["spw"], 1.0 - f["rpw"]) if not (
            f["spw"] != f["spw"] or f["rpw"] != f["rpw"]) else np.nan

        f["avg_games_per_set"] = _mean(st.games_per_set, STAT_WINDOW)
        f["avg_total_games"] = _mean(st.total_games, STAT_WINDOW)
        f["avg_margin"] = _mean(st.margin, STAT_WINDOW)

        # EWMA form. Same underlying signal as winrate_10/25/50 but decayed
        # smoothly instead of a hard 25-match cutoff.
        for k in EWMA_HALFLIVES:
            f[f"ew_winrate_{k}"] = st.ew_result[k].get()
            f[f"ew_spw_{k}"] = st.ew_spw[k].get()
            f[f"ew_rpw_{k}"] = st.ew_rpw[k].get()
            f[f"ew_margin_{k}"] = st.ew_margin[k].get()

        # Strength of the opposition those rates were compiled against.
        f["opp_elo_25"] = _mean(st.opp_elo, STAT_WINDOW)
        f["opp_elo_10"] = _mean(st.opp_elo, 10)

        f["decider_winrate"] = _safe_div(st.decider_w, st.decider_n)
        f["decider_n"] = st.decider_n
        f["tb_winrate"] = _safe_div(st.tb_w, st.tb_n)
        f["tb_n"] = st.tb_n
        f["decider_wr_s"] = _shrunk(st.decider_w, st.decider_n, SHRINK_K["decider"])
        f["tb_wr_s"] = _shrunk(st.tb_w, st.tb_n, SHRINK_K["tb"])
        f["career_wr_s"] = _shrunk(st.career_w, st.career_n, SHRINK_K["career"])

        # Serve/return over and above what the opposition normally concedes.
        for k in EWMA_HALFLIVES:
            f[f"ew_adj_spw_{k}"] = st.ew_adj_spw[k].get()
            f[f"ew_adj_rpw_{k}"] = st.ew_adj_rpw[k].get()
        # Surface-specific serve/return, blended toward the all-surface figure.
        for key, book, overall in (("spw_surf", st.surf_spw, f["ew_spw_slow"]),
                                   ("rpw_surf", st.surf_rpw, f["ew_rpw_slow"])):
            e = book.get(surface) if surface else None
            if e is None or e.get() != e.get():
                f[key] = overall
            else:
                n = st.surf_stat_n.get(surface, 0)
                w = n / (n + SURF_SERVE_K)
                f[key] = w * e.get() + (1.0 - w) * overall
        # ... and the opponent-adjusted version of both, by surface.
        for key, book, overall in (
                ("adj_spw_surf", st.surf_adj_spw, f["ew_adj_spw_slow"]),
                ("adj_rpw_surf", st.surf_adj_rpw, f["ew_adj_rpw_slow"])):
            e = book.get(surface) if surface else None
            if e is None or e.get() != e.get() or overall != overall:
                f[key] = overall
            else:
                n = st.surf_stat_n.get(surface, 0)
                w = n / (n + SURF_SERVE_K)
                f[key] = w * e.get() + (1.0 - w) * overall

        # Matches played on this surface since the last switch. A player fresh
        # off a surface change is a different proposition from one mid-swing.
        f["surf_streak"] = st.surf_streak if st.last_surface == surface else 0

        f["rest_days"] = max(date_ord - st.last_date, 0) if st.last_date is not None else np.nan
        f["matches_14d"] = sum(1 for d in st.dates if 0 <= date_ord - d <= 14)
        f["matches_30d"] = sum(1 for d in st.dates if 0 <= date_ord - d <= 30)
        f["minutes_14d"] = float(sum(
            m for d, m in zip(st.dates, st.minutes)
            if 0 <= date_ord - d <= 14 and m == m
        )) if st.dates else np.nan
        return f

    def features_for(self, row, *, force_winner_first: bool | None = None) -> dict:
        """Emit the pre-match feature row for a match (does not mutate state).

        `force_winner_first` pins the p1/p2 orientation instead of hashing it.
        Live prediction uses it to put a chosen player on the p1 side; training
        leaves it None so orientation stays outcome-independent.
        """
        surface = row["surface"] if isinstance(row["surface"], str) else ""
        date_ord = _play_ord(row)
        w, l = row["winner_id"], row["loser_id"]

        winner_first = (
            _side_is_winner_first(row["match_id"])
            if force_winner_first is None else force_winner_first
        )
        p1, p2 = (w, l) if winner_first else (l, w)
        y = 1 if winner_first else 0

        f1 = self._player_feats(p1, surface, date_ord)
        f2 = self._player_feats(p2, surface, date_ord)

        out = {"match_id": row["match_id"], "seq": row["seq"],
               "tourney_date": row["tourney_date"], "p1_id": p1, "p2_id": p2,
               "y_win": y}
        for k, v in f1.items():
            out[f"p1_{k}"] = v
        for k, v in f2.items():
            out[f"p2_{k}"] = v

        # side-specific raw attributes, mapped from winner/loser to p1/p2
        def side(col_w, col_l):
            return (row[col_w], row[col_l]) if winner_first else (row[col_l], row[col_w])

        for name, (cw, cl) in {
            "rank": ("winner_rank", "loser_rank"),
            "rank_points": ("winner_rank_points", "loser_rank_points"),
            "age": ("winner_age", "loser_age"),
            "ht": ("winner_ht", "loser_ht"),
            "seed": ("winner_seed", "loser_seed"),
        }.items():
            a, b = side(cw, cl)
            out[f"p1_{name}"], out[f"p2_{name}"] = a, b

        hw, hl = side("winner_hand", "loser_hand")
        out["p1_lefty"] = 1.0 if hw == "L" else (0.0 if hw == "R" else np.nan)
        out["p2_lefty"] = 1.0 if hl == "L" else (0.0 if hl == "R" else np.nan)

        for c in ("p1_rank", "p2_rank"):
            out[f"log_{c}"] = math.log(out[c]) if out[c] and out[c] > 0 else np.nan

        # head-to-head, oriented to p1
        # .get, not []: indexing a defaultdict inserts the key, so every
        # prediction grew the saved and cached engines.
        out["h2h_p1"] = self.h2h.get((p1, p2), 0)
        out["h2h_p2"] = self.h2h.get((p2, p1), 0)
        out["h2h_n"] = out["h2h_p1"] + out["h2h_p2"]
        out["h2h_surf_p1"] = self.h2h_surf.get((p1, p2, surface), 0)
        out["h2h_surf_p2"] = self.h2h_surf.get((p2, p1, surface), 0)

        # match context
        out["best_of"] = row["best_of"]
        out["round_idx"] = row["round_idx"]
        out["draw_size"] = row["draw_size"]
        out["indoor"] = row["indoor"]
        out["surface"] = surface
        out["level"] = row["level"]
        out["is_challenger"] = row["is_challenger"]

        # Symmetric serve features: the totals and spread targets depend on how
        # both players serve, not on which is called p1.
        out["spw_sum"] = out["p1_spw"] + out["p2_spw"]
        out["hold_prod"] = out["p1_hold_pct"] * out["p2_hold_pct"]
        # Expected serve-point win rate for each server: their own rate and
        # what the opposing returner typically concedes, averaged.
        out["exp_serve_p1"] = (out["p1_spw"] + 1.0 - out["p2_rpw"]) / 2.0
        out["exp_serve_p2"] = (out["p2_spw"] + 1.0 - out["p1_rpw"]) / 2.0
        out["exp_serve_sum"] = out["exp_serve_p1"] + out["exp_serve_p2"]
        out.update(self._speed_feats(row, surface, int(row["tourney_date"]) // 10000))

        # pairwise differences (the model's main signal)
        for base in ("elo", "elo_surf", "winrate_10", "winrate_25", "winrate_50",
                     "surf_winrate", "spw", "rpw", "hold_pct", "break_pct",
                     "serve_edge", "first_won", "second_won", "ace_rate",
                     "career_winrate", "rest_days", "matches_14d",
                     "avg_total_games", "avg_margin", "avg_games_per_set",
                     "ew_winrate_fast", "ew_winrate_slow", "ew_spw_fast",
                     "ew_spw_slow", "ew_rpw_fast", "ew_rpw_slow",
                     "ew_margin_fast", "ew_margin_slow", "opp_elo_25",
                     "opp_elo_10", "decider_winrate", "tb_winrate",
                     "surf_streak", "decider_wr_s", "tb_wr_s", "career_wr_s",
                     "ew_adj_spw_slow", "ew_adj_rpw_slow", "spw_surf",
                     "rpw_surf", "adj_spw_surf", "adj_rpw_surf"):
            a, b = out.get(f"p1_{base}"), out.get(f"p2_{base}")
            out[f"d_{base}"] = (a - b) if (a is not None and b is not None
                                           and a == a and b == b) else np.nan
        out["d_rank"] = _diff(out.get("p1_rank"), out.get("p2_rank"))
        out["d_log_rank"] = _diff(out.get("log_p1_rank"), out.get("log_p2_rank"))
        out["d_rank_points"] = _diff(out.get("p1_rank_points"), out.get("p2_rank_points"))
        out["d_age"] = _diff(out.get("p1_age"), out.get("p2_age"))
        out["d_ht"] = _diff(out.get("p1_ht"), out.get("p2_ht"))
        out["d_h2h"] = out["h2h_p1"] - out["h2h_p2"]
        out["elo_prob"] = 1.0 / (1.0 + 10 ** ((out["p2_elo"] - out["p1_elo"]) / 400.0))
        out["elo_surf_prob"] = (
            1.0 / (1.0 + 10 ** ((out["p2_elo_surf"] - out["p1_elo_surf"]) / 400.0))
            if not (out["p1_elo_surf"] != out["p1_elo_surf"]
                    or out["p2_elo_surf"] != out["p2_elo_surf"]) else np.nan
        )
        return out

    def _speed_feats(self, row, surface: str, year: int) -> dict:
        """How fast this tournament has played in previous seasons.

        Strictly prior seasons (year < this match's year): the current
        season's matches at the same event are already in the aggregates but
        are never read, so a round-1 and a final see the same index and nothing
        from the event itself can leak into it.
        """
        nan = np.nan
        out = {"tspeed_ace": nan, "tspeed_spw": nan, "tspeed_gps": nan,
               "tspeed_n": 0, "tspeed3_spw": nan, "tspeed3_ace": nan}
        key = _tourney_key(row)
        years = self.speed.get((key, surface)) if key else None
        if not years:
            return out
        tot = [0.0] * len(_SPEED_FIELDS)
        tot3 = [0.0] * len(_SPEED_FIELDS)
        for y in range(year - SPEED_YEARS, year):
            bucket = years.get(y)
            if bucket:
                tot = [t + x for t, x in zip(tot, bucket)]
                if y >= year - SPEED_YEARS_SHORT:
                    tot3 = [t + x for t, x in zip(tot3, bucket)]
        if tot3[3] > 0:
            out["tspeed3_spw"] = tot3[2] / tot3[3]
        if tot3[1] > 0:
            out["tspeed3_ace"] = tot3[0] / tot3[1]
        ace, ace_pts, won, svpt, games, sets, n = tot
        if ace_pts > 0:
            out["tspeed_ace"] = ace / ace_pts
        if svpt > 0:
            out["tspeed_spw"] = won / svpt
        if sets > 0:
            out["tspeed_gps"] = games / sets
        out["tspeed_n"] = n
        return out

    # -- writing -----------------------------------------------------------
    def update(self, row) -> None:
        """Fold a played match into state. Call only after features_for()."""
        status = row["status"]
        if status in NOT_PLAYED:
            return  # nothing, or nothing countable, happened on court

        surface = row["surface"] if isinstance(row["surface"], str) else ""
        date_ord = _play_ord(row)
        w, l = row["winner_id"], row["loser_id"]

        # Opponent strength is what was known going in, so read it before this
        # match touches either rating.
        w_pre = self.elo.get(w, "overall", date_ord)
        l_pre = self.elo.get(l, "overall", date_ord)

        # Layoff decay owed to every scope is realised now, while the old clock
        # still says how long they were idle.
        self.elo.advance(w, date_ord)
        self.elo.advance(l, date_ord)

        # Elo: a retirement is still a win, so it rates; a walkover does not.
        # Surface goes first: a surface rating's first update is seeded from
        # the overall rating, which must still be the pre-match value.
        if surface:
            self.elo.update(w, l, surface, day=date_ord)
        self.elo.update(w, l, "overall", day=date_ord)
        # Stop the layoff clock only once both scopes have updated -- they share
        # it, and touching earlier would cancel the decay the surface update is
        # owed. Every later read at this date_ord then sees a zero-length gap.
        self.elo.touch(w, date_ord)
        self.elo.touch(l, date_ord)

        self.h2h[(w, l)] += 1
        if surface:
            self.h2h_surf[(w, l, surface)] += 1

        sw, sl = self.players[w], self.players[l]
        sw.results.append(1.0)
        sl.results.append(0.0)
        for k in EWMA_HALFLIVES:
            sw.ew_result[k].push(1.0)
            sl.ew_result[k].push(0.0)
        sw.opp_elo.append(l_pre)
        sl.opp_elo.append(w_pre)

        # Surface continuity
        for st_, in ((sw,), (sl,)):
            if surface:
                st_.surf_streak = (st_.surf_streak + 1
                                   if st_.last_surface == surface else 1)
                st_.last_surface = surface
        if surface:
            sw.surf_results.setdefault(surface, _dq()).append(1.0)
            sl.surf_results.setdefault(surface, _dq()).append(0.0)
        sw.career_w += 1
        sw.career_n += 1
        sl.career_n += 1
        for st in (sw, sl):
            st.last_date = date_ord if st.last_date is None else max(st.last_date, date_ord)
            st.dates.append(date_ord)
            st.minutes.append(row["minutes"] if not pd.isna(row["minutes"]) else np.nan)

        # Tiebreaks, from who took each 7-6 set. Counted for any match that
        # was played, retirements included: a completed tiebreak is a real
        # result whatever happened afterwards.
        wtb, ltb = tiebreaks_won(row.get("score"))
        if wtb or ltb:
            sw.tb_w += wtb
            sl.tb_w += ltb
            sw.tb_n += wtb + ltb
            sl.tb_n += wtb + ltb

        # Serve/return rates need real counters; skip when absent.
        if row["has_stats"] == 1:
            # What each side was expected to concede, read before this match
            # touches either EWMA: the opponent's state going in, not after.
            w_ret, w_srv = sw.ew_rpw["slow"].get(), sw.ew_spw["slow"].get()
            l_ret, l_srv = sl.ew_rpw["slow"].get(), sl.ew_spw["slow"].get()
            self._push_stats(sw, sl, row, w_side=True)
            self._push_stats(sl, sw, row, w_side=False)
            for st_, opp_ret, opp_srv in ((sw, l_ret, l_srv), (sl, w_ret, w_srv)):
                for k in EWMA_HALFLIVES:
                    if st_.spw:
                        st_.ew_spw[k].push(st_.spw[-1])
                        if opp_ret == opp_ret:
                            st_.ew_adj_spw[k].push(st_.spw[-1] - (1.0 - opp_ret))
                    if st_.rpw:
                        st_.ew_rpw[k].push(st_.rpw[-1])
                        if opp_srv == opp_srv:
                            st_.ew_adj_rpw[k].push(st_.rpw[-1] - (1.0 - opp_srv))
                if surface and st_.spw and st_.rpw:
                    if opp_ret == opp_ret:
                        st_.surf_adj_spw.setdefault(surface, _Ewma(EWMA_HALFLIVES["slow"])
                                                    ).push(st_.spw[-1] - (1.0 - opp_ret))
                    if opp_srv == opp_srv:
                        st_.surf_adj_rpw.setdefault(surface, _Ewma(EWMA_HALFLIVES["slow"])
                                                    ).push(st_.rpw[-1] - (1.0 - opp_srv))
                    st_.surf_spw.setdefault(surface, _Ewma(EWMA_HALFLIVES["slow"])
                                            ).push(st_.spw[-1])
                    st_.surf_rpw.setdefault(surface, _Ewma(EWMA_HALFLIVES["slow"])
                                            ).push(st_.rpw[-1])
                    st_.surf_stat_n[surface] = st_.surf_stat_n.get(surface, 0) + 1

        # Totals/spread history only from cleanly completed matches, so a
        # retirement never depresses a player's games-per-set average.
        if row["totals_usable"] == 1 and row["total_sets"]:
            gps = row["total_games"] / row["total_sets"]
            sw.games_per_set.append(gps)
            sl.games_per_set.append(gps)
            sw.total_games.append(row["total_games"])
            sl.total_games.append(row["total_games"])
            sw.margin.append(row["game_margin"])
            sl.margin.append(-row["game_margin"])
            for k in EWMA_HALFLIVES:
                sw.ew_margin[k].push(float(row["game_margin"]))
                sl.ew_margin[k].push(-float(row["game_margin"]))

            # Deciding set: went the distance for the format.
            needed = 3 if (row["best_of"] or 3) >= 5 else 2
            if row["l_sets"] == needed - 1:
                sw.decider_w += 1
                sw.decider_n += 1
                sl.decider_n += 1

        # Tournament speed. Stored per season; _speed_feats reads only earlier ones.
        key = _tourney_key(row)
        if key:
            b = self.speed.setdefault((key, surface), {}).setdefault(
                int(row["tourney_date"]) // 10000, [0.0] * len(_SPEED_FIELDS))
            if row["has_stats"] == 1:
                aces = row["w_ace"] + row["l_ace"]
                if aces == aces:             # ace counts can be missing alone
                    b[0] += aces
                    b[1] += row["w_svpt"] + row["l_svpt"]
                b[2] += (row["w_1stWon"] + row["w_2ndWon"]
                         + row["l_1stWon"] + row["l_2ndWon"])
                b[3] += row["w_svpt"] + row["l_svpt"]
                b[6] += 1
            if row["totals_usable"] == 1 and row["total_sets"]:
                b[4] += row["total_games"]
                b[5] += row["total_sets"]

        self.last_seq = max(self.last_seq, int(row["seq"]))

    @staticmethod
    def _push_stats(me: PlayerState, opp: PlayerState, row, *, w_side: bool) -> None:
        p = "w_" if w_side else "l_"
        q = "l_" if w_side else "w_"
        svpt, firstIn = row[p + "svpt"], row[p + "1stIn"]
        firstWon, secondWon = row[p + "1stWon"], row[p + "2ndWon"]
        svgms, bpf, bps = row[p + "SvGms"], row[p + "bpFaced"], row[p + "bpSaved"]
        o_svpt = row[q + "svpt"]
        o_first, o_second = row[q + "1stWon"], row[q + "2ndWon"]
        o_svgms, o_bpf, o_bps = row[q + "SvGms"], row[q + "bpFaced"], row[q + "bpSaved"]

        if svpt and svpt > 0:
            me.spw.append((firstWon + secondWon) / svpt)
            me.first_in.append(_safe_div(firstIn, svpt))
            me.first_won.append(_safe_div(firstWon, firstIn))
            me.second_won.append(_safe_div(secondWon, svpt - firstIn))
            me.ace_rate.append(_safe_div(row[p + "ace"], svpt))
            me.df_rate.append(_safe_div(row[p + "df"], svpt))
        if o_svpt and o_svpt > 0:
            me.rpw.append(1.0 - (o_first + o_second) / o_svpt)
        if bpf and bpf > 0:
            me.bp_saved.append(bps / bpf)
        # Missing break-point counts must stay missing. `max(0.0, nan)` is 0.0
        # and `min(1.0, nan)` is 1.0 in Python, so a NaN used to become a 0%
        # hold or a 100% break.
        if svgms and svgms > 0 and not (pd.isna(bpf) or pd.isna(bps)):
            # break points converted against us is the usual proxy for breaks
            breaks_lost = max(bpf - bps, 0)
            me.hold.append(max(0.0, 1.0 - breaks_lost / svgms))
        if o_svgms and o_svgms > 0 and not (pd.isna(o_bpf) or pd.isna(o_bps)):
            breaks_made = max(o_bpf - o_bps, 0)
            me.brk.append(min(1.0, breaks_made / o_svgms))

    # -- persistence -------------------------------------------------------
    # State is per tour, so the filename carries one. There is deliberately no
    # tourless default: a single shared "feature_state.pkl" is exactly how one
    # tour's ratings would end up answering for the other.
    def save(self, path=None, tour: str = "atp") -> None:
        path = path or (ARTIFACTS / f"feature_state_{tour}.pkl")
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path=None, tour: str = "atp") -> "FeatureEngine":
        path = path or (ARTIFACTS / f"feature_state_{tour}.pkl")
        with open(path, "rb") as fh:
            return pickle.load(fh)


def _diff(a, b):
    # x != x is the NaN test without pd.isna's dispatch cost, hot in the loop.
    if a is None or b is None or a != a or b != b:
        return np.nan
    return a - b


REQUIRED = [
    "match_id", "seq", "tourney_date", "match_date", "date_is_exact", "round",
    "winner_id", "loser_id", "surface",
    "level", "is_challenger", "status", "best_of", "round_idx", "draw_size",
    "indoor", "minutes", "has_stats", "totals_usable", "total_games",
    "total_sets", "game_margin", "winner_rank", "loser_rank",
    "winner_rank_points", "loser_rank_points", "winner_age", "loser_age",
    "winner_ht", "loser_ht", "winner_seed", "loser_seed", "winner_hand",
    "loser_hand", "score", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_SvGms",
    "w_bpSaved", "w_bpFaced", "w_ace", "w_df", "l_svpt", "l_1stIn", "l_1stWon",
    "l_2ndWon", "l_SvGms", "l_bpSaved", "l_bpFaced", "l_ace", "l_df",
]


def build_features(matches: pd.DataFrame, engine: FeatureEngine | None = None,
                   *, emit_from_seq: int = -1,
                   snapshot_start: int | None = None) -> tuple[pd.DataFrame, FeatureEngine]:
    """Walk matches in order, emitting a feature row for each.

    `emit_from_seq` lets an incremental run replay history to warm the state
    without re-emitting rows it already stored.

    `snapshot_start` (YYYYMMDD) records every player's overall rating at the
    moment the walk first reaches a match on or after that date, i.e. the state
    after all earlier matches and before any later one.
    """
    engine = engine or FeatureEngine()
    matches = matches.sort_values("seq")
    rows = matches.to_dict("records")
    out = []
    want_snap = (snapshot_start is not None and (
        engine.season_snap is None or engine.season_snap["start"] != snapshot_start))
    for i, row in enumerate(rows):
        if want_snap and int(row["tourney_date"]) >= snapshot_start:
            day0 = _date_to_ord(snapshot_start)
            engine.season_snap = {"start": int(snapshot_start), "ratings": {
                pid: engine.elo.get(pid, "overall", day0)
                for pid in engine.elo.players()}}
            want_snap = False
        if row["seq"] > emit_from_seq:
            out.append(engine.features_for(row))
        engine.update(row)
        if i % 25000 == 0 and i:
            log.info("  features %d/%d", i, len(rows))
    return pd.DataFrame(out), engine
