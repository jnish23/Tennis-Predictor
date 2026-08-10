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

import hashlib
import logging
import math
import pickle
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tennis.config import ARTIFACTS
from tennis.features.elo import EloBook

log = logging.getLogger(__name__)

FORM_WINDOWS = (10, 25, 50)
SURF_WINDOW = 25
STAT_WINDOW = 25
MAXLEN = 60  # longest window we ever read


def _date_to_ord(d: int) -> int:
    y, m, dd = d // 10000, (d // 100) % 100, d % 100
    return int(pd.Timestamp(year=y, month=max(m, 1), day=max(dd, 1)).toordinal())


def _side_is_winner_first(match_id: str) -> bool:
    return hashlib.md5(match_id.encode()).digest()[0] % 2 == 0


def _mean(dq, n=None):
    if not dq:
        return np.nan
    vals = list(dq)[-n:] if n else list(dq)
    return float(np.mean(vals)) if vals else np.nan


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
    # clutch: deciding sets and tiebreaks
    decider_w: int = 0
    decider_n: int = 0
    tb_w: int = 0
    tb_n: int = 0
    # surface continuity
    last_surface: str = ""
    surf_streak: int = 0


class FeatureEngine:
    """Holds all running state; can be pickled to support incremental updates."""

    def __init__(self) -> None:
        self.elo = EloBook()
        self.players: dict[str, PlayerState] = defaultdict(PlayerState)
        self.h2h: dict[tuple[str, str], int] = defaultdict(int)
        self.h2h_surf: dict[tuple[str, str, str], int] = defaultdict(int)
        self.last_seq: int = -1

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
            pd.isna(f["spw"]) or pd.isna(f["rpw"])) else np.nan

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

        # Matches played on this surface since the last switch. A player fresh
        # off a surface change is a different proposition from one mid-swing.
        f["surf_streak"] = st.surf_streak if st.last_surface == surface else 0

        f["rest_days"] = (date_ord - st.last_date) if st.last_date is not None else np.nan
        f["matches_14d"] = sum(1 for d in st.dates if 0 <= date_ord - d <= 14)
        f["matches_30d"] = sum(1 for d in st.dates if 0 <= date_ord - d <= 30)
        f["minutes_14d"] = float(np.nansum(
            [m for d, m in zip(st.dates, st.minutes) if 0 <= date_ord - d <= 14]
        )) if st.dates else np.nan
        return f

    def features_for(self, row, *, force_winner_first: bool | None = None) -> dict:
        """Emit the pre-match feature row for a match (does not mutate state).

        `force_winner_first` pins the p1/p2 orientation instead of hashing it.
        Live prediction uses it to put a chosen player on the p1 side; training
        leaves it None so orientation stays outcome-independent.
        """
        surface = row["surface"] if isinstance(row["surface"], str) else ""
        date_ord = _date_to_ord(int(row["tourney_date"]))
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
        out["h2h_p1"] = self.h2h[(p1, p2)]
        out["h2h_p2"] = self.h2h[(p2, p1)]
        out["h2h_n"] = out["h2h_p1"] + out["h2h_p2"]
        out["h2h_surf_p1"] = self.h2h_surf[(p1, p2, surface)]
        out["h2h_surf_p2"] = self.h2h_surf[(p2, p1, surface)]

        # match context
        out["best_of"] = row["best_of"]
        out["round_idx"] = row["round_idx"]
        out["draw_size"] = row["draw_size"]
        out["indoor"] = row["indoor"]
        out["surface"] = surface
        out["level"] = row["level"]
        out["is_challenger"] = row["is_challenger"]

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
                     "surf_streak"):
            a, b = out.get(f"p1_{base}"), out.get(f"p2_{base}")
            out[f"d_{base}"] = (a - b) if (a is not None and b is not None
                                           and not (pd.isna(a) or pd.isna(b))) else np.nan
        out["d_rank"] = _diff(out.get("p1_rank"), out.get("p2_rank"))
        out["d_log_rank"] = _diff(out.get("log_p1_rank"), out.get("log_p2_rank"))
        out["d_rank_points"] = _diff(out.get("p1_rank_points"), out.get("p2_rank_points"))
        out["d_age"] = _diff(out.get("p1_age"), out.get("p2_age"))
        out["d_ht"] = _diff(out.get("p1_ht"), out.get("p2_ht"))
        out["d_h2h"] = out["h2h_p1"] - out["h2h_p2"]
        out["elo_prob"] = 1.0 / (1.0 + 10 ** ((out["p2_elo"] - out["p1_elo"]) / 400.0))
        out["elo_surf_prob"] = (
            1.0 / (1.0 + 10 ** ((out["p2_elo_surf"] - out["p1_elo_surf"]) / 400.0))
            if not (pd.isna(out["p1_elo_surf"]) or pd.isna(out["p2_elo_surf"])) else np.nan
        )
        return out

    # -- writing -----------------------------------------------------------
    def update(self, row) -> None:
        """Fold a played match into state. Call only after features_for()."""
        status = row["status"]
        if status == "walkover":
            return  # nothing happened on court

        surface = row["surface"] if isinstance(row["surface"], str) else ""
        date_ord = _date_to_ord(int(row["tourney_date"]))
        w, l = row["winner_id"], row["loser_id"]

        # Elo: a retirement is still a win, so it rates; a walkover does not.
        self.elo.update(w, l, "overall", day=date_ord)
        if surface:
            self.elo.update(w, l, surface, day=date_ord)
        # Stop the layoff clock only once both scopes have updated -- they share
        # it, and touching earlier would cancel the decay the surface update is
        # owed. Every later read at this date_ord then sees a zero-length gap.
        self.elo.touch(w, date_ord)
        self.elo.touch(l, date_ord)

        self.h2h[(w, l)] += 1
        if surface:
            self.h2h_surf[(w, l, surface)] += 1

        sw, sl = self.players[w], self.players[l]
        # Opponent strength is read from Elo *before* this match updates it,
        # so it reflects what was known going in.
        w_pre, l_pre = self.elo.get(w, "overall"), self.elo.get(l, "overall")
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
            st.last_date = date_ord
            st.dates.append(date_ord)
            st.minutes.append(row["minutes"] if not pd.isna(row["minutes"]) else np.nan)

        # Serve/return rates need real counters; skip when absent.
        if row["has_stats"] == 1:
            self._push_stats(sw, sl, row, w_side=True)
            self._push_stats(sl, sw, row, w_side=False)
            for st_ in (sw, sl):
                for k in EWMA_HALFLIVES:
                    if st_.spw:
                        st_.ew_spw[k].push(st_.spw[-1])
                    if st_.rpw:
                        st_.ew_rpw[k].push(st_.rpw[-1])

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
            if row["tiebreaks"]:
                # Attributed at match level, not per tiebreak: the source gives
                # tiebreak counts but not who won each one.
                sw.tb_w += 1
                sw.tb_n += 1
                sl.tb_n += 1

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
        if svgms and svgms > 0:
            # break points converted against us is the usual proxy for breaks
            breaks_lost = max((bpf or 0) - (bps or 0), 0)
            me.hold.append(max(0.0, 1.0 - breaks_lost / svgms))
        if o_svgms and o_svgms > 0:
            breaks_made = max((o_bpf or 0) - (o_bps or 0), 0)
            me.brk.append(min(1.0, breaks_made / o_svgms))

    # -- persistence -------------------------------------------------------
    def save(self, path=None) -> None:
        path = path or (ARTIFACTS / "feature_state.pkl")
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path=None) -> "FeatureEngine":
        path = path or (ARTIFACTS / "feature_state.pkl")
        with open(path, "rb") as fh:
            return pickle.load(fh)


def _diff(a, b):
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return np.nan
    return a - b


REQUIRED = [
    "match_id", "seq", "tourney_date", "winner_id", "loser_id", "surface",
    "level", "is_challenger", "status", "best_of", "round_idx", "draw_size",
    "indoor", "minutes", "has_stats", "totals_usable", "total_games",
    "total_sets", "game_margin", "winner_rank", "loser_rank",
    "winner_rank_points", "loser_rank_points", "winner_age", "loser_age",
    "winner_ht", "loser_ht", "winner_seed", "loser_seed", "winner_hand",
    "loser_hand", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_SvGms",
    "w_bpSaved", "w_bpFaced", "w_ace", "w_df", "l_svpt", "l_1stIn", "l_1stWon",
    "l_2ndWon", "l_SvGms", "l_bpSaved", "l_bpFaced", "l_ace", "l_df",
]


def build_features(matches: pd.DataFrame, engine: FeatureEngine | None = None,
                   *, emit_from_seq: int = -1) -> tuple[pd.DataFrame, FeatureEngine]:
    """Walk matches in order, emitting a feature row for each.

    `emit_from_seq` lets an incremental run replay history to warm the state
    without re-emitting rows it already stored.
    """
    engine = engine or FeatureEngine()
    matches = matches.sort_values("seq")
    rows = matches.to_dict("records")
    out = []
    for i, row in enumerate(rows):
        if row["seq"] > emit_from_seq:
            out.append(engine.features_for(row))
        engine.update(row)
        if i % 25000 == 0 and i:
            log.info("  features %d/%d", i, len(rows))
    return pd.DataFrame(out), engine
