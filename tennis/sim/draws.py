"""Recover tournament draws, and replay past events against the simulator.

Two ways a draw reaches the simulator:

* **Past events** -- reconstructed from our own match table. The first round of a
  completed tournament *is* the draw, and who-beat-whom recovers the bracket
  tree exactly, so no external source and no manual entry is needed. Validated
  at 99.3% across 5,158 events; the rest are malformed upstream (one 32-draw
  with 55 players) or team events with no bracket.
* **Upcoming events** -- see tennis/ingest/draws_api.py, which needs a feed,
  because no source we hold publishes a bracket before it is played.

`match_num` is deliberately not used to recover bracket order. It encodes the
bracket in some files and not others (it held for only 712 of 5,662 events), so
the result graph is the reliable signal.

Replaying a past event must use feature state from *before* it started. Today's
Elo already contains that tournament's results, and simulating with it would be
the same leakage class CLAUDE.md flags as a standing risk.
"""
from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd

from tennis.db.schema import connect
from tennis.models.predict import MatchContext
from tennis.sim.bracket import BYE, Draw

log = logging.getLogger(__name__)

# Team events and round-robins have no single-elimination bracket.
EXCLUDED_LEVELS = ("davis_cup", "finals")
EXCLUDED_ROUNDS = ("RR", "BR", "Q1", "Q2", "Q3")

TOURNEY_SQL = f"""
SELECT m.tourney_key, m.round, m.round_idx, m.match_num, m.winner_id, m.loser_id,
       m.seq, m.status, m.score, m.best_of, m.total_games, m.game_margin, m.totals_usable,
       t.name, t.surface, t.level, t.indoor, t.draw_size, t.season, t.tourney_date,
       t.is_challenger
FROM matches m JOIN tournaments t USING(tourney_key)
WHERE t.level NOT IN {EXCLUDED_LEVELS}
  AND m.round NOT IN {EXCLUDED_ROUNDS}
"""


def _expand(player: str, ridx: int, rounds: list, won: dict, order: dict) -> list:
    """Slots of the sub-bracket whose winner is `player` after round `ridx`."""
    pos = order[ridx]
    size = 2 ** (pos + 1)
    m = won.get((ridx, player))
    if m is None:
        # Player advanced without playing: a bye occupies the other half.
        sub = _expand(player, rounds[pos - 1], rounds, won, order) if pos > 0 else [player]
        out = sub + [BYE] * (size - len(sub))
    elif pos == 0:
        out = [m[0], m[1]]
    else:
        prev = rounds[pos - 1]
        out = (_expand(m[0], prev, rounds, won, order)
               + _expand(m[1], prev, rounds, won, order))
    return (out + [BYE] * size)[:size]


def reconstruct_slots(g: pd.DataFrame) -> list[str] | None:
    """Bracket-ordered player ids for one tournament, or None if not recoverable."""
    g = g.dropna(subset=["round_idx"])
    rounds = sorted(g["round_idx"].unique())
    if len(rounds) < 2:
        return None
    final = g[g["round_idx"] == rounds[-1]]
    if len(final) != 1:
        return None  # team event or duplicated final

    won = {(r.round_idx, r.winner_id): (r.winner_id, r.loser_id) for r in g.itertuples()}
    order = {v: i for i, v in enumerate(rounds)}
    champ = final.iloc[0]["winner_id"]

    old = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old, 10_000))
    try:
        slots = _expand(champ, rounds[-1], rounds, won, order)
    except RecursionError:
        return None
    finally:
        sys.setrecursionlimit(old)

    played = set(g["winner_id"]) | set(g["loser_id"])
    got = [s for s in slots if s != BYE]
    if len(got) != len(set(got)) or set(got) != played:
        return None  # inconsistent source data
    return slots


def load_tournament(con, tourney_key: str) -> pd.DataFrame:
    return pd.read_sql(TOURNEY_SQL + " AND m.tourney_key = ?", con, params=(tourney_key,))


def build_draw(con, tourney_key: str) -> tuple[Draw, pd.DataFrame] | None:
    """Reconstruct a past tournament as a simulator-ready Draw."""
    g = load_tournament(con, tourney_key)
    if g.empty:
        return None
    slots = reconstruct_slots(g)
    if slots is None:
        return None

    meta = g.iloc[0]
    names = pd.read_sql(
        "SELECT player_id, name FROM players WHERE player_id IN (%s)"
        % ",".join("?" * len({s for s in slots if s != BYE})),
        con, params=[s for s in dict.fromkeys(slots) if s != BYE],
    ).set_index("player_id")["name"].to_dict()

    bo = pd.to_numeric(g.get("best_of"), errors="coerce").dropna()
    ctx = MatchContext(
        surface=meta["surface"] if isinstance(meta["surface"], str) else "Hard",
        level=meta["level"],
        best_of=int(bo.mode().iloc[0]) if len(bo) else 3,
        indoor=float(meta["indoor"]) if pd.notna(meta["indoor"]) else 0.0,
        draw_size=float(len(slots)),
        tourney_date=int(meta["tourney_date"]),
        is_challenger=int(meta["is_challenger"]),
    )
    draw = Draw(name=str(meta["name"]), slots=slots, ctx=ctx, player_names=names)
    return draw, g


def list_replayable(con, min_season: int = 2010, limit: int | None = None) -> pd.DataFrame:
    """Every past tournament whose bracket can be recovered, newest first."""
    g = pd.read_sql(TOURNEY_SQL + " AND t.season >= ?", con, params=(min_season,))
    rows = []
    for tk, sub in g.groupby("tourney_key"):
        slots = reconstruct_slots(sub)
        if slots is None:
            continue
        m = sub.iloc[0]
        rows.append({
            "tourney_key": tk, "name": m["name"], "season": int(m["season"]),
            "tourney_date": int(m["tourney_date"]), "surface": m["surface"],
            "level": m["level"], "draw_size": len(slots),
            "n_players": len([s for s in slots if s != BYE]),
            "min_seq": int(sub["seq"].min()),
            "label": f"{int(m['season'])} · {m['name']} · {m['level']} · "
                     f"{m['surface']} · {len(slots)} draw",
        })
    out = pd.DataFrame(rows).sort_values("tourney_date", ascending=False)
    return out.head(limit) if limit else out


# --------------------------------------------------------------------------
# actual results, for comparison against a simulation
# --------------------------------------------------------------------------
def actual_progression(g: pd.DataFrame, slots: list[str]) -> pd.DataFrame:
    """How far each player actually got, as reach-probabilities of 0 or 1.

    Shaped to line up column-for-column with `bracket.simulate` output so the
    two can be joined directly.
    """
    from tennis.sim.bracket import round_names

    n_rounds = int(np.log2(len(slots)))
    labels = round_names(n_rounds)
    rounds = sorted(g["round_idx"].dropna().unique())

    players = [s for s in dict.fromkeys(slots) if s != BYE]
    rec = {p: {lab: 0.0 for lab in labels} for p in players}

    # `simulate` counts a player as "reaching" a label when they advance *out*
    # of that round. The field of the next round is exactly that set, which also
    # credits byes for free -- a player who skipped a round still shows up in
    # the next one without ever appearing as a winner.
    for i, ridx in enumerate(rounds[:n_rounds]):
        if i + 1 < len(rounds):
            nxt = g[g["round_idx"] == rounds[i + 1]]
            advanced = set(nxt["winner_id"]) | set(nxt["loser_id"])
        else:
            advanced = set(g.loc[g["round_idx"] == ridx, "winner_id"])  # champion
        for p in advanced:
            if p in rec:
                rec[p][labels[i]] = 1.0

    return pd.DataFrame([{"player_id": p, **rec[p]} for p in players])


def compare(sim: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    """Join simulated reach-probabilities against what actually happened."""
    labels = [c for c in sim.columns if c not in ("player_id", "player_name")]
    s = sim.melt(id_vars=["player_id", "player_name"], value_vars=labels,
                 var_name="round", value_name="sim_prob")
    a = actual.melt(id_vars=["player_id"], value_vars=[c for c in labels
                                                       if c in actual.columns],
                    var_name="round", value_name="actual")
    out = s.merge(a, on=["player_id", "round"], how="left")
    out["actual"] = out["actual"].fillna(0.0)
    out["surprise"] = out["actual"] - out["sim_prob"]
    return out


def calibration_of_replay(cmp: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Were the simulation's stated probabilities honest across many events?"""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(cmp["sim_prob"], edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        rows.append({
            "bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}",
            "n": int(m.sum()),
            "predicted": round(float(cmp.loc[m, "sim_prob"].mean()), 4),
            "actual": round(float(cmp.loc[m, "actual"].mean()), 4),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# pre-tournament feature state
# --------------------------------------------------------------------------
def engine_as_of(seq: int):
    """FeatureEngine holding only what was known before match `seq`.

    Replays history rather than reusing the live engine: the saved state has
    already absorbed the tournament we are about to simulate, and using it would
    leak the result into the prediction.
    """
    from tennis.features.build import FeatureEngine, build_features
    from tennis.features.pipeline import load_matches

    con = connect()
    m = load_matches(con)
    con.close()
    m = m[m["seq"] < seq]
    _, engine = build_features(m, FeatureEngine(), emit_from_seq=10 ** 12)
    return engine


# --------------------------------------------------------------------------
# conditional simulation: fix the rounds already played, simulate the rest
# --------------------------------------------------------------------------
def walk_bracket(draw: Draw, g: pd.DataFrame, through_round: int | None = None):
    """Advance the bracket using real results.

    Returns ``(resolved, field, completed_rounds)`` where `resolved` is the
    ``{round: {pair: winner_id}}`` map `bracket.simulate` accepts, `field` is who
    stands in each slot after the last fully-known round, and `completed_rounds`
    counts how many rounds are fully decided.

    Stops at the first round that is not fully played: once one match of a round
    is outstanding, the *next* round's pairings are unknown, so nothing beyond it
    can be pinned.
    """
    wins = {(r.winner_id, r.loser_id) for r in g.itertuples()}
    cur = list(draw.slots)
    resolved: dict[int, dict[int, str]] = {}
    completed = 0

    for r in range(draw.n_rounds):
        if through_round is not None and r >= through_round:
            break
        nxt, this_round = [], {}
        for i in range(0, len(cur), 2):
            a, b = cur[i], cur[i + 1]
            if a is None or b is None:
                w = None
            elif a == BYE and b == BYE:
                w = BYE
            elif b == BYE:
                w = a
            elif a == BYE:
                w = b
            elif (a, b) in wins:
                w = a
            elif (b, a) in wins:
                w = b
            else:
                w = None                      # fixture exists but not played yet
            if w not in (None, BYE):
                this_round[i // 2] = w
            nxt.append(w)
        resolved[r] = this_round
        if any(x is None for x in nxt):
            break                             # round incomplete: stop here
        cur = nxt
        completed = r + 1
    return resolved, cur, completed


def round_start_seq(g: pd.DataFrame, round_index: int) -> int | None:
    """`seq` of the first match of the Nth round, for rebuilding feature state."""
    rounds = sorted(g["round_idx"].dropna().unique())
    if round_index <= 0 or round_index >= len(rounds):
        return None
    sub = g[g["round_idx"] == rounds[round_index]]
    return int(sub["seq"].min()) if len(sub) else None


def replay_from_round(con, tourney_key: str, from_round: int = 0):
    """Everything needed to simulate a past event from a chosen round.

    Feature state is rebuilt to *just before* that round, so a round-2 simulation
    sees each player's round-1 result in their form and Elo. That matters: recent
    form is the second-largest permutation-importance family, and a bracket run
    entirely off pre-tournament features ignores what just happened on court.
    """
    built = build_draw(con, tourney_key)
    if built is None:
        return None
    draw, g = built
    resolved, _, completed = walk_bracket(draw, g, through_round=from_round)
    seq = round_start_seq(g, from_round)
    return {"draw": draw, "matches": g, "resolved": resolved,
            "completed_rounds": completed, "state_seq": seq,
            "n_rounds": draw.n_rounds}


def score_lookup(g: pd.DataFrame) -> dict:
    """{(winner_id, loser_id): score} for a tournament's played matches."""
    if g.empty or "score" not in g.columns:
        return {}
    return {(r.winner_id, r.loser_id): r.score
            for r in g.itertuples() if isinstance(getattr(r, "score", None), str)}
