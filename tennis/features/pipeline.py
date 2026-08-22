"""Build (or incrementally extend) the feature table from the match table.

**One engine per tour, and separate artifacts.** The ATP and WTA never play
each other, so every rating in here -- Elo, form, serve and return rates,
opponent strength -- is meaningful only within its own tour. Running one engine
over both would not corrupt player state, since the two ID namespaces are
disjoint, but it would silently pool anything global the engine ever grows, and
it leaves the two tours sharing one artifact that nothing downstream can
separate. Splitting at the source is cheaper than auditing every statistic
forever, and it gives the dashboard the per-tour files its selector needs.
"""
from __future__ import annotations

import logging
import time

import pandas as pd

from tennis.config import ARTIFACTS
from tennis.db.schema import connect, get_state, set_state
from tennis.features.build import REQUIRED, FeatureEngine, build_features

log = logging.getLogger(__name__)

def features_path(tour: str = "atp"):
    return ARTIFACTS / f"features_{tour}.parquet"


def state_path(tour: str = "atp"):
    return ARTIFACTS / f"feature_state_{tour}.pkl"


# Kept so the ATP research scripts keep resolving; new code should call
# features_path(tour) rather than assume a tour.
FEATURES_PATH = features_path("atp")
STATE_PATH = state_path("atp")

MATCH_SQL = """
SELECT m.*, t.surface, t.level, t.is_challenger, t.indoor, t.draw_size
FROM matches m JOIN tournaments t USING(tourney_key)
WHERE m.tour = ?
ORDER BY m.seq
"""

# Labels carried alongside the features. y_win is produced by the engine (it
# depends on p1/p2 orientation); these are the totals/spread targets.
LABEL_SQL = """
SELECT match_id, total_games, total_sets, game_margin, totals_usable, status,
       best_of, minutes
FROM matches
"""


def load_matches(con, tour: str = "atp") -> pd.DataFrame:
    df = pd.read_sql(MATCH_SQL, con, params=(tour,))
    if df.empty:
        raise RuntimeError(f"no matches for tour {tour!r}")
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise RuntimeError(f"match frame missing columns: {missing}")
    return df


def _attach_labels(feats: pd.DataFrame, con) -> pd.DataFrame:
    labels = pd.read_sql(LABEL_SQL, con)
    out = feats.merge(labels, on="match_id", how="left")
    # Spread is expressed p1-relative: positive means p1 won more games.
    out["y_total_games"] = out["total_games"].where(out["totals_usable"] == 1)
    out["y_total_sets"] = out["total_sets"].where(out["totals_usable"] == 1)
    signed = out["game_margin"].where(out["totals_usable"] == 1)
    out["y_spread"] = signed * (out["y_win"] * 2 - 1)
    return out


def run(full: bool = True, tour: str = "atp") -> dict:
    """Full rebuild, or incremental extension from the saved engine state."""
    t0 = time.time()
    features_p, state_p = features_path(tour), state_path(tour)
    con = connect()
    matches = load_matches(con, tour)

    engine = None
    emit_from = -1
    prior = None
    if not full and state_p.exists() and features_p.exists():
        engine = FeatureEngine.load(state_p)
        emit_from = engine.last_seq
        prior = pd.read_parquet(features_p)
        # Replay only what the engine has not already folded in.
        matches = matches[matches["seq"] > engine.last_seq]
        log.info("incremental: %d new matches after seq %d", len(matches), emit_from)
        if matches.empty:
            con.close()
            return {"new_rows": 0, "total_rows": len(prior), "seconds": 0.0}

    feats, engine = build_features(matches, engine, emit_from_seq=emit_from)
    feats = _attach_labels(feats, con)

    if prior is not None and len(prior):
        feats = pd.concat([prior, feats], ignore_index=True)
        feats = feats.drop_duplicates("match_id", keep="last")
    feats = feats.sort_values("seq").reset_index(drop=True)
    feats["tour"] = tour
    feats.to_parquet(features_p, index=False)
    engine.save(state_p)

    # Mirror Elo into the DB so the dashboard can read current ratings. Scoped
    # by tour: this clears only this tour's rows, so rebuilding one tour never
    # wipes the other's ratings.
    rows = [(pid, scope, rating, n, seq, tour)
            for pid, scope, rating, n, seq in engine.elo.to_rows(int(engine.last_seq))]
    con.execute("DELETE FROM elo_state WHERE tour = ?", (tour,))
    con.executemany(
        "INSERT INTO elo_state(player_id,scope,rating,n_matches,last_seq,tour) "
        "VALUES(?,?,?,?,?,?)", rows,
    )
    set_state(con, f"features_last_seq_{tour}", int(engine.last_seq))
    con.commit()
    con.close()

    return {
        "tour": tour,
        "total_rows": len(feats),
        "n_features": feats.shape[1],
        "seconds": round(time.time() - t0, 1),
        "date_range": (int(feats["tourney_date"].min()), int(feats["tourney_date"].max())),
        "label_balance": round(float(feats["y_win"].mean()), 4),
        "totals_labelled": int(feats["y_total_games"].notna().sum()),
    }


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    full = "--incremental" not in sys.argv
    tours = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--tour=")]
    for t in (tours or ["atp", "wta"]):
        for k, v in run(full=full, tour=t).items():
            print(f"{k}: {v}")
