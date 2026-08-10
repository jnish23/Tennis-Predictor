"""Build (or incrementally extend) the feature table from the match table."""
from __future__ import annotations

import logging
import time

import pandas as pd

from tennis.config import ARTIFACTS
from tennis.db.schema import connect, get_state, set_state
from tennis.features.build import REQUIRED, FeatureEngine, build_features

log = logging.getLogger(__name__)

FEATURES_PATH = ARTIFACTS / "features.parquet"
STATE_PATH = ARTIFACTS / "feature_state.pkl"

MATCH_SQL = """
SELECT m.*, t.surface, t.level, t.is_challenger, t.indoor, t.draw_size
FROM matches m JOIN tournaments t USING(tourney_key)
ORDER BY m.seq
"""

# Labels carried alongside the features. y_win is produced by the engine (it
# depends on p1/p2 orientation); these are the totals/spread targets.
LABEL_SQL = """
SELECT match_id, total_games, total_sets, game_margin, totals_usable, status,
       best_of, minutes
FROM matches
"""


def load_matches(con) -> pd.DataFrame:
    df = pd.read_sql(MATCH_SQL, con)
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


def run(full: bool = True) -> dict:
    """Full rebuild, or incremental extension from the saved engine state."""
    t0 = time.time()
    con = connect()
    matches = load_matches(con)

    engine = None
    emit_from = -1
    prior = None
    if not full and STATE_PATH.exists() and FEATURES_PATH.exists():
        engine = FeatureEngine.load(STATE_PATH)
        emit_from = engine.last_seq
        prior = pd.read_parquet(FEATURES_PATH)
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
    feats.to_parquet(FEATURES_PATH, index=False)
    engine.save(STATE_PATH)

    # Mirror Elo into the DB so the dashboard can read current ratings.
    rows = engine.elo.to_rows(int(engine.last_seq))
    con.execute("DELETE FROM elo_state")
    con.executemany(
        "INSERT INTO elo_state(player_id,scope,rating,n_matches,last_seq) "
        "VALUES(?,?,?,?,?)", rows,
    )
    set_state(con, "features_last_seq", int(engine.last_seq))
    con.commit()
    con.close()

    return {
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
    for k, v in run(full="--incremental" not in sys.argv).items():
        print(f"{k}: {v}")
