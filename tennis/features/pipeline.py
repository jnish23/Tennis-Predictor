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
from tennis.features.elo import season_start

log = logging.getLogger(__name__)

SEASON_SCOPE = "season_start"


def features_path(tour: str = "atp"):
    return ARTIFACTS / f"features_{tour}.parquet"


def state_path(tour: str = "atp"):
    return ARTIFACTS / f"feature_state_{tour}.pkl"


# Kept so the ATP research scripts keep resolving; new code should call
# features_path(tour) rather than assume a tour.
FEATURES_PATH = features_path("atp")
STATE_PATH = state_path("atp")

MATCH_SQL = """
SELECT m.*, t.name AS tourney_name, t.surface, t.level, t.is_challenger, t.indoor, t.draw_size
FROM matches m JOIN tournaments t USING(tourney_key)
WHERE m.tour = ?
ORDER BY m.seq
"""

# Labels carried alongside the features, taken from the match frame that is
# already loaded (m.*). y_win is produced by the engine (it depends on p1/p2
# orientation); these are the totals/spread targets. `best_of` and `minutes` are
# deliberately absent: the feature row carries best_of already, and merging a
# second copy produced best_of_x/best_of_y, so the model never saw a best_of
# column at all (it was NaN for every training row).
LABEL_COLS = ["match_id", "total_games", "total_sets", "game_margin",
              "totals_usable", "status", "minutes"]


def load_matches(con, tour: str = "atp") -> pd.DataFrame:
    df = pd.read_sql(MATCH_SQL, con, params=(tour,))
    if df.empty:
        raise RuntimeError(f"no matches for tour {tour!r}")
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise RuntimeError(f"match frame missing columns: {missing}")
    return df


def _attach_labels(feats: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    out = feats.merge(matches[LABEL_COLS], on="match_id", how="left")
    # Spread is expressed p1-relative: positive means p1 won more games.
    out["y_total_games"] = out["total_games"].where(out["totals_usable"] == 1)
    out["y_total_sets"] = out["total_sets"].where(out["totals_usable"] == 1)
    signed = out["game_margin"].where(out["totals_usable"] == 1)
    out["y_spread"] = signed * (out["y_win"] * 2 - 1)
    return out


def _full_rebuild_reason(prior: pd.DataFrame, matches: pd.DataFrame) -> str | None:
    """Why the saved state cannot be extended, or None if it can.

    Incremental correctness needs the engine to have folded in exactly a prefix
    of what the database holds now. `seq` cannot tell us that: the reload
    renumbers it on every run, so "seq > last_seq" silently skipped 1,134
    matches and double-counted others whose ids had moved. Match ids are stable,
    so compare those.
    """
    have = set(matches["match_id"])
    gone = ~prior["match_id"].isin(have)
    if gone.any():
        # Their results are already folded into ratings; they cannot be undone.
        return f"{int(gone.sum())} folded-in matches no longer exist"
    new = matches[~matches["match_id"].isin(set(prior["match_id"]))]
    if len(new) and new["tourney_date"].min() < prior["tourney_date"].max():
        # A match older than state already folded would be rated with the
        # future in view -- exactly the leak this engine exists to prevent.
        return "new matches pre-date the saved state"
    return None


def _renumber(feats: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Bring stored rows' seq (and date) in line with the current numbering."""
    cur = matches.set_index("match_id")
    out = feats.copy()
    out["seq"] = out["match_id"].map(cur["seq"]).astype("int64")
    return out


def run(full: bool = True, tour: str = "atp") -> dict:
    """Full rebuild, or incremental extension from the saved engine state."""
    t0 = time.time()
    features_p, state_p = features_path(tour), state_path(tour)
    con = connect()
    all_matches = matches = load_matches(con, tour)
    # The season the newest match belongs to; the engine snapshots the ratings
    # as they stood on its first day.
    opener = season_start(int(all_matches["tourney_date"].max()))

    engine = None
    prior = None
    if not full and state_p.exists() and features_p.exists():
        prior = pd.read_parquet(features_p)
        reason = _full_rebuild_reason(prior, matches)
        engine = None if reason else FeatureEngine.load(state_p)
        if engine is not None and getattr(engine, "season_snap", None) is None:
            # A state saved before snapshots existed cannot recover the opener.
            reason, engine = "saved state has no season-opener snapshot", None
        if reason:
            log.warning("incremental not possible (%s); rebuilding %s in full",
                        reason, tour)
            prior = None
        else:
            matches = matches[~matches["match_id"].isin(set(prior["match_id"]))]
            log.info("incremental: %d new matches", len(matches))
            if matches.empty:
                # Nothing to fold in, but seq was renumbered by the reload.
                # Rewriting is a 145 MB write, so only do it if seq moved.
                renumbered = _renumber(prior, all_matches)
                if not renumbered["seq"].equals(prior["seq"]):
                    renumbered.to_parquet(features_p, index=False)
                con.close()
                return {"tour": tour, "new_rows": 0, "total_rows": len(prior),
                        "seconds": round(time.time() - t0, 1)}

    feats, engine = build_features(matches, engine, snapshot_start=opener)
    feats = _attach_labels(feats, matches)
    engine.last_seq = int(all_matches["seq"].max())

    if prior is not None and len(prior):
        feats = pd.concat([_renumber(prior, all_matches), feats], ignore_index=True)
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
    # The rating on the first day of the current season, for the dashboard's
    # season-to-date view. Its own scope, so everything filtering on the real
    # scopes ('overall', a surface) never sees it; the date is in pipeline_state.
    snap = engine.season_snap
    rows += [(pid, SEASON_SCOPE, r, 0, int(engine.last_seq), tour)
             for pid, r in snap["ratings"].items()]
    set_state(con, f"elo_season_start_{tour}", int(snap["start"]))
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
