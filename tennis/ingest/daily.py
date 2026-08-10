"""Daily ingestion job.

Pulls the ongoing-tournament files and the current-season odds workbook,
refreshes the match table, and extends the feature table incrementally.

Scheduling note: the ongoing files carry *completed* matches only -- they hold
no unplayed fixtures (verified: zero rows without a score or winner). They can
tell us a tournament has started, but they cannot supply a bracket in advance,
which is why draw entry is manual. See tennis/sim/bracket.py.

Retraining is deliberately not daily. Feature state updates every day; models
are refit on a slower cadence via `--retrain`, since one day of matches cannot
move a model fit on 200k rows.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone

from tennis.config import ARTIFACTS
from tennis.db.schema import connect, set_state
from tennis.ingest.download import sync_tennisdata, sync_tennismylife
from tennis.ingest.load import load_all
from tennis.ingest.odds import join_odds

log = logging.getLogger(__name__)


def run_daily(*, retrain: bool = False, full_history: bool = False) -> dict:
    t0 = time.time()
    report = {"started": datetime.now(timezone.utc).isoformat()}

    # 1. fetch (every file is cached locally as it lands)
    fetched = sync_tennismylife(ongoing_only=not full_history)
    report["files_fetched"] = [p.name for p in fetched]
    this_year = datetime.now(timezone.utc).year
    odds_files = sync_tennisdata(range(this_year, this_year + 1))
    report["odds_files"] = [p.name for p in odds_files]

    # 2. rebuild the match table.
    # Full reload rather than an upsert: it takes ~25s for 200k rows, and it is
    # the only way corrections to already-published rows upstream (which do
    # happen -- scores get fixed, stats get backfilled) actually reach us.
    report["load"] = load_all()

    # 3. odds join + exact match dates
    con = connect()
    _, odds_stats = join_odds(con)
    report["odds"] = odds_stats

    # 3b. Live prices. `capture` must run every time the job does -- it records
    # the market as it stands right now, and a moment not captured is gone for
    # good. `resolve_snapshots` runs after the match load above, so fixtures
    # priced on an earlier run and played since can now be matched to a result.
    from tennis.ingest.odds_live import capture, resolve_snapshots
    try:
        report["odds_live"] = capture()
        report["odds_live_resolved"] = resolve_snapshots(con)
    except Exception as exc:                 # never let a scrape stop the job
        log.warning("live odds capture failed: %s", exc)
        report["odds_live"] = {"error": str(exc)}

    # 4. features, incrementally from saved engine state
    from tennis.features.pipeline import run as build_feats
    report["features"] = build_feats(full=False)

    if retrain:
        from tennis.models.train import run_backtest, train_production_models
        report["backtest"] = run_backtest()
        report["production"] = {"n_train": train_production_models()["n_train"]}

    set_state(con, "last_daily_run", report["started"])
    con.commit()
    con.close()

    report["seconds"] = round(time.time() - t0, 1)
    (ARTIFACTS / "last_daily_run.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Daily tennis data ingestion")
    ap.add_argument("--retrain", action="store_true",
                    help="also refit models and rerun the walk-forward backtest")
    ap.add_argument("--full-history", action="store_true",
                    help="re-check every historical file, not just ongoing ones")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(json.dumps(run_daily(retrain=args.retrain,
                               full_history=args.full_history), indent=2, default=str))


if __name__ == "__main__":
    main()
