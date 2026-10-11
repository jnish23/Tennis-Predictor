"""Pipeline entrypoints for the scheduled cloud jobs.

    python -m tennis.ops.cloud daily  --out dist/
    python -m tennis.ops.cloud weekly --out dist/
    python -m tennis.ops.cloud odds   --out dist/

`daily` rebuilds the whole site from the raw files and `weekly` additionally
re-runs the walk-forward backtest and refits the production models. Both end by
packing the serve and state bundles into `--out`; the workflow uploads them.
The jobs do not touch GitHub themselves, so they run the same way on a laptop.

Features are rebuilt in full every run rather than extended. The incremental
path is correct now (it keys on match_id), but a full rebuild needs no saved
engine state, is deterministic, and picks up upstream corrections to matches
already folded in. Costs about a minute and a half of compute.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import time
from datetime import date, datetime, timezone
from pathlib import Path

from tennis.config import ARTIFACTS, DATA, RAW_ONGOING_DATED, RAW_TML, TOURS, ONGOING_FILES
from tennis.db.schema import connect, init_db, set_state
from tennis.ops import bundle

log = logging.getLogger(__name__)

DRAW_LOOKAHEAD_DAYS = 3


def _step(report: dict, name: str):
    """Context manager recording wall time per step."""
    class _S:
        def __enter__(self_):
            self_.t = time.time()
            log.info("== %s", name)
        def __exit__(self_, *exc):
            report.setdefault("seconds_by_step", {})[name] = round(time.time() - self_.t, 1)
    return _S()


def save_dated_ongoing(today: date | None = None) -> list[Path]:
    """Copy each ongoing file under a dated name.

    The live files are overwritten in place and a tournament leaves them before
    the yearly file catches up; the per-tournament archive covers the data, and
    this keeps every file as pulled (CLAUDE.md: cache every CSV).
    """
    stamp = (today or date.today()).strftime("%Y%m%d")
    out = []
    for name in ONGOING_FILES:
        src = RAW_TML / name
        if src.exists():
            dst = RAW_ONGOING_DATED / f"{stamp}_{name}"
            shutil.copy2(src, dst)
            out.append(dst)
    return out


def prefetch_draws(report: dict) -> None:
    """Fetch draw sheets for events starting within a few days.

    The hosted dashboard serves these from the bundle and never calls the
    provider, so this is the only place quota is spent. Skipped without a key.
    """
    if not os.getenv("RAPIDAPI_KEY"):
        report["draws"] = "skipped: no RAPIDAPI_KEY"
        return
    from tennis.ingest import draws_api as d

    got = {}
    year = date.today().year
    for tour in TOURS:
        try:
            events = d.discover_tournaments(days=DRAW_LOOKAHEAD_DAYS, tour=tour)
        except d.DrawFeedError as exc:
            got[tour] = f"error: {exc}"
            continue
        n = 0
        for tid in (events["tournament_id"].tolist() if len(events) else []):
            try:
                d.tournament_draw(tid, year, tour=tour)
                d.tournament_fixtures(tid, tour=tour)
                n += 1
            except d.DrawFeedError as exc:
                log.warning("draw %s/%s: %s", tour, tid, exc)
                break                      # budget or auth; no point continuing
        got[tour] = n
    report["draws"] = got


def run(kind: str = "daily", *, offline: bool = False, out: Path | None = None) -> dict:
    if kind not in ("daily", "weekly"):
        raise ValueError(kind)
    t0 = time.time()
    if kind == "daily" and not all((ARTIFACTS / f).exists() for f in
                                   ("models_pooled.pkl", "backtest_pooled.parquet")):
        # Nothing to carry forward (first run, or the state was lost): the
        # daily path would have no models or backtest to report on.
        log.warning("no models/backtest in state; running the weekly path instead")
        kind = "weekly"
    report: dict = {"kind": kind, "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "offline": offline}

    # Restore the rows that live outside the database between runs. A fresh
    # runner has no database, so build it before anything else reads it.
    con = init_db()
    report["snapshots_restored"] = bundle.import_snapshots(con, DATA / bundle.SNAPSHOTS_FILE)
    report["odds_extra_restored"] = bundle.import_odds_extra(con, DATA / bundle.ODDS_EXTRA_FILE)
    con.close()

    if not offline:
        from tennis.ingest.download import sync_tennisdata, sync_tennismylife
        with _step(report, "fetch"):
            fetched = sync_tennismylife(ongoing_only=(kind == "daily"))
            report["files_fetched"] = len(fetched)
            save_dated_ongoing()
            this_year = date.today().year
            sync_tennisdata(range(this_year, this_year + 1))

    from tennis.features.pipeline import run as build_features
    from tennis.ingest.load import load_all
    from tennis.ingest.odds import join_odds
    from tennis.ingest.odds_live import resolve_snapshots
    from tennis.models import evaluate

    with _step(report, "load"):
        report["load"] = load_all()
    with _step(report, "odds"):
        con = connect()
        _, report["odds_join"] = join_odds(con)
        report["snapshots_resolved"] = resolve_snapshots(con)
        con.close()
    with _step(report, "features"):
        report["features"] = {t: build_features(full=True, tour=t) for t in TOURS}

    if kind == "weekly":
        from tennis.models.train import run_backtest, train_production_models
        with _step(report, "backtest"):
            report["backtest"] = run_backtest()
        with _step(report, "train"):
            report["production"] = {"n_train": train_production_models()["n_train"]}

    with _step(report, "reports"):
        report["reports"] = {t: evaluate.full_report(tour=t)["n_matches"] for t in TOURS}

    if not offline:
        with _step(report, "draws"):
            prefetch_draws(report)

    con = connect()
    max_date = con.execute("SELECT MAX(tourney_date) FROM matches").fetchone()[0]
    set_state(con, "last_daily_run", report["started"])
    con.commit()
    con.close()
    report["data_through"] = max_date
    report["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report["seconds"] = round(time.time() - t0, 1)
    (ARTIFACTS / "last_daily_run.json").write_text(json.dumps(report, indent=2, default=str))

    if out is not None:
        out = Path(out)
        with _step(report, "pack"):
            report["serve_bundle"] = bundle.pack_serve(out / "serve.tar.zst")
            report["state_bundle"] = bundle.pack_state(out / "state.tar.zst")
        (ARTIFACTS / "last_daily_run.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def odds(*, out: Path | None = None, capture_fn=None) -> dict:
    """Capture today's prices and append them to the snapshots parquet.

    Runs against a scratch database seeded from the parquet, so the job needs
    nothing but that one file. `capture_fn` exists for tests.
    """
    from tennis.ingest.odds_live import capture

    con = init_db()
    n0 = bundle.import_snapshots(con, DATA / bundle.SNAPSHOTS_FILE)
    con.close()
    res = (capture_fn or capture)()
    con = connect()
    n1 = bundle.export_snapshots(con, Path(out) / bundle.SNAPSHOTS_FILE
                                 if out else DATA / bundle.SNAPSHOTS_FILE)
    con.close()
    return {"restored": n0, "total": n1, "capture": res}


def main() -> None:
    ap = argparse.ArgumentParser(description="Cloud pipeline entrypoints")
    ap.add_argument("kind", choices=["daily", "weekly", "odds"])
    ap.add_argument("--out", type=Path, help="directory for the bundles / parquet")
    ap.add_argument("--offline", action="store_true",
                    help="skip downloads and the draw fetch (rebuild from what is on disk)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    res = odds(out=args.out) if args.kind == "odds" else run(args.kind, offline=args.offline, out=args.out)
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
