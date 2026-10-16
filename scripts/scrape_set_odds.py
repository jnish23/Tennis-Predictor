"""Run the OddsPortal set-price scrape over a list of events, then resolve.

Resumable (failed units retry automatically): finished (league, season, period) units are skipped, so Ctrl-C and
re-run at will. Stops after MAX_CONSECUTIVE_ERRORS failures in a row, because
that pattern means the site is throttling this session and hammering on would
only prolong it. Re-run later; nothing is lost.

    python scripts/scrape_set_odds.py                       # default pilot list
    python scripts/scrape_set_odds.py atp-madrid:2024 atp-rome:2024 --periods 1,2

League slugs are OddsPortal's (the part of the results URL after the country,
e.g. .../tennis/china/atp-shanghai-2024/results/ -> atp-shanghai). A wrong slug
just logs an error. Expect ~30 s per match: a Masters is ~1 hour per period.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tennis.db.schema import init_db  # noqa: E402
from tennis.ingest import odds_oddsportal as op  # noqa: E402

DEFAULT_EVENTS = ["atp-shanghai:2024", "atp-madrid:2024", "atp-rome:2024",
                  "atp-cincinnati:2024"]
MAX_CONSECUTIVE_ERRORS = 3
PAUSE_BETWEEN_EVENTS = 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("events", nargs="*", help="league:season, e.g. atp-madrid:2024")
    ap.add_argument("--periods", default="1", help="comma list of 1,2")
    ap.add_argument("--preview", action="store_true",
                    help="visible books only (fast, usually a single book)")
    a = ap.parse_args()
    events = [e.split(":") for e in (a.events or DEFAULT_EVENTS)]
    periods = [int(p) for p in a.periods.split(",")]

    errors = 0
    for league, season in events:
        for p in periods:
            r = op.scrape_event(league, season, p, full=not a.preview)
            print(r, flush=True)
            if r.get("status") == "error":
                errors += 1
                if errors >= MAX_CONSECUTIVE_ERRORS:
                    print("\nStopping: repeated failures -- the site is likely "
                          "throttling. Try again later.")
                    return finish()
            elif r.get("status") == "ok":
                errors = 0
            time.sleep(PAUSE_BETWEEN_EVENTS)
    finish()


def finish():
    print("\nresolve:", op.resolve_matches())
    for k, v in op.status().items():
        print(f"  {k}: {v}")
    print("\nnext: python scripts/backtest_set_markets.py --period 1")


if __name__ == "__main__":
    init_db().close()
    main()
