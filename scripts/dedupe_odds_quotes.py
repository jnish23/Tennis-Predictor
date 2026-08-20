"""Remove duplicate h2h quotes and close the hole that let them in.

SQLite lets NULL into the columns of an ordinary PRIMARY KEY -- a documented
legacy quirk, not a bug in our schema -- and NULL never equals NULL, so a key
containing one never collides. `odds_quotes` keys on
(te_id, book, market, line, line_unit, side) and stores NULL line/line_unit for
h2h, exactly as the schema comment says it should. The result is that
INSERT OR REPLACE deduplicates every totals and handicap row correctly and
deduplicates *no* h2h row at all: re-fetching a page appends another full copy
of its moneyline quotes.

Measured before this ran: 13,926,170 non-h2h rows for 13,926,170 distinct keys
(perfect), against 5,542,184 h2h rows for 3,084,772 distinct keys -- 2,457,412
redundant rows, up to 3 copies of the same quote. All 2,455,308 duplicate
groups held identical prices, so collapsing them loses nothing.

The fix is a unique index over the same columns with the NULLs folded to
sentinels, which INSERT OR REPLACE honours like any other uniqueness
constraint. It costs an index rather than rewriting a 19M-row table, which
matters while the backfill is mid-flight.

Deletes in batches so no single transaction holds the write lock long enough
for a yielding backfill to give up on us. Idempotent: safe to re-run.

Run:  python scripts/dedupe_odds_quotes.py [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tennis.db.lock import exclusive_write  # noqa: E402
from tennis.db.schema import connect  # noqa: E402

BATCH = 200_000

# Fold the NULLs to values that cannot occur naturally: no real handicap or
# total sits at -9999, and no real unit is the empty string.
UNIQUE_INDEX = """
CREATE UNIQUE INDEX IF NOT EXISTS odds_quotes_logical_key
ON odds_quotes(te_id, book, market, COALESCE(line, -9999.0),
               COALESCE(line_unit, ''), side)
"""

# Keep the lowest rowid in each group; the copies are identical, so which one
# survives is arbitrary, but picking deterministically makes re-runs stable.
FIND_DUPES = """
SELECT rowid FROM odds_quotes
WHERE rowid NOT IN (
    SELECT MIN(rowid) FROM odds_quotes
    GROUP BY te_id, book, market, COALESCE(line, -9999.0),
             COALESCE(line_unit, ''), side)
LIMIT ?
"""


def counts(con) -> tuple[int, int]:
    total = con.execute("SELECT COUNT(*) FROM odds_quotes").fetchone()[0]
    uniq = con.execute(
        "SELECT COUNT(*) FROM (SELECT DISTINCT te_id, book, market, "
        "COALESCE(line, -9999.0), COALESCE(line_unit, ''), side "
        "FROM odds_quotes)").fetchone()[0]
    return total, uniq


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be removed and change nothing")
    a = ap.parse_args()

    con = connect()
    con.execute("PRAGMA busy_timeout=60000")
    total, uniq = counts(con)
    print(f"odds_quotes: {total:,} rows, {uniq:,} distinct logical keys, "
          f"{total - uniq:,} redundant")
    if total == uniq:
        print("nothing to remove")
    elif a.dry_run:
        print("dry run: no changes made")
        con.close()
        return

    if total != uniq:
        # Claim the database so the backfill yields at its next batch boundary
        # rather than fighting us for the write lock.
        with exclusive_write("dedupe odds_quotes"):
            removed, t0 = 0, time.time()
            while True:
                rows = [r[0] for r in con.execute(FIND_DUPES, (BATCH,))]
                if not rows:
                    break
                con.execute("BEGIN IMMEDIATE")
                try:
                    con.executemany("DELETE FROM odds_quotes WHERE rowid=?",
                                    [(r,) for r in rows])
                    con.commit()
                except Exception:
                    con.rollback()
                    raise
                removed += len(rows)
                print(f"  removed {removed:,} in {time.time() - t0:.0f}s")
            print(f"removed {removed:,} duplicate rows")

    now_total, now_uniq = counts(con)
    if now_total != now_uniq:
        raise RuntimeError(
            f"still {now_total - now_uniq:,} duplicates after dedupe; "
            "not creating the unique index")

    if a.dry_run:
        print("dry run: index not created")
    else:
        print("creating unique index ...")
        t0 = time.time()
        con.execute(UNIQUE_INDEX)
        con.commit()
        print(f"index created in {time.time() - t0:.0f}s")
    print(f"final: {now_total:,} rows, {now_uniq:,} distinct keys")
    con.close()


if __name__ == "__main__":
    main()
