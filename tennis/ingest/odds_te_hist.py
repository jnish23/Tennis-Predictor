"""Historical odds backfill from tennisexplorer match pages.

The find that motivates this: `/match-detail/?id=N` carries, in static HTML,
**over/under and Asian handicap lines on games** as well as sets, plus opening
*and* closing prices per bookmaker with timestamps. Two consequences.

* The totals and spread models have never been measured against a market --
  no source held before this one carried a games line. That is worth more than
  any modelling change currently on the table.
* Open/close per book is real closing-line value, not the 3-hourly proxy
  `odds_live` produces.

**Two tiers, because they cost wildly different amounts.** A results day-page
lists every match with its id and a moneyline, at one request per day: the whole
history is ~7,900 requests. The line markets only exist on the detail page, one
request per match, which is ~200k requests. Tier A is therefore run to
completion first -- it produces the id map everything else needs -- and Tier B
grinds through it afterwards.

**Everything is checkpointed.** A run of this size will be interrupted. Each
day-page and each detail page is recorded in `te_scrape_log` the moment it
lands, writes commit in batches rather than at the end, and a restart skips
whatever is already logged. Killing the process costs at most one batch.

Both tiers walk **newest-first**. Recent seasons carry the opening odds, have
the denser games lines, and match the era the model is scored in, so an
interrupted backfill has already banked the half worth testing on.

Day pages are fetched with `type=all` rather than `type=atp-single` because the
cost is identical -- one request either way -- so WTA match ids are banked now
for free even though WTA detail pages are deferred. `tour_of` tags each row so
the tiers stay separable at query time.

robots.txt (checked 2026-08) disallows `/redirect/`, `/terms-of-use/` and
`/contact/` only. Requests are spaced uniformly around a 2-second mean and every
page is cached to disk, so a re-run of the same range costs nothing.
"""
from __future__ import annotations

import logging
import random
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

from tennis.config import DATA
from tennis.db.schema import connect

log = logging.getLogger(__name__)

BASE = "https://www.tennisexplorer.com"
CACHE = DATA / "raw" / "te"
# Randomised spacing averaging DELAY seconds. A fixed interval is both a
# recognisable signature and needlessly bursty; drawing uniformly either side of
# the mean keeps the long-run rate identical while looking like traffic.
DELAY = 2.0
JITTER = 1.0             # uniform(DELAY-JITTER, DELAY+JITTER)
BATCH = 25               # commit every N detail pages
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120 Safari/537.36")
SKIP_TERMS = ("itf", "futures", "utr", "exhibition")


def tour_of(tournament: str) -> str:
    """atp | wta | challenger, inferred from the tournament name.

    Day pages are fetched with type=all because that costs exactly one request
    either way -- the same as asking for ATP alone. Tagging the tour here means
    WTA ids are banked for free now and its detail pages can be fetched later,
    in any order, without ever re-walking the calendar.
    """
    t = (tournament or "").lower()
    if "wta" in t:
        return "wta"
    if "challenger" in t:
        return "challenger"
    return "atp"

_MARKETS = {1: "h2h", 2: "totals", 3: "handicap", 4: "correct_score"}


def _sleep() -> None:
    time.sleep(random.uniform(DELAY - JITTER, DELAY + JITTER))


def fetch(path: str, *, cache_key: str) -> str:
    """GET with an on-disk cache. Cached pages never hit the network again."""
    sub = CACHE / cache_key[:6]
    sub.mkdir(parents=True, exist_ok=True)
    f = sub / f"{cache_key}.html"
    if f.exists():
        return f.read_text(encoding="utf-8", errors="replace")
    _sleep()
    r = requests.get(f"{BASE}{path}", headers={"User-Agent": UA}, timeout=45)
    r.raise_for_status()
    f.write_text(r.text, encoding="utf-8")
    return r.text


# --------------------------------------------------------------------------
# checkpointing
# --------------------------------------------------------------------------
def _done(con, kind: str) -> set[str]:
    return {r[0] for r in con.execute(
        "SELECT key FROM te_scrape_log WHERE kind=? AND status IN ('ok','empty')",
        (kind,))}


def _mark(con, kind: str, key: str, status: str, n: int = 0, note: str = "") -> None:
    con.execute(
        "INSERT OR REPLACE INTO te_scrape_log(kind,key,status,n_rows,note,done_at) "
        "VALUES (?,?,?,?,?,?)",
        (kind, str(key), status, n, note[:200],
         datetime.now(timezone.utc).isoformat(timespec="seconds")))


def status() -> dict:
    """Where a restart would pick up. Safe to call any time."""
    con = connect()
    try:
        q = lambda s, *a: con.execute(s, a).fetchone()[0]  # noqa: E731
        out = {
            "days_done": q("SELECT COUNT(*) FROM te_scrape_log WHERE kind='day'"),
            "day_errors": q("SELECT COUNT(*) FROM te_scrape_log "
                            "WHERE kind='day' AND status='error'"),
            "matches_known": q("SELECT COUNT(*) FROM te_matches"),
            "details_done": q("SELECT COUNT(*) FROM te_matches WHERE detail_done=1"),
            "details_remaining": q("SELECT COUNT(*) FROM te_matches WHERE detail_done=0"),
            "quotes": q("SELECT COUNT(*) FROM odds_quotes"),
        }
        r = con.execute("SELECT MIN(key), MAX(key) FROM te_scrape_log "
                        "WHERE kind='day' AND status IN ('ok','empty')").fetchone()
        out["day_range"] = f"{r[0]} .. {r[1]}" if r[0] else None
        return out
    finally:
        con.close()


# --------------------------------------------------------------------------
# tier A: results day pages
# --------------------------------------------------------------------------
def parse_day(html: str, day: date) -> list[dict]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    out, tournament, skip = [], "", False
    for table in soup.find_all("table", class_="result"):
        rows = table.find_all("tr")
        i = 0
        while i < len(rows):
            r = rows[i]
            if "head" in (r.get("class") or []):
                td = r.find("td", class_="t-name")
                tournament = td.get_text(" ", strip=True) if td else ""
                skip = any(t in tournament.lower() for t in SKIP_TERMS)
                i += 1
                continue
            link = r.find("a", href=re.compile(r"/match-detail/\?id=\d+"))
            if link is not None and i + 1 < len(rows):
                nxt = rows[i + 1]
                if not skip:
                    te_id = int(re.search(r"id=(\d+)", link["href"]).group(1))
                    def nm(row):
                        td = row.find("td", class_="t-name")
                        return (td.get_text(" ", strip=True).split("(")[0].strip()
                                if td else "")
                    prices = []
                    for td in r.find_all("td"):
                        if set(td.get("class") or []) & {"course", "coursew"}:
                            try:
                                prices.append(float(td.get_text(strip=True)))
                            except ValueError:
                                prices.append(None)
                    prices = (prices + [None, None])[:2]
                    if nm(r) and nm(nxt):
                        out.append({
                            "te_id": te_id,
                            "play_date": int(day.strftime("%Y%m%d")),
                            "tournament": tournament,
                            "p1_name": nm(r), "p2_name": nm(nxt),
                            "score": " ".join(
                                td.get_text(strip=True) for td in r.find_all("td")
                                if "score" in (td.get("class") or [])).strip(),
                            "p1_odds": prices[0], "p2_odds": prices[1],
                            "tour": tour_of(tournament),
                        })
                i += 2
                continue
            i += 1
    return out


def scrape_days(start: date, end: date, *, tour: str = "all",
                newest_first: bool = True) -> dict:
    """Tier A. Resumable: days already logged are skipped without a request.

    Walks backwards from `end` by default. Recent seasons are the ones worth
    testing on -- they carry opening odds, the games lines are denser, and they
    match the era the model is scored in -- so an interrupted run should already
    have banked the useful half.
    """
    con = connect()
    done = _done(con, "day")
    n_days = n_rows = n_err = skipped = 0
    step = timedelta(days=-1 if newest_first else 1)
    day = end if newest_first else start
    try:
        while start <= day <= end:
            key = day.strftime("%Y%m%d")
            if key in done:
                skipped += 1
                day += step
                continue
            try:
                html = fetch(
                    f"/results/?type={tour}&year={day.year}"
                    f"&month={day.month:02d}&day={day.day:02d}",
                    cache_key=f"day{key}")
                rows = parse_day(html, day)
                con.executemany(
                    "INSERT OR IGNORE INTO te_matches"
                    "(te_id,play_date,tour,tournament,p1_name,p2_name,score,"
                    " p1_odds,p2_odds,fetched_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    [(r["te_id"], r["play_date"], r["tour"], r["tournament"],
                      r["p1_name"], r["p2_name"], r["score"],
                      r["p1_odds"], r["p2_odds"],
                      datetime.now(timezone.utc).isoformat(timespec="seconds"))
                     for r in rows])
                _mark(con, "day", key, "ok" if rows else "empty", len(rows))
                n_rows += len(rows)
            except Exception as exc:
                _mark(con, "day", key, "error", 0, str(exc))
                n_err += 1
                log.warning("day %s failed: %s", key, exc)
            n_days += 1
            if n_days % 10 == 0:      # commit in batches, not at the end
                con.commit()
                log.info("  days %d done, %d matches, %d errors", n_days, n_rows, n_err)
            day += step
        con.commit()
    finally:
        con.commit()
        con.close()
    return {"days_fetched": n_days, "days_skipped": skipped,
            "matches": n_rows, "errors": n_err}


# --------------------------------------------------------------------------
# tier B: match detail pages
# --------------------------------------------------------------------------
def _price_cell(td) -> tuple[float | None, str | None, float | None, str | None]:
    """(close, closed_at, open, opened_at) from one price cell.

    The visible number is the closing price. A hover table, present only when
    the price moved, holds the close on its first row and the open on its
    third. Cells without that table have one price and no history.
    """
    div = td.find("div", class_="odds-in")
    if div is None:
        return None, None, None, None
    change = div.find("div", class_="odds-change-div")
    head = div.find(string=True, recursive=False)
    try:
        close = float(str(head).strip())
    except (TypeError, ValueError):
        close = None
    if change is None:
        return close, None, None, None
    rows = change.find_all("tr")
    def cell(row, i):
        tds = row.find_all("td")
        return tds[i].get_text(strip=True) if len(tds) > i else ""
    closed_at = cell(rows[0], 0) if rows else None
    opened_at = open_px = None
    if len(rows) >= 3:
        opened_at = cell(rows[2], 0)
        try:
            open_px = float(cell(rows[2], 1))
        except ValueError:
            open_px = None
    return close, closed_at, open_px, opened_at


def parse_detail(html: str, te_id: int) -> list[dict]:
    """Every quote on a match page, across all four market tabs."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    out = []
    for tab, market in _MARKETS.items():
        div = soup.find(id=f"oddsMenu-{tab}-data")
        if div is None:
            continue
        line = line_unit = None
        sides = ("p1", "p2")
        for tr in div.find_all("tr"):
            txt = tr.get_text(" ", strip=True)
            # A sub-table header states the line and its unit, e.g.
            # "Over/Under 21.5 games" or "Asian Handicap -1.5 sets".
            m = re.match(r"(Over/Under|Asian Handicap|Correct Score)\s*"
                         r"(-?\d+(?:\.\d+)?)?\s*(sets?|games?)?", txt)
            if m and tr.find("td", class_="k1") is None and len(txt) < 40:
                line = float(m.group(2)) if m.group(2) else None
                unit = m.group(3)
                line_unit = ("games" if unit and unit.startswith("game")
                             else "sets" if unit else None)
                sides = (("over", "under") if market == "totals" else ("p1", "p2"))
                continue
            book_el = tr.find("span", class_="t")
            if book_el is None:
                continue
            book = book_el.get_text(strip=True)
            cells = [tr.find("td", class_="k1"), tr.find("td", class_="k2")]
            if market == "correct_score":
                # The score itself is the side; it sits in the row's own cells.
                score = next((td.get_text(strip=True) for td in tr.find_all("td")
                              if re.fullmatch(r"\d:\d", td.get_text(strip=True))), None)
                if score is None or cells[0] is None:
                    continue
                c, ca, o, oa = _price_cell(cells[0])
                if c is not None:
                    out.append(dict(te_id=te_id, book=book, market=market,
                                    line=None, line_unit=None, side=score,
                                    price_close=c, closed_at=ca,
                                    price_open=o, opened_at=oa))
                continue
            for side, td in zip(sides, cells):
                if td is None:
                    continue
                c, ca, o, oa = _price_cell(td)
                if c is None:
                    continue
                out.append(dict(te_id=te_id, book=book, market=market,
                                line=line, line_unit=line_unit, side=side,
                                price_close=c, closed_at=ca,
                                price_open=o, opened_at=oa))
    return out


def scrape_details(limit: int | None = None, *, order: str = "recent") -> dict:
    """Tier B. Picks up wherever it left off; commits every `BATCH` matches."""
    con = connect()
    sql = ("SELECT te_id FROM te_matches WHERE detail_done=0 ORDER BY play_date "
           + ("DESC" if order == "recent" else "ASC"))
    if limit:
        sql += f" LIMIT {int(limit)}"
    todo = [r[0] for r in con.execute(sql)]
    n_ok = n_err = n_q = 0
    try:
        for i, te_id in enumerate(todo, 1):
            try:
                html = fetch(f"/match-detail/?id={te_id}", cache_key=f"m{te_id}")
                quotes = parse_detail(html, te_id)
                if quotes:
                    con.executemany(
                        "INSERT OR REPLACE INTO odds_quotes"
                        "(te_id,book,market,line,line_unit,side,price_close,"
                        " closed_at,price_open,opened_at) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?)",
                        [(q["te_id"], q["book"], q["market"], q["line"],
                          q["line_unit"], q["side"], q["price_close"],
                          q["closed_at"], q["price_open"], q["opened_at"])
                         for q in quotes])
                con.execute("UPDATE te_matches SET detail_done=1 WHERE te_id=?", (te_id,))
                _mark(con, "detail", te_id, "ok" if quotes else "empty", len(quotes))
                n_ok += 1
                n_q += len(quotes)
            except Exception as exc:
                _mark(con, "detail", te_id, "error", 0, str(exc))
                n_err += 1
                log.warning("detail %s failed: %s", te_id, exc)
            if i % BATCH == 0:
                con.commit()
                log.info("  %d/%d details, %d quotes, %d errors",
                         i, len(todo), n_q, n_err)
        con.commit()
    finally:
        con.commit()
        con.close()
    return {"details": n_ok, "quotes": n_q, "errors": n_err,
            "remaining": max(0, len(todo) - n_ok - n_err)}


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="tennisexplorer historical odds")
    ap.add_argument("--days", nargs=2, metavar=("START", "END"),
                    help="tier A, inclusive, YYYY-MM-DD")
    ap.add_argument("--details", type=int, nargs="?", const=0,
                    help="tier B; optional cap on how many pages this run")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    if a.status:
        for k, v in status().items():
            print(f"{k}: {v}")
    elif a.days:
        s, e = (datetime.strptime(x, "%Y-%m-%d").date() for x in a.days)
        print(scrape_days(s, e))
    elif a.details is not None:
        print(scrape_details(a.details or None))
    else:
        ap.print_help()
