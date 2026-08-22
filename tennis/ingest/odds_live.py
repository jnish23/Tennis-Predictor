"""Live odds capture from tennisexplorer.com, including Challengers.

Why our own scraper rather than the tennis_data repo that wraps the same site:

* **No licence.** That repository publishes no licence, so its code is all
  rights reserved and cannot be copied into this project. Its *published data*
  is fine to read, which is what `backfill_from_github` does.
* **It accumulates nothing.** Its `matches.json` is overwritten every six
  hours, so as a source it has no memory -- and history is the entire point
  here, since checkbestodds.com stops in 2022.
* **Its documented behaviour is not its actual behaviour.** `main_only` claims
  to drop Challengers, but the filter list is
  `["itf","futures","utr","exhibition"]` and "challenger" is absent, so they
  pass through by accident rather than by design. We depend on Challengers.
* It disables TLS certificate verification with no need; the certificate
  validates normally.

robots.txt (checked 2026-08) disallows only `/redirect/`, `/terms-of-use/` and
`/contact/`. The match pages are permitted. Requests are spaced by `DELAY`.

**Every capture is kept.** A single scrape is one arbitrary moment in a moving
market. Repeated captures let `resolve_snapshots` promote the *last price seen
before play* into the match-keyed `odds` table, which is the closest thing to a
closing line this project has ever had -- checkbestodds carried no timestamp at
all, so its prices could never be called closing.
"""
from __future__ import annotations

import logging
import re
import time
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests

from tennis.db.schema import connect
from tennis.ingest.load import norm_name

log = logging.getLogger(__name__)

BASE = "https://www.tennisexplorer.com"
SOURCE = "tennisexplorer"
BOOK = "TE_CLOSE"          # book code used for the resolved closing-line proxy
DELAY = 2.0
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120 Safari/537.36")
# Tiers below the Challenger floor of this project's scope.
SKIP_TERMS = ("itf", "futures", "utr", "exhibition")


def surname_key(n: str) -> str:
    """'Jannik Sinner' and 'Sinner J.' both key to 'sinner|j'.

    tennisexplorer writes 'Sinner J.'; our players table holds 'Jannik Sinner'.
    Detecting the abbreviated style rather than assuming first-name-first
    matters -- assuming would key 'Sinner J.' as 'j|s' and never join.
    """
    toks = norm_name(n).split()
    if not toks:
        return ""
    initials = [t for t in toks if len(t) == 1]
    words = [t for t in toks if len(t) > 1]
    if not words:
        return toks[0]
    if initials:
        return f"{words[-1]}|{initials[0]}"
    if len(words) == 1:
        return words[0]
    return f"{words[-1]}|{words[0][0]}"


def fetch(tour: str, day: date | None = None) -> str:
    url = f"{BASE}/matches/?type={tour}-single"
    if day:
        url += f"&year={day.year}&month={day.month:02d}&day={day.day:02d}"
    time.sleep(DELAY)
    r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    return r.text


def parse(html: str, tour: str, day: date) -> pd.DataFrame:
    """Fixtures and prices from one matches page.

    A match spans two table rows: the first carries the time, player one and
    both prices, the second carries player two. Seeds are stripped -- '(5)' is
    draw furniture, not part of a name.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    out, tournament, skip = [], "", False
    for table in soup.find_all("table", class_="result"):
        rows = table.find_all("tr")
        i = 0
        while i < len(rows):
            r = rows[i]
            cls = r.get("class", [])
            if "head" in cls:
                td = r.find("td", class_="t-name")
                tournament = td.get_text(" ", strip=True) if td else ""
                skip = any(t in tournament.lower() for t in SKIP_TERMS)
                i += 1
                continue
            # A match's first row is identified by its time cell, not by a
            # class: only the first match in each table carries "fRow", while
            # later ones alternate one/two, so keying on fRow finds one match
            # per tournament and silently drops the rest.
            if r.find("td", class_="time") is not None and i + 1 < len(rows):
                nxt = rows[i + 1]
                if not skip:
                    def txt(row, klass):
                        td = row.find("td", class_=klass)
                        return td.get_text(" ", strip=True) if td else ""

                    p1, p2 = txt(r, "t-name"), txt(nxt, "t-name")
                    if p1 and p2:
                        o1, o2 = _prices(r)
                        out.append({
                            "play_date": int(day.strftime("%Y%m%d")),
                            "start_time": _time(txt(r, "time")),
                            "tour": tour, "tournament": tournament,
                            "p1_name": _clean(p1), "p2_name": _clean(p2),
                            "p1_odds": o1, "p2_odds": o2,
                            "status": _status(r, nxt),
                        })
                i += 2
                continue
            i += 1
    return pd.DataFrame(out)


def _clean(name: str) -> str:
    """Drop the seed marker: 'Shelton B. (5)' -> 'Shelton B.'"""
    return name.split("(")[0].strip()


def _time(raw: str) -> str | None:
    """'14:30 Live streams 1xBet' -> '14:30'. The cell carries ad copy."""
    m = re.match(r"\s*(\d{1,2}:\d{2})", raw or "")
    return m.group(1) if m else None


def _status(row, nxt) -> str:
    """'finished' once the site marks a winner, else 'upcoming'.

    The day page lists everything scheduled for that date regardless of whether
    it has been played, and the capture agent runs through the day, so roughly
    half of what it collects at any moment is already over. Without this the
    betting screen quoted a Cincinnati match that had finished fourteen hours
    earlier, at its pre-match price, indistinguishable from a live one.

    The marker is the same `coursew` class `_prices` has to work around: the
    site relabels the winner's price cell only once there is a winner. A score
    cell is the corroborating signal, since an unplayed match has none.
    """
    for r in (row, nxt):
        for td in r.find_all("td"):
            if "coursew" in set(td.get("class") or []):
                return "finished"
    score = row.find("td", class_="result")
    if score and score.get_text(strip=True):
        return "finished"
    return "upcoming"


def _prices(row) -> tuple[float | None, float | None]:
    """Both decimal prices, in player order.

    The two price cells are *not* reliably classed the same way. An upcoming
    match carries two `course` cells; once a winner exists the site relabels
    the winner's cell `coursew`. Reading `coursew` as "player one" therefore
    captures only matches that have already finished -- exactly backwards for a
    live collector, and the reason an earlier version returned prices for two
    completed main-tour matches and none of the 39 upcoming Challengers.
    Taking the first two price cells in document order works in both states.
    """
    vals = []
    for td in row.find_all("td"):
        cls = set(td.get("class") or [])
        if cls & {"course", "coursew"}:
            try:
                vals.append(float(td.get_text(strip=True).replace(",", ".")))
            except ValueError:
                vals.append(None)
    vals = (vals + [None, None])[:2]
    return vals[0], vals[1]


def capture(day: date | None = None, tours=("atp", "wta"),
            days_ahead: int = 1) -> dict:
    """Scrape now and append to `odds_snapshots`. Safe to run repeatedly.

    Fetches tomorrow's page as well as today's. A night session that starts at
    or after midnight in the site's own clock is filed under the *next* date
    even though it belongs to tonight's play: the Cincinnati semi-final
    Nakashima-Tiafoe was listed at 00:00 on the following day's page while the
    other semi-final sat at 19:30 on today's. Fetching only today missed it
    entirely, and would have kept missing it until the day it started. Costs
    one extra request per tour, against a page that carries a handful of rows.
    """
    day = day or datetime.now(timezone.utc).date()
    days = [day + timedelta(days=i) for i in range(days_ahead + 1)]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    frames = []
    for t in tours:
        for d in days:
            try:
                frames.append(parse(fetch(t, d), t, d))
            except Exception as exc:
                log.warning("%s capture failed for %s: %s", t, d, exc)
    if not frames:
        return {"captured": 0}
    df = pd.concat(frames, ignore_index=True)
    # Check for rows before touching columns. An unparseable page yields a
    # frame with no columns at all, so `df.p1_odds` raises AttributeError
    # rather than returning nothing -- the agent would crash instead of
    # logging a quiet zero. Now that tomorrow's page is fetched too, an empty
    # parse is routine: out of season, or simply nothing scheduled yet.
    if df.empty or "p1_odds" not in df.columns:
        return {"captured": 0}
    df = df[df.p1_odds.notna() & df.p2_odds.notna()]
    if df.empty:
        return {"captured": 0}
    df["captured_at"], df["source"] = now, SOURCE

    con = connect()
    cols = ["captured_at", "play_date", "start_time", "tour", "tournament",
            "p1_name", "p2_name", "p1_odds", "p2_odds", "source", "status"]
    con.executemany(
        f"INSERT OR REPLACE INTO odds_snapshots({','.join(cols)}) "
        f"VALUES ({','.join('?' * len(cols))})",
        df[cols].itertuples(index=False, name=None))
    con.commit()
    total = con.execute("SELECT COUNT(*) FROM odds_snapshots").fetchone()[0]
    con.close()
    n_ch = int(df.tournament.str.lower().str.contains("challenger").sum())
    return {"captured": len(df), "challenger": n_ch, "table_total": int(total)}


def resolve_snapshots(con=None) -> dict:
    """Promote the last capture before play into the match-keyed `odds` table.

    Only fixtures whose result we already hold can be resolved, so this is run
    after the daily match ingest. The last capture before start is the closing
    price we actually observed; earlier captures stay in `odds_snapshots` for
    line-movement work.
    """
    close = con is None
    con = con or connect()
    snaps = pd.read_sql(
        "SELECT * FROM odds_snapshots WHERE source = ?", con, params=(SOURCE,))
    if snaps.empty:
        if close:
            con.close()
        return {"resolved": 0}

    # Last capture wins per fixture.
    snaps = snaps.sort_values("captured_at").drop_duplicates(
        subset=["play_date", "p1_name", "p2_name"], keep="last")
    snaps["pair"] = [tuple(sorted(x)) for x in
                     zip(snaps.p1_name.map(surname_key),
                         snaps.p2_name.map(surname_key))]

    m = pd.read_sql(
        """SELECT m.match_id, m.tourney_date, pw.name w, pl.name l
           FROM matches m
           JOIN players pw ON pw.player_id = m.winner_id
           JOIN players pl ON pl.player_id = m.loser_id""", con)
    m["pair"] = [tuple(sorted(x)) for x in
                 zip(m.w.map(surname_key), m.l.map(surname_key))]

    j = snaps.merge(m, on="pair", how="inner")
    j["gap"] = (pd.to_datetime(j.play_date.astype(str)) -
                pd.to_datetime(j.tourney_date.astype(str))).dt.days.abs()
    j = j[j.gap <= 7].sort_values("gap").drop_duplicates(subset=["match_id"])

    p1_is_winner = j.p1_name.map(surname_key) == j.w.map(surname_key)
    j["win_price"] = j.p1_odds.where(p1_is_winner, j.p2_odds)
    j["lose_price"] = j.p2_odds.where(p1_is_winner, j.p1_odds)
    j["book"] = BOOK
    con.executemany(
        "INSERT OR REPLACE INTO odds(match_id, book, win_price, lose_price) "
        "VALUES (?,?,?,?)",
        j[["match_id", "book", "win_price", "lose_price"]].itertuples(
            index=False, name=None))
    con.commit()
    n = con.execute("SELECT COUNT(*) FROM odds WHERE book=?", (BOOK,)).fetchone()[0]
    if close:
        con.close()
    return {"matched": len(j), "odds_rows": int(n)}


def backfill_from_github(limit: int | None = None) -> dict:
    """Seed history from the tennis_data repo's committed snapshots.

    Reading published data, not copying code -- the licence question does not
    arise. That repo commits `matches.json` every six hours and has done since
    2026-03-10, so its git history is an archive of roughly five months of
    fixtures and prices that we would otherwise have to wait to collect. It
    carries no explicit date, so each snapshot is dated by its commit.
    """
    api = "https://api.github.com/repos/Mriganka-codes/tennis_data/commits"
    raw = "https://raw.githubusercontent.com/Mriganka-codes/tennis_data"
    h = {"User-Agent": "tennis-predictor", "Accept": "application/vnd.github+json"}
    commits, page = [], 1
    while True:
        r = requests.get(api, headers=h, params={"per_page": 100, "page": page},
                         timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        commits += [(c["sha"], c["commit"]["author"]["date"]) for c in batch]
        page += 1
        if limit and len(commits) >= limit:
            break
    commits = commits[:limit] if limit else commits

    rows = []
    for sha, when in commits:
        try:
            d = requests.get(f"{raw}/{sha}/matches.json", headers=h, timeout=30).json()
        except Exception:
            continue
        day = int(when[:10].replace("-", ""))
        for m in d.get("matches", []):
            if m.get("odds1") is None or m.get("odds2") is None:
                continue
            t = (m.get("tournament") or "")
            if any(x in t.lower() for x in SKIP_TERMS):
                continue
            rows.append({
                "captured_at": when, "play_date": day,
                "start_time": m.get("time"), "tour": (m.get("tour") or "").lower(),
                "tournament": t, "p1_name": _clean(m.get("player1", "")),
                "p2_name": _clean(m.get("player2", "")),
                "p1_odds": m.get("odds1"), "p2_odds": m.get("odds2"),
                "source": SOURCE,
            })
    if not rows:
        return {"backfilled": 0}
    df = pd.DataFrame(rows)
    con = connect()
    cols = ["captured_at", "play_date", "start_time", "tour", "tournament",
            "p1_name", "p2_name", "p1_odds", "p2_odds", "source"]
    con.executemany(
        f"INSERT OR REPLACE INTO odds_snapshots({','.join(cols)}) "
        f"VALUES ({','.join('?' * len(cols))})",
        df[cols].itertuples(index=False, name=None))
    con.commit()
    total = con.execute("SELECT COUNT(*) FROM odds_snapshots").fetchone()[0]
    con.close()
    return {"commits": len(commits), "backfilled": len(df),
            "table_total": int(total)}


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    # Run as a module from the repo root -- `python -m tennis.ingest.odds_live`.
    # Executing the file by path puts tennis/ingest on sys.path instead of the
    # project root, so the `tennis` package itself is not importable.
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backfill", action="store_true",
                    help="seed history from the tennis_data repo's commits")
    ap.add_argument("--resolve", action="store_true",
                    help="promote the last pre-match capture into `odds`")
    ap.add_argument("--limit", type=int, default=None,
                    help="backfill: newest N commits only (default: all)")
    a = ap.parse_args()
    if a.backfill:
        print(backfill_from_github(limit=a.limit))
    elif a.resolve:
        print(resolve_snapshots())
    else:
        print(capture())
