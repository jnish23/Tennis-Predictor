"""checkbestodds.com historical odds -- the only Challenger prices we can get.

tennis-data.co.uk is ATP main tour only, which left roughly half our match
volume with no price at all and made the "are Challenger markets softer?"
question untestable. This source lists 234 Challenger tournaments alongside 89
ATP ones, so it also supplies a same-source main-tour control.

**robots.txt** (checked 2026-08): `/tennis-odds/` is not disallowed. The
disallow list covers `/bdl/` and the per-bookmaker landing pages (`/Pinnacle`,
`/Bet365`, ...), none of which this module touches. Pages are fetched once,
cached to disk, and never re-fetched; requests are spaced by `DELAY` seconds.

**What the price actually is, and why it matters.** The listing gives *best
odds* -- the maximum across whichever bookmakers covered the match. That is not
a single book's closing line:

* it is an upper bound on price, so any ROI computed from it is optimistic
  against what one account could actually have taken;
* the implied overround is understated whenever two books are combined, because
  you are taking the best side of each;
* it carries no timestamp, so it is not necessarily a *closing* price.

For Challengers the practical reality is thinner than that suggests -- spot
checks find single-bookmaker coverage (Bet365 alone, 8.1% margin) where a
main-tour match would have a dozen books at 2-3%. Per-match overround is
computed on load so the analysis can report what it is really working with
rather than assuming.
"""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import pandas as pd
import requests

from tennis.config import DATA
from tennis.db.schema import connect
from tennis.ingest.load import norm_name

log = logging.getLogger(__name__)

BASE = "https://checkbestodds.com"
INDEX = f"{BASE}/historical-tennis-odds"
CACHE = DATA / "raw" / "cbo"
CACHE.mkdir(parents=True, exist_ok=True)
DELAY = 2.0
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120 Safari/537.36")
BOOK = "CBO"  # book code written into the odds table

_ROW = re.compile(
    r'<tr>\s*<td class="l2 match">\s*'
    r'<span ts="(?P<ts>\d+)"[^>]*>[^<]*</span>\s*'
    r'<a href="(?P<href>[^"]+)">\s*(?P<names>[^<]+)</a>\s*</td>\s*'
    r'<td class="r">\s*<b>(?P<o1>[\d.]+)</b></td>\s*'
    r'<td class="r">\s*<b>(?P<o2>[\d.]+)</b></td>',
    re.S)


def fetch(path: str, *, refresh: bool = False) -> str:
    """GET a page, cached on disk. `path` is site-relative."""
    key = re.sub(r"[^a-z0-9]+", "_", path.lower()).strip("_") + ".html"
    f = CACHE / key
    if f.exists() and not refresh:
        return f.read_text(encoding="utf-8", errors="replace")
    time.sleep(DELAY)
    r = requests.get(f"{BASE}{path}", headers={"User-Agent": UA}, timeout=45)
    r.raise_for_status()
    f.write_text(r.text, encoding="utf-8")
    return r.text


def tournament_slugs(kind: str | None = None) -> list[str]:
    """Every historical-odds tournament page, optionally filtered by tier."""
    slugs = sorted(set(re.findall(
        r"/tennis-odds/(historical-odds-[a-z0-9\-]+)", fetch("/historical-tennis-odds"))))
    if kind:
        slugs = [s for s in slugs if s.startswith(f"historical-odds-{kind}")]
    return slugs


def parse_page(html: str, slug: str) -> pd.DataFrame:
    """Rows of (timestamp, both players, both prices) from one tournament page.

    Player names are split on ' - '. Hyphenated names are safe because the
    source writes them unspaced ('Jan-Lennard Struff'), so only the separator
    carries surrounding whitespace.
    """
    rows = []
    for m in _ROW.finditer(html):
        parts = [p.strip() for p in m.group("names").split(" - ")]
        if len(parts) != 2 or not all(parts):
            continue
        rows.append({
            "ts": int(m.group("ts")),
            "p1_name": parts[0], "p2_name": parts[1],
            "p1_odds": float(m.group("o1")), "p2_odds": float(m.group("o2")),
            "cbo_slug": slug, "cbo_url": m.group("href"),
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    df["tourney_date"] = df["date"].dt.strftime("%Y%m%d").astype(int)
    df["tier"] = slug.replace("historical-odds-", "").split("-")[0]
    return df


def scrape(kinds=("challenger", "atp"), *, limit: int | None = None) -> pd.DataFrame:
    out = []
    for kind in kinds:
        slugs = tournament_slugs(kind)[:limit]
        log.info("%s: %d tournament pages", kind, len(slugs))
        for i, s in enumerate(slugs, 1):
            try:
                out.append(parse_page(fetch(f"/tennis-odds/{s}"), s))
            except Exception as exc:      # one dead page must not kill the run
                log.warning("%s failed: %s", s, exc)
            if i % 25 == 0:
                log.info("  %s %d/%d", kind, i, len(slugs))
    df = pd.concat([d for d in out if not d.empty], ignore_index=True)
    # The same match can appear under two tournament slugs when an event is
    # renamed between years; keep one row per (players, day).
    df = df.drop_duplicates(subset=["ts", "p1_name", "p2_name"])
    return df


# --------------------------------------------------------------------------
# per-book detail, and the reason it is needed
# --------------------------------------------------------------------------
_BOOK_ROW = re.compile(
    r'<tr>\s*<td><div class="bookName">.*?class="toSort"[^>]*>(?P<book>[^<]+)</a></td>\s*'
    r'<td>\s*<span class="toSort noDsp">(?P<o1>[\d.]+)</span>.*?'
    r'<td>\s*<span class="toSort noDsp">(?P<o2>[\d.]+)</span>', re.S)

# Overround band a best-odds pair must fall in, calibrated against Pinnacle
# rather than assumed. Best odds legitimately sits *below* 1.0 -- that is a
# genuine cross-book arbitrage, and this site exists partly to surface them --
# so a 1.0 floor would throw away real data, and specifically the
# best-priced rows that any ROI depends on.
#
# The floor comes from measurement. On 19,525 matches priced by both this
# source and Pinnacle, correlation between the two devigged probabilities by
# overround band runs: [0.5,0.7) 0.486, [0.7,0.85) 0.841, [0.85,0.92) 0.962,
# [0.92,0.96) 0.987, [0.96,0.99) 0.994, [0.99,1.0) 0.997, [1.0,1.05) 0.998.
# Contamination from transposed bookmakers dominates below ~0.96 and is
# negligible above it, so that is the cut.
OVR_MIN, OVR_MAX = 0.96, 1.20


def parse_detail(html: str) -> pd.DataFrame:
    """Per-bookmaker prices for one match.

    Exists because the listing page's "best odds" cannot be trusted on its own.
    It takes the maximum per side *independently across books*, so a single
    bookmaker with its two sides transposed poisons one column. Observed:
    Cecchinato-Sergeyev 2013-02-05 listed 3.47 / 26.00, while eleven of twelve
    books had ~3.20 / ~1.29 and only Titanbet showed 1.01 / 26.00. Believing
    that row means pricing a 1.29 favourite at 26.00 and booking an ROI that
    was never available.
    """
    rows = []
    for m in _BOOK_ROW.finditer(html):
        b = m.group("book").strip()
        if not b or b.lower().startswith("bookmaker"):
            continue
        rows.append({"book": b, "o1": float(m.group("o1")),
                     "o2": float(m.group("o2"))})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["overround"] = 1 / df.o1 + 1 / df.o2
    return df


def sane(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose implied overround is not a possible two-way market."""
    o = 1 / df["p1_odds"] + 1 / df["p2_odds"]
    return df[(o >= OVR_MIN) & (o <= OVR_MAX)].copy()


# --------------------------------------------------------------------------
# joining to our own matches
# --------------------------------------------------------------------------
def _key(name: str) -> str:
    """'Matteo Donati' -> 'donati|m'. Same shape as the tennis-data join."""
    toks = norm_name(name).split()
    if len(toks) < 2:
        return norm_name(name)
    return f"{toks[-1]}|{toks[0][0]}"


def join_to_matches(df: pd.DataFrame, con=None) -> pd.DataFrame:
    """Attach our match_id by (date window, both player keys).

    Dates are matched within +/-7 days. Archived seasons stamp a match with the
    week its tournament began while this source carries the true playing day,
    so a second-week match is legitimately six days adrift. Measured: a +/-3
    window matched 47% of pairs that exist in our data, +/-7 matches 82%, and
    +/-14 adds only a further 1% -- so 7 captures the calendar offset without
    reaching into a different event. A pair can only meet once inside one
    tournament, and the nearest date wins regardless.
    """
    close = con is None
    con = con or connect()
    m = pd.read_sql(
        """SELECT m.match_id, m.tourney_date, m.winner_id, m.loser_id,
                  pw.name AS w_name, pl.name AS l_name, t.is_challenger
           FROM matches m
           JOIN players pw ON pw.player_id = m.winner_id
           JOIN players pl ON pl.player_id = m.loser_id
           JOIN tournaments t USING(tourney_key)""", con)
    if close:
        con.close()

    m["kw"], m["kl"] = m["w_name"].map(_key), m["l_name"].map(_key)
    m["pair"] = [tuple(sorted(x)) for x in zip(m["kw"], m["kl"])]
    m["d"] = pd.to_datetime(m["tourney_date"].astype(str), format="%Y%m%d")

    df = df.copy()
    df["pair"] = [tuple(sorted(x)) for x in
                  zip(df["p1_name"].map(_key), df["p2_name"].map(_key))]
    df["d"] = pd.to_datetime(df["tourney_date"].astype(str), format="%Y%m%d")

    merged = df.merge(m, on="pair", how="inner", suffixes=("", "_m"))
    merged["gap"] = (merged["d"] - merged["d_m"]).dt.days.abs()
    merged = merged[merged["gap"] <= 7]
    # Nearest date wins when a pair met more than once in a season.
    merged = merged.sort_values("gap").drop_duplicates(subset=["match_id"])
    merged = merged.drop_duplicates(subset=["ts", "pair"])

    # Re-orient onto winner/loser, which is how the odds table stores prices.
    p1_is_winner = merged["p1_name"].map(_key) == merged["kw"]
    merged["win_price"] = merged["p1_odds"].where(p1_is_winner, merged["p2_odds"])
    merged["lose_price"] = merged["p2_odds"].where(p1_is_winner, merged["p1_odds"])
    merged["overround"] = 1 / merged["p1_odds"] + 1 / merged["p2_odds"]
    return merged


def load_to_db(df: pd.DataFrame, con=None) -> dict:
    close = con is None
    con = con or connect()
    rows = df[["match_id", "win_price", "lose_price"]].copy()
    rows["book"] = BOOK
    con.execute("DELETE FROM odds WHERE book = ?", (BOOK,))
    con.executemany(
        "INSERT INTO odds(match_id, book, win_price, lose_price) VALUES (?,?,?,?)",
        rows[["match_id", "book", "win_price", "lose_price"]].itertuples(
            index=False, name=None))
    con.commit()
    n = con.execute("SELECT COUNT(*) FROM odds WHERE book=?", (BOOK,)).fetchone()[0]
    if close:
        con.close()
    return {"inserted": int(n)}


def run(kinds=("challenger", "atp")) -> dict:
    raw = scrape(kinds)
    joined = join_to_matches(raw)
    stats = load_to_db(joined)
    return {"scraped": len(raw), "joined": len(joined), **stats}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    for k, v in run().items():
        print(f"{k}: {v}")
