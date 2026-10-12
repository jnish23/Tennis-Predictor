"""Set-level winner prices from OddsPortal, via the OddsHarvester CLI.

Why this exists: nothing already in the database prices a *set*. The Stage 0
prescreen (scripts/set_markets_prescreen.py) shows the structure the market has
to respect -- sets are strongly correlated (Beta kappa ~4-5) and the O/U-sets
line already prices that correctly -- so whether a first- or second-set line
carries a gap is only answerable against real set prices.

Built like `odds_te_hist.py`: every scrape is cached as the raw JSON the tool
wrote, every (league, season, period) unit is checkpointed in `op_scrape_log`
the moment it lands, and ingestion is a pure function of the cached files, so
re-parsing never costs a request.

OddsHarvester is driven as a subprocess (it has its own Playwright venv; point
ODDSHARVESTER_BIN at it). One scrape at a time, guarded by a lock, because two
browsers against one site doubles the request rate rather than the throughput.

**A zero-record result is an error, not "empty".** OddsPortal serves a
degraded page when it throttles a session -- the results list renders with no
rows and the tool exits 0. Recording that as done would hide the block and skip
the event forever, so it is logged `error` and retried. This module does not
try to defeat a block: if the site stops answering, the run stops.

Known limits, stated because the plan depends on them:
* The tool returns one price per bookmaker per set, with no open/close label.
  Treat them as the page's current (post-match = closing) quotes; opening
  prices and line movement need `--odds-history` and are not collected.
* `home`/`away` is OddsPortal's order, not the winner. Orientation is resolved
  once in `resolve_matches` and checked against our own set-1 result.
"""
from __future__ import annotations

import ast
import json
import logging
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from tennis.config import DATA
from tennis.db.lock import AlreadyRunning, single_instance
from tennis.db.schema import connect, init_db
from tennis.ingest.parse import set_winners

log = logging.getLogger(__name__)

CACHE = DATA / "raw" / "oddsportal"
BIN = Path(os.environ.get(
    "ODDSHARVESTER_BIN",
    Path.home() / "odds-portal-scraper" / ".venv" / "bin" / "oddsharvester"))
PERIOD_FLAG = {1: "1st_set", 2: "2nd_set"}
PERIOD_NAME = {"FirstSet": 1, "SecondSet": 2}
REQUEST_DELAY = 2.0          # seconds between match pages, passed to the tool
TIMEOUT = 6 * 3600           # a tournament is ~50-100 pages at 20-60 s each


# --------------------------------------------------------------------------
# parsing -- pure functions, no I/O
# --------------------------------------------------------------------------
_SET_TOK = re.compile(r"^\s*(\d+)(?:\s+(\d+))?\s*:\s*(\d+)(?:\s+(\d+))?\s*$")


def parse_partial(text) -> list[tuple[int, int]]:
    """'(7:6 4 , 1:6, 4:6)' -> [(7, 6), (1, 6), (4, 6)] as (home, away) games.

    A tiebreak score trails the games it belongs to: '7:6 4' is 7-6(4) and
    '6 5 :7' is 6(5)-7. The tiebreak points are dropped; only games matter for
    the set winner.
    """
    if not isinstance(text, str):
        return []
    out = []
    for tok in text.strip().strip("()").split(","):
        m = _SET_TOK.match(tok)
        if m:
            out.append((int(m.group(1)), int(m.group(3))))
    return out


def op_id_of(link: str) -> str:
    """OddsPortal's own match id: the URL fragment, else the last path token."""
    if not link:
        return ""
    frag = link.rsplit("#", 1)[-1] if "#" in link else ""
    return frag or link.rstrip("/").rsplit("-", 1)[-1]


def _as_list(v):
    if isinstance(v, list):
        return v
    if isinstance(v, str) and v.strip().startswith("["):
        try:
            return ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return []
    return []


def _price(x) -> float | None:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if f > 1.0 else None


def parse_records(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Raw OddsHarvester records -> (match rows, per-book set price rows)."""
    matches, quotes = [], []
    for r in records:
        oid = op_id_of(r.get("match_link", ""))
        if not oid:
            continue
        date = pd.to_datetime(str(r.get("match_date", ""))[:10], errors="coerce")
        if pd.isna(date):
            continue
        matches.append({
            "op_id": oid, "link": r.get("match_link"),
            "play_date": int(date.strftime("%Y%m%d")),
            "league": r.get("league_name"), "season": _int(r.get("season")),
            "home_name": r.get("home_team"), "away_name": r.get("away_team"),
            "home_sets": _int(r.get("home_score")),
            "away_sets": _int(r.get("away_score")),
            "partial": r.get("partial_results"),
        })
        for q in _as_list(r.get("match_winner_market")):
            period = PERIOD_NAME.get(q.get("period"))
            hp, ap = _price(q.get("player_1")), _price(q.get("player_2"))
            if period and hp and ap and q.get("bookmaker_name"):
                quotes.append({"op_id": oid, "book": q["bookmaker_name"],
                               "period": period, "home_price": hp,
                               "away_price": ap})
    return matches, quotes


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def read_cache_file(path: Path) -> list[dict]:
    if path.suffix == ".csv":
        return pd.read_csv(path).to_dict("records")
    data = json.loads(path.read_text())
    return data if isinstance(data, list) else []


# --------------------------------------------------------------------------
# checkpointing + ingestion
# --------------------------------------------------------------------------
def _mark(con, key: str, status: str, n: int = 0, note: str = "") -> None:
    con.execute(
        "INSERT OR REPLACE INTO op_scrape_log(kind,key,status,n_rows,note,done_at) "
        "VALUES ('event',?,?,?,?,?)",
        (key, status, n, note[:200],
         datetime.now(timezone.utc).isoformat(timespec="seconds")))
    con.commit()


def _done(con) -> set[str]:
    return {r[0] for r in con.execute(
        "SELECT key FROM op_scrape_log WHERE kind='event' AND status='ok'")}


def ingest_file(path: Path, con) -> dict:
    """Parse one cached file into op_matches / set_quotes (idempotent)."""
    ms, qs = parse_records(read_cache_file(path))
    con.executemany(
        "INSERT INTO op_matches(op_id,link,play_date,league,season,home_name,"
        "away_name,home_sets,away_sets,partial,source_file) "
        "VALUES(:op_id,:link,:play_date,:league,:season,:home_name,:away_name,"
        ":home_sets,:away_sets,:partial,:src) "
        "ON CONFLICT(op_id) DO UPDATE SET partial=excluded.partial, "
        "home_sets=excluded.home_sets, away_sets=excluded.away_sets",
        [{**m, "src": path.name} for m in ms])
    con.executemany(
        "INSERT OR REPLACE INTO set_quotes(op_id,book,period,home_price,away_price) "
        "VALUES(:op_id,:book,:period,:home_price,:away_price)", qs)
    con.commit()
    return {"matches": len(ms), "quotes": len(qs),
            "matches_with_prices": len({q["op_id"] for q in qs})}


def cache_path(league: str, season: int | str, period: int) -> Path:
    return CACHE / league / str(season) / f"period{period}.json"


def scrape_event(league: str, season: int | str, period: int = 1, *,
                 full: bool = True, force: bool = False) -> dict:
    """Scrape one league-season-period, cache the raw JSON, checkpoint, ingest.

    `full` asks for every bookmaker (`--full-scrape`); the default preview mode
    returned a single book (bet365.us) and empty prices on about a third of
    matches in the first pilot, which is useless for a consensus price.
    """
    key = f"{league}|{season}|{period}"
    out = cache_path(league, season, period)
    con = init_db()
    try:
        if key in _done(con) and not force:
            return {"key": key, "skipped": "already done"}
        if not BIN.exists():
            raise FileNotFoundError(f"OddsHarvester not found at {BIN}; set ODDSHARVESTER_BIN")
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp.json")
        tmp.unlink(missing_ok=True)
        cmd = [str(BIN), "historic", "-s", "tennis", "-l", league,
               "--season", str(season), "-m", "match_winner",
               "--period", PERIOD_FLAG[period], "--headless", "-f", "json",
               "-o", str(tmp), "--request-delay", str(REQUEST_DELAY),
               "--full-scrape" if full else "--preview-only"]
        t0 = time.time()
        try:
            with single_instance("oddsportal-scrape"):
                proc = subprocess.run(cmd, capture_output=True, text=True,
                                      timeout=TIMEOUT)
        except AlreadyRunning as exc:
            raise SystemExit(f"refusing to start: {exc}")
        except subprocess.TimeoutExpired:
            _mark(con, key, "error", 0, "timeout")
            return {"key": key, "status": "error", "note": "timeout"}
        n = len(read_cache_file(tmp)) if tmp.exists() else 0
        if proc.returncode != 0 or n == 0:
            note = (f"rc={proc.returncode}, 0 records: throttled/blocked or no such "
                    f"event" if n == 0 else f"rc={proc.returncode}")
            _mark(con, key, "error", n, note)
            return {"key": key, "status": "error", "note": note,
                    "tail": proc.stderr[-300:]}
        tmp.replace(out)
        res = ingest_file(out, con)
        _mark(con, key, "ok", res["matches"], f"{res['matches_with_prices']} priced, "
              f"{time.time() - t0:.0f}s")
        return {"key": key, "status": "ok", **res}
    finally:
        con.close()


def ingest_cache() -> dict:
    """Re-parse every cached file. Costs no requests."""
    con = init_db()
    try:
        tot = {"files": 0, "matches": 0, "quotes": 0}
        for f in sorted(CACHE.glob("*/*/period*.json")):
            r = ingest_file(f, con)
            tot["files"] += 1
            tot["matches"] += r["matches"]
            tot["quotes"] += r["quotes"]
        return tot
    finally:
        con.close()


# --------------------------------------------------------------------------
# resolution -- same pair + date-window approach as odds_te_hist
# --------------------------------------------------------------------------
def resolve_matches(con=None) -> dict:
    """Fill `match_id`, `home_is_winner` and `set1_agree` on op_matches.

    Pair of surname keys, order-independent, matched inside [-1, +21] days of
    the event's start (see odds_te_hist.resolve_matches for why one-sided).
    `set1_agree` is the hand-check made automatic: OddsPortal's own set-1 games
    against ours, oriented through the join. A join that pairs the right names
    to the wrong match shows up as disagreement, so the rate is the quality
    figure for the join and rows that disagree are excluded downstream.
    """
    from tennis.ingest.odds_live import surname_key

    close = con is None
    con = con or init_db()
    try:
        op = pd.read_sql(
            "SELECT op_id, play_date, home_name, away_name, partial FROM op_matches "
            "WHERE match_id IS NULL AND home_name NOT LIKE '%/%' "
            "AND away_name NOT LIKE '%/%'", con)
        if op.empty:
            return {"resolved": 0, "note": "nothing unresolved"}
        m = pd.read_sql(
            """SELECT m.match_id, m.tourney_date, m.score,
                      pw.name AS w_name, pl.name AS l_name
               FROM matches m
               JOIN players pw ON pw.player_id = m.winner_id
               JOIN players pl ON pl.player_id = m.loser_id
               WHERE m.tour = 'atp'""", con)
        op["kh"] = op.home_name.map(surname_key)
        op["ka"] = op.away_name.map(surname_key)
        op["pair"] = [tuple(sorted(x)) for x in zip(op.kh, op.ka)]
        m["kw"] = m.w_name.map(surname_key)
        m["kl"] = m.l_name.map(surname_key)
        m["pair"] = [tuple(sorted(x)) for x in zip(m.kw, m.kl)]
        j = op.merge(m, on="pair", how="inner")
        j["gap"] = (pd.to_datetime(j.play_date.astype(str), format="%Y%m%d")
                    - pd.to_datetime(j.tourney_date.astype(str), format="%Y%m%d")
                    ).dt.days
        j = j[j.gap.between(-1, 21)].sort_values("gap")
        j = j.drop_duplicates("op_id").drop_duplicates("match_id")
        j["home_is_winner"] = (j.kh == j.kw).astype(int)

        agree = []
        for r in j.itertuples():
            ours = set_winners(r.score)
            theirs = parse_partial(r.partial)
            if not ours or not theirs:
                agree.append(None)
                continue
            home_won_s1 = theirs[0][0] > theirs[0][1]
            agree.append(int(ours[0] == int(home_won_s1 == bool(r.home_is_winner))))
        j["set1_agree"] = agree
        con.executemany(
            "UPDATE op_matches SET match_id=?, home_is_winner=?, set1_agree=? "
            "WHERE op_id=?",
            [(r.match_id, int(r.home_is_winner),
              None if r.set1_agree is None else int(r.set1_agree), r.op_id)
             for r in j.itertuples()])
        con.commit()
        known = j.set1_agree.dropna()
        return {"candidates": len(op), "resolved": len(j),
                "rate_pct": round(len(j) / len(op) * 100, 1),
                "set1_agree_pct": round(float(known.mean()) * 100, 1) if len(known) else None}
    finally:
        if close:
            con.close()


def status() -> dict:
    con = connect()
    try:
        q = lambda s: con.execute(s).fetchone()[0]  # noqa: E731
        return {
            "events_ok": q("SELECT COUNT(*) FROM op_scrape_log WHERE status='ok'"),
            "events_error": q("SELECT COUNT(*) FROM op_scrape_log WHERE status='error'"),
            "matches": q("SELECT COUNT(*) FROM op_matches"),
            "matches_priced": q("SELECT COUNT(DISTINCT op_id) FROM set_quotes"),
            "resolved": q("SELECT COUNT(*) FROM op_matches WHERE match_id IS NOT NULL"),
            "set1_agree_pct": con.execute(
                "SELECT ROUND(100.0*AVG(set1_agree),1) FROM op_matches "
                "WHERE set1_agree IS NOT NULL").fetchone()[0],
            "quotes": q("SELECT COUNT(*) FROM set_quotes"),
        }
    finally:
        con.close()


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="OddsPortal set-winner prices")
    ap.add_argument("--event", nargs=2, metavar=("LEAGUE", "SEASON"),
                    help="e.g. atp-shanghai 2024")
    ap.add_argument("--periods", default="1", help="comma list of 1,2")
    ap.add_argument("--preview", action="store_true",
                    help="visible books only (fast, usually one book)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--ingest-cache", action="store_true")
    ap.add_argument("--resolve", action="store_true")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    if a.event:
        for p in (int(x) for x in a.periods.split(",")):
            print(scrape_event(a.event[0], a.event[1], p,
                               full=not a.preview, force=a.force))
    elif a.ingest_cache:
        print(ingest_cache())
    elif a.resolve:
        print(resolve_matches())
    elif a.status:
        for k, v in status().items():
            print(f"{k}: {v}")
    else:
        ap.print_help()
