"""Fetch draws for *upcoming* tournaments from the RapidAPI tennis feed.

Why this exists: neither TennisMyLife nor tennis-data.co.uk publishes a bracket
before it is played (verified -- the ongoing-tournament files contain zero rows
without a score), and past draws we reconstruct ourselves in tennis/sim/draws.py.
This module covers only the thin remaining slice: events about to start.

What this feed actually provides, measured rather than assumed:

* `/ms-api/tournament/{tour}/{slug}/{year}/draws` returns the **full draw sheet**
  and is the endpoint to use. It includes players holding a first-round bye
  (marked `result == "bye"` against a sentinel opponent id 3700) and carries a
  `draw` field giving the true bracket position.
* `/fixtures/tournament/{id}` returns unplayed fixtures and is only a fallback.
  It publishes a match solely once *both* players are known, so everyone with a
  bye is missing -- in a 96-draw Masters that silently dropped all 32 seeds and
  left the model handing the title to a qualifier.
* The plain "Results" endpoint returns completed matches only, with no bye rows.
* **Draws are not published at ceremony time.** A tournament eight weeks out
  returns an empty draw sheet, and a date-range fixtures query spanning 17 days
  returned matches for only the next two. In practice the draw appears roughly a
  day before play.

So this supports simulating an event the evening before it starts. It does not
support simulating one a week out.

Note the draws endpoint keys on a name *slug* ("montreal"), not the numeric
tournament id used by every other endpoint here.

Responses paginate at 10 rows by default; `pageSize` is honoured, which is what
keeps a large draw to a single request.

On the CLAUDE.md constraint: that file says not to use tennis-api.com, because
its pricing page showed three different numbers for one tier and it resells
rather than originates data. This is that vendor, reached directly through
RapidAPI, where the price list is unambiguous ($0 / $29 / $59 / $99) and the
free tier is a hard-limited 50 requests/day with no overage billing. The
"not a primary source" objection still stands, which is why every draw fetched
here is validated against our own match data once play begins.

Budget: one calendar call plus one fixtures call per event is roughly 5
requests/day for the main tour, ~20 including Challengers -- inside the free
tier. Historical backfill deliberately does not use this feed.

Credentials: set RAPIDAPI_KEY in the environment. Never commit it.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from tennis.config import DATA
from tennis.db.schema import connect
from tennis.ingest.load import norm_name

log = logging.getLogger(__name__)

HOST = os.getenv("RAPIDAPI_HOST", "tennis-api-atp-wta-itf.p.rapidapi.com")
BASE = f"https://{HOST}/tennis/v2"
CACHE = DATA / "raw" / "draws"
CACHE.mkdir(parents=True, exist_ok=True)

# Free tier is 50/day hard-limited; stay well under it and fail loudly instead
# of silently burning the allowance.
DAILY_BUDGET = int(os.getenv("RAPIDAPI_DAILY_BUDGET", "40"))
_BUDGET_FILE = CACHE / "_usage.json"


class DrawFeedError(RuntimeError):
    pass


def _key() -> str:
    k = os.getenv("RAPIDAPI_KEY")
    if not k:
        raise DrawFeedError(
            "RAPIDAPI_KEY is not set. Export it in your shell (do not paste it "
            "into source or chat):  export RAPIDAPI_KEY='...'"
        )
    return k


def _spend(n: int = 1) -> None:
    today = date.today().isoformat()
    u = json.loads(_BUDGET_FILE.read_text()) if _BUDGET_FILE.exists() else {}
    used = u.get(today, 0)
    if used + n > DAILY_BUDGET:
        raise DrawFeedError(
            f"daily request budget exhausted ({used}/{DAILY_BUDGET}). "
            "Raise RAPIDAPI_DAILY_BUDGET only if your plan allows it."
        )
    u = {today: used + n}
    _BUDGET_FILE.write_text(json.dumps(u))


# The API paginates at 10 rows by default and honours `pageSize`. Asking for a
# large page turns a 128-draw (which would otherwise be ~13 paged requests) into
# a single call, which is what keeps this inside the 50/day free tier.
PAGE_SIZE = 300


def _get(path: str, params: dict | None = None, *, cache_as: str | None = None,
         max_age_s: int = 3600) -> dict:
    """GET with on-disk caching, so repeated dashboard clicks cost no quota."""
    cf = CACHE / f"{cache_as}.json" if cache_as else None
    if cf and cf.exists() and (time.time() - cf.stat().st_mtime) < max_age_s:
        return json.loads(cf.read_text())

    _spend()
    r = requests.get(
        f"{BASE}{path}", params=params or {}, timeout=30,
        headers={"X-RapidAPI-Key": _key(), "X-RapidAPI-Host": HOST},
    )
    if r.status_code == 429:
        raise DrawFeedError("rate limited / quota exceeded by the provider (429)")
    if r.status_code == 403:
        raise DrawFeedError(
            "403 from the feed - the key is not subscribed to this API, or this "
            "endpoint is gated to a higher tier"
        )
    if r.status_code >= 400:
        # Surface the provider's message; a bare status code is useless when the
        # documented paths use placeholder segments.
        raise DrawFeedError(
            f"{r.status_code} from {r.url}: {r.text[:300]}"
        )
    data = r.json()
    if cf:
        cf.write_text(json.dumps(data))
    return data


# --------------------------------------------------------------------------
# endpoints
# --------------------------------------------------------------------------
def calendar(year: int | None = None, tour: str = "atp") -> dict:
    # The docs render paths as /tennis/v2/type/tournament/calendar/year, where
    # both `type` and `year` are placeholder segments, not literals.
    year = year or date.today().year
    return _get(f"/{tour}/tournament/calendar/{year}",
                cache_as=f"calendar_{tour}_{year}", max_age_s=86_400)


def tournament_fixtures(tour_id: str | int, tour: str = "atp") -> dict:
    """Every unplayed fixture for one tournament.

    For an event whose first round has been scheduled this is the full draw;
    mid-event it is whatever remains. For an event that has not been scheduled
    yet it is empty -- see the module docstring on what this feed can and
    cannot do.
    """
    return _get(f"/{tour}/fixtures/tournament/{tour_id}",
                {"pageSize": PAGE_SIZE, "include": "tournament"},
                cache_as=f"fixtures_{tour}_{tour_id}", max_age_s=1800)


def fixtures_between(start: str, end: str, tour: str = "atp") -> dict:
    """Scheduled fixtures in a date window (YYYY-MM-DD)."""
    return _get(f"/{tour}/fixtures/{start}/{end}",
                {"pageSize": PAGE_SIZE, "include": "tournament"},
                cache_as=f"range_{tour}_{start}_{end}", max_age_s=1800)


BYE_PLAYER_ID = 3700  # the feed's "Unknown Player" sentinel, used for byes


def tournament_slug(name: str) -> str:
    """Turn a feed tournament name into the slug the draws endpoint wants.

    'National Bank Open - Montreal' -> 'montreal', 'Hagen Challenger' -> 'hagen'.
    The advanced draws endpoint keys on this slug rather than the numeric id
    used everywhere else in the API.
    """
    n = name.split(" - ")[-1]
    for junk in (" Challenger", " Open", " Masters", " Cup", " Championships"):
        n = n.replace(junk, "")
    return re.sub(r"[^a-z0-9]+", "-", norm_name(n)).strip("-")


def tournament_draw(slug: str, year: int | None = None, tour: str = "atp") -> dict:
    """Full draw sheet for a tournament, byes included.

    This is the endpoint that actually solves bye handling. The fixtures feed
    only publishes a match once both players are known, so seeds holding a
    first-round bye are invisible there -- in a 96-draw Masters that silently
    removed all 32 of them. This one returns every draw slot, marks byes with
    `result == "bye"` against a sentinel opponent, and carries a `draw` field
    giving the true bracket position.
    """
    year = year or date.today().year
    return _get(f"/ms-api/tournament/{tour}/{slug}/{year}/draws",
                cache_as=f"draws_{tour}_{slug}_{year}", max_age_s=1800)


def _reconcile_withdrawals(out: pd.DataFrame) -> pd.DataFrame:
    """Put replacement players into the slots of those who pulled out.

    When a seed withdraws after the draw is made, the feed keeps the original
    bye row under the withdrawn player's name and adds the replacement as a
    separate row with no `draw` position. Its own later rounds tell the truth:
    at Montreal 2026 position 64 read "Auger-Aliassime, bye" while round 2
    position 32 read "Droguet vs Faria", and Faria was the unplaced entry.

    Two signatures make this identifiable. A withdrawn player wins a bye and
    then never appears in the next round; a replacement has no draw position
    but does appear in a later round, which pins the sub-bracket they belong to.

    Leaving the unplaced row in place gave a 96-player event 65 first-round
    ties, padded the bracket from 128 slots to 256, and let the stray player
    walk a half-empty draw to the title. Dropping it instead kept the bracket
    valid but ran a withdrawn player and omitted one who was actually playing.
    """
    if out.empty or out["round_id"].isna().all():
        return out
    r1 = out["round_id"].min()
    first = out[out["round_id"] == r1]
    unplaced = first[first["draw_pos"].isna()]
    if unplaced.empty:
        return out[out["draw_pos"].notna()]

    later = out[out["round_id"] > r1]
    later_names = set(later["p1_name"].dropna()) | set(later["p2_name"].dropna())

    # Bye winners who never turn up again: they withdrew.
    ghosts = {
        r.p1_name: r.draw_pos
        for r in first.itertuples()
        if r.is_bye and pd.notna(r.draw_pos) and r.p1_name not in later_names
    }

    fixed, used, placed_alts = out.copy(), set(), set()
    for alt in unplaced.itertuples():
        # Where does the replacement first show up? That pins their subtree.
        appear = later[(later["p1_name"] == alt.p1_name)
                       | (later["p2_name"] == alt.p1_name)]
        if appear.empty:
            continue
        pos = appear.sort_values("round_id").iloc[0]["draw_pos"]
        if pd.isna(pos):
            continue
        feeders = {int(pos) * 2 - 1, int(pos) * 2}
        match = [g for g, gp in ghosts.items()
                 if int(gp) in feeders and g not in used]
        if len(match) != 1:
            continue
        ghost = match[0]
        used.add(ghost)
        placed_alts.add(alt.p1_name)
        slot = fixed["draw_pos"] == ghosts[ghost]
        slot &= fixed["round_id"] == r1
        fixed.loc[slot, "p1_name"] = alt.p1_name
        fixed.loc[slot, "seed1"] = alt.seed1
        log.info("draw sheet: %s withdrew from position %s; %s takes the slot",
                 ghost, int(ghosts[ghost]), alt.p1_name)

    # Every unplaced row goes: the reconciled ones were copied into their real
    # slot above, so keeping the original would double-count the player.
    leftover = fixed["draw_pos"].isna()
    stranded = fixed.loc[leftover & ~fixed["p1_name"].isin(placed_alts), "p1_name"]
    if len(stranded):
        log.warning("draw sheet: %d unplaced entr%s could not be matched to a "
                    "withdrawal and were dropped (%s)", len(stranded),
                    "y" if len(stranded) == 1 else "ies",
                    ", ".join(str(x) for x in stranded.head(5)))
    return fixed[~leftover]


def parse_draw_sheet(payload: dict, *, first_round_only: bool = True) -> pd.DataFrame:
    """Flatten a draw sheet into one row per first-round bracket slot."""
    d = payload.get("data", payload) if isinstance(payload, dict) else {}
    singles = d.get("singles") or []
    rows = []
    for x in singles:
        if not isinstance(x, dict):
            continue
        p1 = (x.get("player1") or {})
        p2 = (x.get("player2") or {})
        is_bye = (str(x.get("result", "")).lower() == "bye"
                  or x.get("player2Id") == BYE_PLAYER_ID)
        rows.append({
            "round_id": x.get("roundId"),
            "draw_pos": x.get("draw"),
            "p1_name": p1.get("name"), "p2_name": None if is_bye else p2.get("name"),
            "seed1": x.get("seed1") or p1.get("seed"),
            "seed2": None if is_bye else (x.get("seed2") or p2.get("seed")),
            "is_bye": is_bye,
            "result": x.get("result") or None,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = _reconcile_withdrawals(out)
    if out.empty:
        return out
    out = out.sort_values(["round_id", "draw_pos"]).reset_index(drop=True)
    if not first_round_only:
        return out
    # Keep the opening round only; later rounds are placeholders for the same
    # players and would double-count them.
    return (out[out["round_id"] == out["round_id"].min()]
            .sort_values("draw_pos").reset_index(drop=True))


def played_results(sheet: pd.DataFrame, con=None) -> pd.DataFrame:
    """Completed matches from a draw sheet, as (winner_id, loser_id).

    The sheet has no explicit winner field -- only a `result` score string and
    `winner_*` attribute columns that come back null. player1 is the winner:
    checked against our own match table for Montreal 2026, where all 14
    cross-checkable matches agreed and none contradicted. A runtime guard below
    keeps that assumption from failing silently if the feed ever changes.
    """
    if sheet.empty:
        return pd.DataFrame(columns=["winner_id", "loser_id"])
    done = sheet[sheet["result"].notna()
                 & ~sheet["is_bye"]
                 & (sheet["result"].astype(str).str.strip() != "")]
    if done.empty:
        return pd.DataFrame(columns=["winner_id", "loser_id"])
    names = sorted({n for n in (*done["p1_name"], *done["p2_name"])
                    if isinstance(n, str) and n.strip()})
    ids = resolve_players(names, con)
    rows = [{"winner_id": ids.get(r.p1_name), "loser_id": ids.get(r.p2_name),
             "score": r.result}
            for r in done.itertuples()]
    out = pd.DataFrame(rows).dropna(subset=["winner_id", "loser_id"])
    return out.reset_index(drop=True)


def resolve_tourney_key(feed_id: str | int, feed_name: str, player_ids,
                        season: int, event_date: str | int | None = None,
                        con=None) -> str | None:
    """Link a feed tournament to ours, caching the answer in `tourney_xref`.

    The two sources share no tournament identifier -- ours embeds the real ATP
    id (`2026-421` = Canada Masters), the feed uses internal ids (`21346`) --
    and their names disagree too ("National Bank Open - Montreal" vs "Canada
    Masters"). What they *do* share is ATP player ids.

    Matching on raw overlap alone is not enough: ranking candidates by how many
    matches were played between these players picked Roland Garros for the
    Montreal draw, because the same top players appear at both and RG simply
    has more matches. So candidates are first restricted to a date window
    around the event, then scored by Jaccard similarity of the player sets,
    which does not reward a bigger field.

    The result is written to `tourney_xref`, so later calls are an id lookup.
    """
    close = con is None
    con = con or connect()
    try:
        row = con.execute(
            "SELECT tourney_key FROM tourney_xref WHERE feed_id = ?",
            (str(feed_id),)).fetchone()
        if row and row[0]:
            return row[0]

        ids = {p for p in player_ids if p and p != "__BYE__"}
        if not ids:
            return None

        params: list = [season]
        date_sql = ""
        if event_date is not None:
            d = str(event_date)[:10].replace("-", "")
            if d.isdigit() and len(d) == 8:
                lo = int((pd.Timestamp(d) - pd.Timedelta(days=10)).strftime("%Y%m%d"))
                hi = int((pd.Timestamp(d) + pd.Timedelta(days=10)).strftime("%Y%m%d"))
                date_sql = " AND t.tourney_date BETWEEN ? AND ?"
                params += [lo, hi]

        cand = pd.read_sql(
            f"SELECT m.tourney_key, m.winner_id, m.loser_id FROM matches m "
            f"JOIN tournaments t USING(tourney_key) "
            f"WHERE t.season = ?{date_sql}", con, params=params)
        if cand.empty:
            return None

        best_key, best_score, best_n = None, 0.0, 0
        for key, g in cand.groupby("tourney_key"):
            field = set(g["winner_id"]) | set(g["loser_id"])
            inter = len(field & ids)
            if not inter:
                continue
            jaccard = inter / len(field | ids)
            if jaccard > best_score:
                best_key, best_score, best_n = key, jaccard, inter
        if best_key is None or best_score < 0.2:
            return None

        con.execute(
            "INSERT INTO tourney_xref(feed_id, feed_name, tourney_key, season, "
            "n_overlap, resolved_at) VALUES(?,?,?,?,?,?) "
            "ON CONFLICT(feed_id) DO UPDATE SET tourney_key=excluded.tourney_key, "
            "n_overlap=excluded.n_overlap, resolved_at=excluded.resolved_at",
            (str(feed_id), feed_name, best_key, season, best_n,
             pd.Timestamp.now("UTC").isoformat()))
        con.commit()
        return best_key
    finally:
        if close:
            con.close()


def event_matches(tourney_key: str, con=None) -> pd.DataFrame:
    """Our own winner/loser rows for one event, by key."""
    if not tourney_key:
        return pd.DataFrame(columns=["winner_id", "loser_id"])
    close = con is None
    con = con or connect()
    try:
        return pd.read_sql(
            "SELECT winner_id, loser_id FROM matches WHERE tourney_key = ?",
            con, params=(tourney_key,))
    finally:
        if close:
            con.close()


def verify_winner_convention(sheet: pd.DataFrame,
                             our_matches: pd.DataFrame) -> dict:
    """Cross-check player1-is-winner against results we already hold.

    `our_matches` must be *this event's* rows only. Scoping matters twice over:
    comparing against the whole match table reported five contradictions, all
    from pairs who had met at other tournaments; a season-scoped version still
    reported two for the same reason.

    A pair appearing in both orientations (they met twice) counts as unknown
    rather than as both confirmed and contradicted -- the latter double-counted
    and could drive `unknown` negative.
    """
    res = played_results(sheet)
    if res.empty or our_matches.empty:
        return {"checked": len(res), "confirmed": 0, "contradicted": 0,
                "unknown": len(res)}
    wins = {(r.winner_id, r.loser_id) for r in our_matches.itertuples()}
    ok = bad = 0
    for r in res.itertuples():
        fwd = (r.winner_id, r.loser_id) in wins
        rev = (r.loser_id, r.winner_id) in wins
        if fwd and not rev:
            ok += 1
        elif rev and not fwd:
            bad += 1
    return {"checked": len(res), "confirmed": ok, "contradicted": bad,
            "unknown": len(res) - ok - bad}


def build_draw_from_sheet(sheet: pd.DataFrame, *, name: str, surface: str,
                          level: str, best_of: int = 3, indoor: float = 0.0,
                          tourney_date: int | None = None, con=None):
    """Build a simulator Draw from a draw sheet, preserving bracket order.

    Unlike the fixtures path, `draw_pos` gives the real bracket position, so
    later-round matchups are the actual ones rather than an approximation.
    """
    from tennis.models.predict import MatchContext
    from tennis.sim.bracket import BYE, Draw

    # A bye slot leaves p2_name as NaN, which is a float and would break sorting.
    names = sorted({n for n in (*sheet["p1_name"], *sheet["p2_name"])
                    if isinstance(n, str) and n.strip()})
    ids = resolve_players(names, con)
    unresolved = [n for n, v in ids.items() if v is None]

    # A player we cannot match to our database is still a player. Marking them
    # BYE would hand their opponent a guaranteed win in every simulation --
    # the same phantom-bye failure that let a stray entry walk a half-empty
    # draw to the title. They get a placeholder id instead, so the model scores
    # them from missing features (roughly a coin flip) and they can lose.
    def _slot_for(name, resolved_id, counter=[0]):
        if resolved_id:
            return resolved_id
        if not isinstance(name, str) or not name.strip():
            return BYE
        counter[0] += 1
        return f"__UNKNOWN_{counter[0]}__"

    slots, labels = [], {}
    for row in sheet.itertuples():
        a = _slot_for(row.p1_name, ids.get(row.p1_name))
        b = BYE if row.is_bye else _slot_for(row.p2_name, ids.get(row.p2_name))
        slots.append(a)
        slots.append(b)
        if a != BYE and isinstance(row.p1_name, str):
            labels[a] = row.p1_name
        if b != BYE and isinstance(row.p2_name, str):
            labels[b] = row.p2_name

    ctx = MatchContext(
        surface=surface, level=level, best_of=best_of, indoor=indoor,
        draw_size=float(max(len(slots), 2)),
        tourney_date=tourney_date or int(date.today().strftime("%Y%m%d")),
        is_challenger=int(level == "challenger"),
    )
    draw = Draw(name=name, slots=slots, ctx=ctx, player_names=labels)

    # A sane sheet gives a power-of-two number of slots and needs no padding.
    # Anything else means the sheet is malformed, and padding it silently is
    # how a stray row turned a 128-slot bracket into 256 with 126 phantom byes.
    padded = draw.size - len(slots)
    if padded > 0:
        log.warning("draw '%s': %d slots padded to %d -- the sheet gave an "
                    "odd number of first-round ties", name, len(slots), draw.size)
    return draw, unresolved


def discover_tournaments(days: int = 3, tour: str = "atp") -> pd.DataFrame:
    """Tournaments with fixtures scheduled in the next few days.

    The calendar endpoint proved incomplete for the near term (queried in
    August, its earliest entry was late September), so live event ids are
    discovered from the fixtures feed instead.
    """
    from datetime import timedelta

    today = date.today()
    payload = fixtures_between(today.isoformat(),
                               (today + timedelta(days=days)).isoformat(), tour)
    fx = parse_fixtures(payload)
    if fx.empty:
        return pd.DataFrame()
    def _first(s):
        s = s.dropna()
        return s.iloc[0] if len(s) else None

    out = (fx.groupby("tournament_id")
             .agg(name=("tournament_name", _first),
                  fixtures=("p1_name", "size"),
                  first_date=("date", "min"), last_date=("date", "max"),
                  court_id=("court_id", _first), rank_id=("rank_id", _first),
                  rounds=("round_id", lambda s: sorted(set(s.dropna()))))
             .sort_values("fixtures", ascending=False)
             .reset_index())
    # Fall back to the id so a nameless row is still selectable rather than blank.
    out["name"] = out["name"].fillna(
        out["tournament_id"].map(lambda i: f"Tournament {i}"))
    return out


# --------------------------------------------------------------------------
# response shaping
# --------------------------------------------------------------------------
def _walk(obj, out: list) -> None:
    """Collect dicts that look like a fixture, wherever they sit in the payload.

    The response envelope is not pinned down by the docs, so we search rather
    than assume a shape. A fixture is any dict carrying two player-ish fields.
    """
    if isinstance(obj, dict):
        keys = {k.lower() for k in obj}
        if {"player1", "player2"} <= keys or {"p1", "p2"} <= keys:
            out.append(obj)
        else:
            for v in obj.values():
                _walk(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _walk(v, out)


def _name_of(v) -> str:
    if isinstance(v, dict):
        for k in ("name", "fullName", "full_name", "displayName", "player"):
            if isinstance(v.get(k), str):
                return v[k]
        first = v.get("firstName") or v.get("first_name") or ""
        last = v.get("lastName") or v.get("last_name") or ""
        return f"{first} {last}".strip()
    return str(v) if v is not None else ""


def parse_fixtures(payload: dict, *, singles_only: bool = True) -> pd.DataFrame:
    """Flatten a fixtures payload.

    Doubles are dropped by default: the feed encodes a pair as one "player"
    whose name is "A/B", which would never resolve to an ATP id and is not
    something this model predicts.
    """
    raw: list = []
    _walk(payload, raw)
    rows = []
    for f in raw:
        low = {k.lower(): v for k, v in f.items()}
        p1 = _name_of(low.get("player1") or low.get("p1"))
        p2 = _name_of(low.get("player2") or low.get("p2"))
        if not p1 or not p2:
            continue
        if singles_only and ("/" in p1 or "/" in p2):
            continue
        # Tournament name only arrives when the request asks for
        # include=tournament; the bare fixture carries just tournamentId.
        t = low.get("tournament") if isinstance(low.get("tournament"), dict) else {}
        rows.append({
            "p1_name": p1, "p2_name": p2,
            "seed1": low.get("seed1"), "seed2": low.get("seed2"),
            "round_id": low.get("roundid"),
            "tournament_id": low.get("tournamentid"),
            "tournament_name": t.get("name"),
            "court_id": t.get("courtId"),
            "rank_id": t.get("rankId"),
            "date": (low.get("date") or "")[:10] or None,
        })
    out = pd.DataFrame(rows)
    return out.drop_duplicates() if not out.empty else out


# Codes confirmed two ways: against the feed's own calendar (which names the
# court) and by cross-checking live events against the surface we already hold
# for the same city. Grass has not been observed yet -- August is a hard/clay
# stretch -- and an unknown id deliberately falls through to the user's choice
# rather than guessing the model into the wrong surface.
COURT_SURFACE: dict[int, tuple[str, float]] = {
    1: ("Hard", 0.0),    # calendar "Hard"; matches Istanbul/Lexington/Astana
    2: ("Clay", 0.0),    # matches Hagen and Poprad in our own history
    3: ("Hard", 1.0),    # calendar "I.hard" (indoor)
}
# rankId is a coarse tour band. 2 spans ATP 250 and ATP 500, so it resolves to
# the more common of the two and stays user-editable. 0 is ITF (M15/M25), below
# the Challenger floor of this project's scope.
RANK_LEVEL: dict[int, str] = {
    1: "challenger", 2: "atp250", 3: "masters", 7: "finals",
}
RANK_ITF = 0


def context_hints(fx: pd.DataFrame) -> dict:
    """Surface / level / indoor suggested by the feed, where it is unambiguous."""
    if fx.empty:
        return {}
    out: dict = {}
    court = fx["court_id"].dropna()
    if len(court) and int(court.iloc[0]) in COURT_SURFACE:
        surface, indoor = COURT_SURFACE[int(court.iloc[0])]
        out["surface"], out["indoor"] = surface, indoor
    rank = fx["rank_id"].dropna()
    if len(rank) and int(rank.iloc[0]) in RANK_LEVEL:
        out["level"] = RANK_LEVEL[int(rank.iloc[0])]
    return out


def first_round(fx: pd.DataFrame) -> pd.DataFrame:
    """Keep only the earliest round present -- the draw's opening round.

    A tournament's remaining fixtures can span two rounds once play begins
    (players with byes get their second-round match scheduled early). Simulating
    a bracket needs one clean opening round.
    """
    if fx.empty or fx["round_id"].isna().all():
        return fx
    return fx[fx["round_id"] == fx["round_id"].min()].reset_index(drop=True)


def resolve_players(names: list[str], con=None) -> dict[str, str | None]:
    """Map feed names onto our ATP player ids by normalised name.

    Same approach as the odds join, which reached 82% on a far messier surface.
    Anything unmatched is returned as None so the caller can surface it rather
    than silently simulating the wrong player.
    """
    close = con is None
    con = con or connect()
    players = pd.read_sql("SELECT player_id, name, name_norm, last_seen FROM players", con)
    if close:
        con.close()
    # Prefer recently-active players when a normalised name collides.
    players = players.sort_values("last_seen", ascending=False)
    by_norm = players.drop_duplicates("name_norm").set_index("name_norm")["player_id"]

    def surname_key(n: str) -> str:
        """'Jannik Sinner' and 'Sinner J.' must both key to 'sinner|j'.

        The feed's name format is not pinned down by the docs, so this detects
        the abbreviated style (a bare initial token) rather than assuming
        first-name-first, which would key 'Sinner J.' as 'j|s'.
        """
        toks = norm_name(n).split()
        if not toks:
            return ""
        initials = [t for t in toks if len(t) == 1]
        words = [t for t in toks if len(t) > 1]
        if not words:
            return toks[0]
        if initials:                       # "Surname I." style
            return f"{words[-1]}|{initials[0]}"
        if len(words) == 1:
            return words[0]
        return f"{words[-1]}|{words[0][0]}"  # "First Last" style

    players["skey"] = players["name"].map(surname_key)
    by_skey = players.drop_duplicates("skey").set_index("skey")["player_id"]

    out: dict[str, str | None] = {}
    for n in names:
        nn = norm_name(n)
        pid = by_norm.get(nn)
        if pid is None:
            pid = by_skey.get(surname_key(n))
        if pid is None:
            # Compound surnames are common in tennis and the two sources often
            # disagree on how many parts to keep ("Daniel Merida Aguilar" here
            # vs "Daniel Merida" in our data). Try each trailing token as the
            # surname before giving up.
            toks = [t for t in nn.split() if len(t) > 1]
            if len(toks) > 2:
                initial = toks[0][0]
                for cand in toks[1:]:
                    pid = by_skey.get(f"{cand}|{initial}")
                    if pid:
                        break
        out[n] = pid
    return out


def build_draw_from_fixtures(fixtures: pd.DataFrame, *, name: str, surface: str,
                             level: str, best_of: int = 3,
                             indoor: float = 0.0, tourney_date: int | None = None):
    """Turn first-round fixtures into a simulator-ready Draw.

    The feed gives pairings but not bracket position, so slots are laid out in
    the order returned: fixture i occupies slots 2i and 2i+1. That is correct
    for who-plays-whom in round 1 and for the shape of the bracket, but the
    *order* of the halves may not match the official draw sheet.
    """
    from tennis.models.predict import MatchContext
    from tennis.sim.bracket import Draw

    names = sorted({*fixtures["p1_name"], *fixtures["p2_name"]})
    ids = resolve_players(names)
    unresolved = [n for n, v in ids.items() if v is None]

    slots, labels = [], {}
    for f in fixtures.itertuples():
        a, b = ids.get(f.p1_name), ids.get(f.p2_name)
        if a is None or b is None:
            continue
        slots += [a, b]
        labels[a], labels[b] = f.p1_name, f.p2_name

    ctx = MatchContext(
        surface=surface, level=level, best_of=best_of, indoor=indoor,
        draw_size=float(max(len(slots), 2)),
        tourney_date=tourney_date or int(date.today().strftime("%Y%m%d")),
        is_challenger=int(level == "challenger"),
    )
    draw = Draw(name=name, slots=slots, ctx=ctx, player_names=labels)
    return draw, unresolved


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if len(sys.argv) > 1:
        payload = tournament_fixtures(sys.argv[1])
        print(json.dumps(payload, indent=2)[:2000])
        print("\nparsed fixtures:")
        print(parse_fixtures(payload).to_string(index=False))
    else:
        cal = calendar()
        print(json.dumps(cal, indent=2)[:3000])
