"""Clean the cached CSVs and load them into SQLite.

Cleaning decisions worth knowing:

* `tourney_date` is the *week* the event began, not the match date. Every match
  in a tournament shares it. Chronological order therefore comes from
  (tourney_date, round_idx, match_num), which is correct within an event.
* A handful of challenger rows carry a `tourney_id` whose year prefix is wrong
  (e.g. 2008 events filed under `2024-2205`). `tourney_date` is right in those
  rows, so season is always derived from the date, never from `tourney_id`.
* Retirements/walkovers/defaults are kept with a `status` label. They are valid
  winner-model rows (someone did win) but are flagged `totals_usable=0` so the
  totals and spread models never see a truncated scoreline as a real total.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from tennis.config import (
    CHALLENGER_LEVELS,
    LEVEL_MAP,
    RAW_ONGOING_ARCHIVE,
    RAW_TML,
    ROUND_ORDER,
    ARTIFACTS,
    START_YEAR,
    SURFACE_MAP,
)
from tennis.db.lock import exclusive_write
from tennis.db.schema import init_db, set_state
from tennis.ingest.parse import parse_score

log = logging.getLogger(__name__)

STAT_COLS = [
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_SvGms",
    "w_bpSaved", "w_bpFaced",
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_SvGms",
    "l_bpSaved", "l_bpFaced",
]
ROUND_IDX = {r: i for i, r in enumerate(ROUND_ORDER)}
_ID_DTYPES = {"winner_id": "string", "loser_id": "string", "tourney_id": "string"}


def norm_name(name: str | float | None) -> str:
    """Accent-stripped, lowercased, punctuation-free name for fuzzy joins."""
    if name is None or (isinstance(name, float) and name != name):
        return ""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace("-", " ").replace(".", " ").replace("'", "")
    return re.sub(r"\s+", " ", s).strip()


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", norm_name(s)).strip("-")


def _tag(df: pd.DataFrame, name: str, priority: int) -> pd.DataFrame:
    df["source_file"] = name
    df["is_challenger_file"] = int("challenger" in name)
    # Tour comes from the filename, the only place it is recorded. The two
    # tours never meet, so every rating pool downstream must stay separate;
    # mislabelling here would silently merge them.
    df["tour"] = "wta" if "wta" in name else "atp"
    # Lower wins a duplicate: the yearly file is the maintainer's settled
    # copy, the live ongoing file is next, an archived snapshot is last.
    df["source_priority"] = priority
    return df


def read_raw(include_ongoing: bool = True) -> pd.DataFrame:
    """Read every cached CSV into one frame, tagged with its source and tour."""
    frames = []
    for path in sorted(RAW_TML.glob("*.csv")):
        name = path.name
        if name.startswith("_"):
            continue
        is_ongoing = "ongoing" in name
        if is_ongoing and not include_ongoing:
            continue
        m = re.match(r"(\d{4})", name)
        if not is_ongoing and (not m or int(m.group(1)) < START_YEAR):
            continue
        # Ids are strings. Left to inference, a column with any blank (the 2026
        # WTA file has some) is read as float and "216016" becomes "216016.0",
        # which split a player into two identities.
        df = pd.read_csv(path, low_memory=False, dtype=_ID_DTYPES)
        if df.empty:
            continue
        frames.append(_tag(df, name, 1 if is_ongoing else 0))
    if include_ongoing:
        frames.extend(_read_ongoing_archive(frames))
    if not frames:
        raise RuntimeError(f"no CSVs found in {RAW_TML}")
    return pd.concat(frames, ignore_index=True)


def _read_ongoing_archive(live: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """Archived per-tournament snapshots of tournaments no longer listed.

    A tournament that has left the ongoing file but is not yet in the yearly
    one exists only here. Once either file carries it again the snapshot is
    redundant -- and stale, so it is skipped rather than merged.
    """
    seen = {(f["tour"].iat[0], t)
            for f in live for t in f["tourney_id"].dropna().unique()}
    out = []
    for path in sorted(RAW_ONGOING_ARCHIVE.glob("*/*.csv")):
        df = pd.read_csv(path, low_memory=False, dtype=_ID_DTYPES)
        if df.empty:
            continue
        df = _tag(df, f"{path.parent.name}.csv", 2)
        if (df["tour"].iat[0], df["tourney_id"].iat[0]) in seen:
            continue
        df["source_file"] = f"archive/{path.parent.name}/{path.name}"
        out.append(df)
    return out


# Hand-maintained merges: alias_id,canonical_id,note. Anything the automatic
# rule below will not touch (the ATP, where two ids with the same name are
# often two different people) goes here once someone has checked it.
ALIAS_CSV = Path(__file__).with_name("player_aliases.csv")
# Tours where same-name + same-country + never-in-one-event is trusted enough
# to merge unattended. The WTA re-issued ids in June 2026 (52 players, e.g.
# 214981 -> 324166), splitting each player's history in two; there, the pattern
# is unambiguous. The ATP's duplicates include genuinely different people
# (Dinko Dinev, SUI and BUL), so they are only ever reported.
AUTO_ALIAS_TOURS = ("wta",)


def find_aliases(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(applied, candidates): id merges to apply, and ATP-style ones to review.

    A pair is a duplicate when the normalised name, tour and country agree and
    the two ids never appear in the same event -- one person cannot enter a
    tournament twice. The id with more matches is canonical.
    """
    need = {"winner_id", "loser_id", "winner_name", "loser_name", "winner_ioc",
            "loser_ioc", "tourney_id", "tour"}
    cols = ["alias_id", "canonical_id", "tour", "reason"]
    empty = pd.DataFrame(columns=cols)
    applied = []
    if ALIAS_CSV.exists():
        man = pd.read_csv(ALIAS_CSV, dtype=str, comment="#")
        for r in man.itertuples():
            applied.append((r.alias_id, r.canonical_id, "", f"manual: {getattr(r, 'note', '')}"))
    cand = []
    if need <= set(df.columns):
        long = pd.concat([
            df[[f"{s}_id", f"{s}_name", f"{s}_ioc", "tourney_id", "tour"]]
              .set_axis(["id", "name", "ioc", "tid", "tour"], axis=1)
            for s in ("winner", "loser")], ignore_index=True)
        long["nn"] = long["name"].map(norm_name)
        # An id's name is the one it overwhelmingly carries. Single rows are
        # sometimes mislabelled upstream (id Z0AK, 63 rows "Tyler Zink", has one
        # "Alfredo Perez" row), and counting them invented duplicate pairs that
        # an unattended merge would have acted on.
        modal = long.groupby("id")["nn"].agg(lambda x: x.mode().iat[0])
        long = long[long["nn"] == long["id"].map(modal)]
        multi = long.groupby(["tour", "nn"])["id"].nunique()
        for (tour, nn) in multi[multi > 1].index:
            d = long[(long["tour"] == tour) & (long["nn"] == nn)]
            n = d["id"].value_counts()
            ioc = d.groupby("id")["ioc"].agg(lambda x: x.mode().iat[0] if x.notna().any() else None)
            ev = {k: set(v) for k, v in d.groupby("id")["tid"]}
            canon = n.index[0]
            merged_ev = set(ev[canon])
            for other in n.index[1:]:
                ok = ioc[other] == ioc[canon] and not (ev[other] & merged_ev)
                if not ok:
                    continue
                if tour in AUTO_ALIAS_TOURS:
                    applied.append((other, canon, tour, "auto: same name+country, never in one event"))
                    merged_ev |= ev[other]
                else:
                    cand.append((other, canon, tour, "candidate: same name+country, never in one event"))
    out = pd.DataFrame(applied, columns=cols) if applied else empty
    return out.drop_duplicates("alias_id"), (pd.DataFrame(cand, columns=cols) if cand else empty)


def clean(df: pd.DataFrame, aliases: pd.DataFrame | None = None) -> pd.DataFrame:
    df = df.copy()
    if "source_priority" not in df.columns:      # frames not built by read_raw
        df["source_priority"] = 0

    # ---- dates & season (from tourney_date, never from tourney_id) ---------
    df["tourney_date"] = pd.to_numeric(df["tourney_date"], errors="coerce")
    df = df[df["tourney_date"].notna()].copy()
    df["tourney_date"] = df["tourney_date"].astype("int64")
    df = df[(df["tourney_date"] > 19000101) & (df["tourney_date"] < 21000101)]
    df["season"] = df["tourney_date"] // 10000
    df = df[df["season"] >= START_YEAR].copy()

    # ---- categorical standardisation --------------------------------------
    df["surface"] = (
        df["surface"].astype("string").str.strip().str.lower().map(SURFACE_MAP)
    )
    df["level"] = df["tourney_level"].astype("string").str.strip().map(LEVEL_MAP)
    # Challenger files are authoritative for their own level regardless of the
    # tourney_level char, which is inconsistent in the older challenger seasons.
    df.loc[df["is_challenger_file"] == 1, "level"] = "challenger"
    df["level"] = df["level"].fillna("other")
    df["is_challenger"] = df["level"].isin(CHALLENGER_LEVELS).astype(int)

    ind = df["indoor"].astype("string").str.strip().str.upper().fillna("")
    df["indoor"] = np.where(ind == "I", 1.0, np.where(ind == "O", 0.0, np.nan))

    df["round"] = df["round"].astype("string").str.strip().str.upper()
    df["round_idx"] = df["round"].map(ROUND_IDX).astype("float")

    df["best_of"] = pd.to_numeric(df["best_of"], errors="coerce")
    df["draw_size"] = pd.to_numeric(df["draw_size"], errors="coerce")
    df["match_num"] = pd.to_numeric(df["match_num"], errors="coerce")

    for c in ["winner_id", "loser_id"]:
        df[c] = (df[c].astype("string").str.strip()
                 .str.replace(r"\.0+$", "", regex=True))
    df = df[df["winner_id"].notna() & df["loser_id"].notna()].copy()
    df = df[(df["winner_id"] != "") & (df["loser_id"] != "")]
    # Merge duplicate ids BEFORE anything is keyed on them: match_id, ratings
    # and form all hang off the id.
    if aliases is None:
        aliases, _ = find_aliases(df)
    if len(aliases):
        amap = dict(zip(aliases["alias_id"], aliases["canonical_id"]))
        df["winner_id"] = df["winner_id"].replace(amap)
        df["loser_id"] = df["loser_id"].replace(amap)
    df = df[df["winner_id"] != df["loser_id"]]  # corrupt self-matches

    for c in [
        "winner_age", "loser_age", "winner_rank", "loser_rank",
        "winner_rank_points", "loser_rank_points", "winner_ht", "loser_ht",
        "winner_seed", "loser_seed", "minutes", *STAT_COLS,
    ]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")

    # ---- score parsing ----------------------------------------------------
    parsed = [parse_score(s, b) for s, b in zip(df["score"], df["best_of"])]
    df["status"] = [p.status for p in parsed]
    df["w_sets"] = [p.w_sets for p in parsed]
    df["l_sets"] = [p.l_sets for p in parsed]
    df["w_games"] = [p.w_games for p in parsed]
    df["l_games"] = [p.l_games for p in parsed]
    df["total_games"] = [p.total_games for p in parsed]
    df["total_sets"] = [p.total_sets for p in parsed]
    df["game_margin"] = [p.game_margin for p in parsed]
    df["tiebreaks"] = [p.tiebreaks for p in parsed]
    df["totals_usable"] = [int(p.usable_for_totals) for p in parsed]

    # Walkovers never happened on court: no games, no stats, and they must not
    # feed serve/return form either.
    df.loc[df["status"] == "walkover", ["w_games", "l_games", "total_games",
                                        "total_sets", "game_margin"]] = 0

    # A match "has stats" when the *point* counters are populated. SvGms is
    # deliberately not required: the WTA files carry no service-game counts at
    # all before 2015, while their point counters run back to the early 2000s
    # at ~78%. Demanding SvGms discarded 19,846 WTA matches that have
    # everything the serve and return rates actually need, and gated the
    # feature builder off entirely for fifteen seasons. Measured against the
    # ATP the change is a no-op -- 12 matches -- because there the two counters
    # arrive together. Statistics that genuinely need SvGms (hold %, break %)
    # are guarded separately where they are computed.
    df["has_stats"] = (
        df[["w_svpt", "l_svpt", "w_1stWon", "l_1stWon",
            "w_2ndWon", "l_2ndWon"]].notna().all(axis=1)
        & (df["w_svpt"].fillna(0) > 0) & (df["l_svpt"].fillna(0) > 0)
    ).astype(int)

    # ---- keys -------------------------------------------------------------
    # The two file generations disagree about what tourney_date means. Archived
    # seasons stamp every match in an event with the week it began; the current
    # season stamps each match with the day it was actually played. Keying on
    # the date therefore shatters live events -- Roland Garros 2026 arrived as
    # 13 separate "tournaments", one per playing day, which breaks bracket
    # reconstruction exactly when we most want it.
    #
    # So events are keyed on (tourney_id, name, tour level) instead, with a
    # date-gap split as a safety net: a few malformed tourney_ids are reused
    # across years (one literal "Rome", and Davis Cup's D001 spans many venues),
    # and those must not collapse into a single event.
    df = df.sort_values(["tourney_id", "tourney_name", "is_challenger", "tourney_date"])
    grp = [df["tourney_id"].astype("string").fillna(""),
           df["tourney_name"].map(_slug), df["is_challenger"].astype(str)]
    base = grp[0] + "|" + grp[1] + "|" + grp[2]
    dt = pd.to_datetime(df["tourney_date"].astype(str), format="%Y%m%d", errors="coerce")
    gap_days = dt.groupby(base, observed=True).diff().dt.days.fillna(0)
    # A jump of more than 30 days inside one key means a different edition.
    df["_edition"] = (gap_days > 30).groupby(base, observed=True).cumsum().astype(int)
    df["tourney_key"] = base + "|" + df["_edition"].astype(str)

    # The event's date is when it began; the per-match date is kept separately
    # and is genuinely exact wherever the source varied it within an event.
    starts = df.groupby("tourney_key", observed=True)["tourney_date"].transform("min")
    n_dates = df.groupby("tourney_key", observed=True)["tourney_date"].transform("nunique")
    df["match_date"] = df["tourney_date"]
    df["date_is_exact"] = (n_dates > 1).astype(int)
    df["tourney_date"] = starts

    # The id must survive a match moving from the ongoing file to the yearly
    # one. `match_num` does not: upstream renumbers it when it consolidates, so
    # an id containing it changed under every row that migrated and orphaned
    # whatever pointed at the old one (features, odds, te_matches). Event,
    # round and the two players identify a match; the handful of true repeats
    # (a pair meeting twice in one round) are source duplicates, not rematches.
    df["match_id"] = (
        df["tourney_key"] + "-" + df["round"].fillna("NA")
        + "-" + df["winner_id"] + "-" + df["loser_id"]
    )

    # ---- dedupe -----------------------------------------------------------
    before = len(df)
    # Prefer rows that carry stats when the same match appears twice.
    df = df.sort_values(["has_stats", "totals_usable", "source_priority"],
                        ascending=[False, False, True], kind="stable")
    df = df.drop_duplicates(subset=["match_id"], keep="first")
    log.info("deduped %d -> %d rows", before, len(df))

    # ---- global chronological ordering ------------------------------------
    # Round robin is played BEFORE the knockout rounds of the same event, but
    # sits last in ROUND_ORDER (an index the model has learned from). Ordering
    # by that index fed ATP Finals / United Cup / Davis Cup group matches their
    # features after the SF and F results were already in -- 15,849 rows. The
    # sort key puts RR first; `round_idx` itself is left alone.
    df["_round_sort"] = df["round_idx"].where(df["round"] != "RR", -1.0)
    df = df.sort_values(
        ["tourney_date", "tourney_key", "_round_sort", "match_num"],
        na_position="last",
    ).reset_index(drop=True).drop(columns=["_round_sort"])
    df["seq"] = np.arange(len(df), dtype="int64")
    # season follows the event's start date, now that tourney_date is the start
    df["season"] = df["tourney_date"] // 10000
    return df.drop(columns=["_edition"])


def check_player_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Verify ATP player IDs mean the same player in main-tour and challenger files.

    CLAUDE.md says IDs are official ATP IDs, which should make the join safe,
    but the task asks us to confirm rather than assume. We compare, for every ID
    seen in both file families, whether the normalised names agree.
    """
    long = pd.concat([
        df[["winner_id", "winner_name", "is_challenger_file"]].rename(
            columns={"winner_id": "pid", "winner_name": "name"}),
        df[["loser_id", "loser_name", "is_challenger_file"]].rename(
            columns={"loser_id": "pid", "loser_name": "name"}),
    ])
    long["name_norm"] = long["name"].map(norm_name)
    agg = long.groupby(["pid", "is_challenger_file"])["name_norm"].agg(
        lambda s: s.value_counts().idxmax()
    ).unstack()
    if not {0, 1} <= set(agg.columns):        # only one file family loaded
        return agg.iloc[0:0]
    both = agg.dropna()
    conflicts = both[both[0] != both[1]]
    log.info(
        "player-id check: %d ids total, %d appear in both main+challenger, "
        "%d name conflicts", long["pid"].nunique(), len(both), len(conflicts)
    )
    return conflicts


def build_players(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for side in ("winner", "loser"):
        cols = {f"{side}_id": "player_id", f"{side}_name": "name",
                f"{side}_hand": "hand", f"{side}_ht": "height_cm",
                f"{side}_ioc": "ioc"}
        p = df[[*cols, "tourney_date", "tour"]].rename(columns=cols)
        parts.append(p)
    long = pd.concat(parts, ignore_index=True)
    long = long[long["player_id"].notna()]

    def _mode(s):
        s = s.dropna()
        return s.value_counts().idxmax() if len(s) else None

    out = long.groupby("player_id").agg(
        name=("name", _mode),
        hand=("hand", _mode),
        height_cm=("height_cm", "median"),
        ioc=("ioc", _mode),
        first_seen=("tourney_date", "min"),
        last_seen=("tourney_date", "max"),
        n_matches=("player_id", "size"),
        # A player belongs to exactly one tour here -- the two never meet, so
        # the mode is the tour, not an approximation of it.
        tour=("tour", _mode),
    ).reset_index()
    out["name_norm"] = out["name"].map(norm_name)
    return out


def build_tournaments(df: pd.DataFrame) -> pd.DataFrame:
    def _mode(s):
        s = s.dropna()
        return s.value_counts().idxmax() if len(s) else None

    return df.groupby("tourney_key").agg(
        tourney_id=("tourney_id", _mode),
        name=("tourney_name", _mode),
        surface=("surface", _mode),
        level=("level", _mode),
        is_challenger=("is_challenger", "max"),
        indoor=("indoor", _mode),
        draw_size=("draw_size", "max"),
        tourney_date=("tourney_date", "min"),
        season=("season", "min"),
        tour=("tour", _mode),
    ).reset_index()


MATCH_COLS = [
    "match_id", "tourney_key", "tour", "match_num", "round", "round_idx",
    "best_of", "tourney_date", "match_date", "date_is_exact", "seq",
    "winner_id", "loser_id", "winner_seed", "loser_seed", "winner_entry",
    "loser_entry", "winner_age", "loser_age", "winner_rank", "loser_rank",
    "winner_rank_points", "loser_rank_points", "winner_hand", "loser_hand",
    "winner_ht", "loser_ht", "score", "status", "minutes", "w_sets", "l_sets",
    "w_games", "l_games", "total_games", "total_sets", "game_margin",
    "tiebreaks", "totals_usable", "has_stats", *STAT_COLS, "source_file",
]


_OLD_ID = re.compile(r"^(?P<key>.*)-(?P<num>-?\d+)-(?P<w>[^-]+)-(?P<l>[^-]+)$")
_NEW_ID = re.compile(r"^(?P<key>.*)-(?P<rnd>[A-Za-z][A-Za-z0-9]*|NA)-(?P<w>[^-]+)-(?P<l>[^-]+)$")
# Tables that reference a match_id and must follow it across the id change.
_REFERRERS = (("odds", "match_id"), ("te_matches", "match_id"))


def _insert(con, table: str, df: pd.DataFrame, cols: list[str]) -> None:
    """Insert on the caller's connection without committing.

    `DataFrame.to_sql` commits on its own, which silently ended the surrounding
    transaction at the first insert: the DELETEs were already permanent, and a
    failure on the next insert left the table empty.
    """
    cols = [c for c in cols if c in df.columns]
    data = df[cols].astype(object).where(df[cols].notna(), None)
    con.executemany(
        f"INSERT INTO {table}({','.join(cols)}) VALUES({','.join('?' * len(cols))})",
        data.itertuples(index=False, name=None),
    )


def _remap_legacy_ids(con, matches: pd.DataFrame, alias: dict | None = None) -> int:
    """Re-point odds/te_matches rows still carrying the old `match_num` ids.

    The old id was `<tourney_key>-<match_num>-<winner>-<loser>`; the new one
    drops match_num, so (tourney_key, winner, loser) recovers the match. Only
    unambiguous triples are remapped.
    """
    alias = alias or {}
    new_ids = set(matches["match_id"])
    trip = (matches["tourney_key"] + "\x1f" + matches["winner_id"]
            + "\x1f" + matches["loser_id"])
    unique = trip[~trip.duplicated(keep=False)]
    lookup = dict(zip(unique, matches.loc[unique.index, "match_id"]))
    pairs = []
    for table, col in _REFERRERS:
        for (old,) in con.execute(f"SELECT DISTINCT {col} FROM {table} WHERE {col} IS NOT NULL"):
            if old in new_ids:
                continue
            m = _OLD_ID.match(old)
            if m:
                w, l = alias.get(m["w"], m["w"]), alias.get(m["l"], m["l"])
                new = lookup.get(f"{m['key']}\x1f{w}\x1f{l}")
                if new:
                    pairs.append((old, new))
                continue
            # current format, but naming a player id that has since been merged
            m = _NEW_ID.match(old)
            if m and (m["w"] in alias or m["l"] in alias):
                new = (f"{m['key']}-{m['rnd']}-{alias.get(m['w'], m['w'])}-"
                       f"{alias.get(m['l'], m['l'])}")
                if new in new_ids:
                    pairs.append((old, new))
    if not pairs:
        return 0
    con.execute("CREATE TEMP TABLE IF NOT EXISTS idmap(old TEXT PRIMARY KEY, new TEXT)")
    con.execute("DELETE FROM idmap")
    con.executemany("INSERT OR REPLACE INTO idmap VALUES(?,?)", pairs)
    for table, col in _REFERRERS:
        # OR REPLACE: two old ids can collapse onto one new id (the source
        # duplicates), and the primary key on odds would otherwise reject it.
        con.execute(
            f"UPDATE OR REPLACE {table} SET {col}=(SELECT new FROM idmap WHERE old={col}) "
            f"WHERE {col} IN (SELECT old FROM idmap)")
    con.execute("DROP TABLE idmap")
    return len(pairs)


def load_all(db_path=None) -> dict:
    con = init_db(db_path) if db_path else init_db()
    raw = read_raw()
    log.info("read %d raw rows", len(raw))
    aliases, candidates = find_aliases(raw)
    df = clean(raw, aliases)
    conflicts = check_player_ids(raw)
    if len(candidates):
        # Not applied: two ids sharing a name are sometimes two people. Review,
        # then add the genuine ones to player_aliases.csv.
        candidates.to_csv(ARTIFACTS / "alias_candidates.csv", index=False)
        log.warning("%d possible duplicate ids left unmerged (see "
                    "artifacts/alias_candidates.csv)", len(candidates))

    players = build_players(df)
    tourneys = build_tournaments(df)
    matches = df[[c for c in MATCH_COLS if c in df.columns]]

    rk = pd.concat([
        df[["winner_id", "tourney_date", "winner_rank", "winner_rank_points"]]
          .rename(columns={"winner_id": "player_id", "tourney_date": "as_of",
                           "winner_rank": "rank", "winner_rank_points": "rank_points"}),
        df[["loser_id", "tourney_date", "loser_rank", "loser_rank_points"]]
          .rename(columns={"loser_id": "player_id", "tourney_date": "as_of",
                           "loser_rank": "rank", "loser_rank_points": "rank_points"}),
    ])
    rk = rk[rk["rank"].notna()].drop_duplicates(["player_id", "as_of"])

    # Claim first so the odds backfill stands down at its next batch, then take
    # SQLite's write lock. Claiming without BEGIN IMMEDIATE would still race an
    # uncooperative writer; BEGIN IMMEDIATE without claiming just loses the
    # race politely every night.
    #
    # Everything below -- all four deletes, all four inserts, the id remap and
    # the state keys -- is ONE transaction. On 2026-08-18 the nightly run
    # collided with the odds backfill, the inserts raised "database is locked"
    # after the deletes had landed, and the whole matches table was destroyed;
    # a later "fix" still let pandas' to_sql commit mid-way. Here a failure at
    # any point rolls back to the previous, complete database.
    with exclusive_write("nightly match reload"):
        con.isolation_level = None            # explicit BEGIN/COMMIT only
        con.execute("BEGIN IMMEDIATE")
        try:
            n_old = con.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
            if n_old and len(matches) < 0.9 * n_old:
                raise RuntimeError(
                    f"refusing to replace {n_old} matches with {len(matches)}: "
                    "a source file is probably missing or truncated")
            for t in ("matches", "tournaments", "players", "rankings"):
                con.execute(f"DELETE FROM {t}")
            _insert(con, "players", players, list(players.columns))
            _insert(con, "tournaments", tourneys, list(tourneys.columns))
            _insert(con, "matches", matches, list(matches.columns))
            _insert(con, "rankings", rk, ["player_id", "as_of", "rank", "rank_points"])
            con.execute("DELETE FROM player_alias")
            if len(aliases):
                con.executemany(
                    "INSERT OR REPLACE INTO player_alias(alias_id,canonical_id,tour,reason) "
                    "VALUES(?,?,?,?)", aliases[["alias_id", "canonical_id", "tour", "reason"]]
                    .itertuples(index=False, name=None))
            remapped = _remap_legacy_ids(con, matches,
                                         dict(zip(aliases["alias_id"], aliases["canonical_id"])))
            set_state(con, "max_seq", int(df["seq"].max()))
            set_state(con, "max_tourney_date", int(df["tourney_date"].max()))
            n = con.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
            if n != len(matches):
                raise RuntimeError(f"matches insert incomplete: {n} of {len(matches)}")
            con.execute("COMMIT")
        except BaseException:
            con.execute("ROLLBACK")
            raise
        finally:
            con.isolation_level = ""
    log.info("remapped %d legacy match ids", remapped)

    stats = {
        "matches": len(matches),
        "players": len(players),
        "tournaments": len(tourneys),
        "rankings": len(rk),
        "id_conflicts": len(conflicts),
        "ids_remapped": remapped,
        "aliases_applied": len(aliases),
        "alias_candidates": len(candidates),
        "date_range": (int(df["tourney_date"].min()), int(df["tourney_date"].max())),
        "status_counts": df["status"].value_counts().to_dict(),
        "challenger_share": float(df["is_challenger"].mean()),
    }
    con.close()
    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for k, v in load_all().items():
        print(f"{k}: {v}")
