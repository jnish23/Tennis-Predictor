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
    RAW_TML,
    ROUND_ORDER,
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


def read_raw(include_ongoing: bool = True) -> pd.DataFrame:
    """Read every cached ATP CSV into one frame, tagged with its source."""
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
        df = pd.read_csv(path, low_memory=False)
        if df.empty:
            continue
        df["source_file"] = name
        df["is_challenger_file"] = int("challenger" in name)
        frames.append(df)
    if not frames:
        raise RuntimeError(f"no CSVs found in {RAW_TML}")
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

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
        df[c] = df[c].astype("string").str.strip()
    df = df[df["winner_id"].notna() & df["loser_id"].notna()].copy()
    df = df[(df["winner_id"] != "") & (df["loser_id"] != "")]
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

    # A match "has stats" only when the serve counters are actually populated.
    df["has_stats"] = (
        df[["w_svpt", "l_svpt", "w_SvGms", "l_SvGms"]].notna().all(axis=1)
        & (df["w_svpt"].fillna(0) > 0)
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

    df["match_id"] = (
        df["tourney_key"] + "-" + df["match_num"].fillna(-1).astype("int64").astype(str)
        + "-" + df["winner_id"] + "-" + df["loser_id"]
    )

    # ---- dedupe -----------------------------------------------------------
    before = len(df)
    # Prefer rows that carry stats when the same match appears twice.
    df = df.sort_values(["has_stats", "totals_usable"], ascending=False)
    df = df.drop_duplicates(subset=["match_id"], keep="first")
    df = df.drop_duplicates(
        subset=["tourney_date", "match_num", "winner_id", "loser_id"], keep="first"
    )
    log.info("deduped %d -> %d rows", before, len(df))

    # ---- global chronological ordering ------------------------------------
    df = df.sort_values(
        ["tourney_date", "tourney_key", "round_idx", "match_num"],
        na_position="last",
    ).reset_index(drop=True)
    df["seq"] = np.arange(len(df), dtype="int64")
    # season follows the event's start date, now that tourney_date is the start
    df["season"] = df["tourney_date"] // 10000
    df["tour"] = "atp"
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
        p = df[[*cols, "tourney_date"]].rename(columns=cols)
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
    ).reset_index()
    out["name_norm"] = out["name"].map(norm_name)
    out["tour"] = "atp"
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
    ).reset_index().assign(tour="atp")


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


def load_all(db_path=None) -> dict:
    con = init_db(db_path) if db_path else init_db()
    raw = read_raw()
    log.info("read %d raw rows", len(raw))
    df = clean(raw)
    conflicts = check_player_ids(raw)

    players = build_players(df)
    tourneys = build_tournaments(df)
    matches = df[[c for c in MATCH_COLS if c in df.columns]]

    # BEGIN IMMEDIATE takes the write lock up front. Without it this block is
    # a live grenade: it deletes all three tables and *then* inserts, so any
    # failure in between leaves them empty. That is not hypothetical -- on
    # 2026-08-18 the nightly run collided with the odds backfill, the inserts
    # raised "database is locked" after the deletes had landed, and the entire
    # 200,918-row matches table was destroyed. Taking the lock first means a
    # busy database fails here, before anything is deleted, instead of halfway
    # through.
    # Claim first so the odds backfill stands down at its next batch, then
    # take SQLite's write lock. Claiming without BEGIN IMMEDIATE would still
    # race an uncooperative writer; BEGIN IMMEDIATE without claiming just loses
    # the race politely every night.
    with exclusive_write("nightly match reload"):
        con.execute("BEGIN IMMEDIATE")
        try:
            con.execute("DELETE FROM matches")
            con.execute("DELETE FROM tournaments")
            con.execute("DELETE FROM players")
            players.to_sql("players", con, if_exists="append", index=False)
            tourneys.to_sql("tournaments", con, if_exists="append", index=False)
            matches.to_sql("matches", con, if_exists="append", index=False)
            con.commit()
        except Exception:
            con.rollback()
            raise

    # pandas commits inside to_sql, so the transaction above is a guard against
    # starting on a locked database rather than a guarantee of atomicity across
    # all three inserts. Verify the outcome directly: an empty matches table is
    # always a failure, never a legitimate state.
    n = con.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    if n == 0:
        raise RuntimeError(
            "matches table is empty after load -- refusing to leave the "
            "database in this state; re-run with no other writer active")

    # rankings snapshots (rank as recorded at match time)
    rk = pd.concat([
        df[["winner_id", "tourney_date", "winner_rank", "winner_rank_points"]]
          .rename(columns={"winner_id": "player_id", "tourney_date": "as_of",
                           "winner_rank": "rank", "winner_rank_points": "rank_points"}),
        df[["loser_id", "tourney_date", "loser_rank", "loser_rank_points"]]
          .rename(columns={"loser_id": "player_id", "tourney_date": "as_of",
                           "loser_rank": "rank", "loser_rank_points": "rank_points"}),
    ])
    rk = rk[rk["rank"].notna()].drop_duplicates(["player_id", "as_of"])
    con.execute("DELETE FROM rankings")
    rk.to_sql("rankings", con, if_exists="append", index=False)

    set_state(con, "max_seq", int(df["seq"].max()))
    set_state(con, "max_tourney_date", int(df["tourney_date"].max()))
    con.commit()

    stats = {
        "matches": len(matches),
        "players": len(players),
        "tournaments": len(tourneys),
        "rankings": len(rk),
        "id_conflicts": len(conflicts),
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
