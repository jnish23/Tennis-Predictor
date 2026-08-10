"""SQLite schema and connection helpers.

Tour-agnostic on purpose: every table that could differ between ATP and WTA
carries a `tour` column even though only 'atp' is populated today, so adding
WTA later is a load-time filter change rather than a migration.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from tennis.config import DB_PATH

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS players (
    player_id     TEXT PRIMARY KEY,
    tour          TEXT NOT NULL DEFAULT 'atp',
    name          TEXT,
    name_norm     TEXT,
    hand          TEXT,
    height_cm     REAL,
    ioc           TEXT,
    first_seen    INTEGER,
    last_seen     INTEGER,
    n_matches     INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_players_norm ON players(name_norm);

CREATE TABLE IF NOT EXISTS tournaments (
    tourney_key   TEXT PRIMARY KEY,      -- surrogate: date + slugged name
    tourney_id    TEXT,
    tour          TEXT NOT NULL DEFAULT 'atp',
    name          TEXT,
    surface       TEXT,
    level         TEXT,                  -- standardised (see config.LEVEL_MAP)
    is_challenger INTEGER,
    indoor        INTEGER,               -- 1 indoor, 0 outdoor, NULL unknown
    draw_size     INTEGER,
    tourney_date  INTEGER,               -- YYYYMMDD, week the event began
    season        INTEGER
);
CREATE INDEX IF NOT EXISTS ix_tourn_date ON tournaments(tourney_date);

CREATE TABLE IF NOT EXISTS matches (
    match_id      TEXT PRIMARY KEY,
    tourney_key   TEXT NOT NULL,
    tour          TEXT NOT NULL DEFAULT 'atp',
    match_num     INTEGER,
    round         TEXT,
    round_idx     INTEGER,               -- ordering within a tournament
    best_of       INTEGER,
    tourney_date  INTEGER,
    match_date    INTEGER,               -- real date once joined to odds, else tourney_date
    date_is_exact INTEGER DEFAULT 0,
    seq           INTEGER,               -- global chronological order
    winner_id     TEXT,
    loser_id      TEXT,
    winner_seed   REAL, loser_seed REAL,
    winner_entry  TEXT,  loser_entry TEXT,
    winner_age    REAL,  loser_age  REAL,
    winner_rank   REAL,  loser_rank REAL,
    winner_rank_points REAL, loser_rank_points REAL,
    winner_hand   TEXT,  loser_hand TEXT,
    winner_ht     REAL,  loser_ht   REAL,
    score         TEXT,
    status        TEXT,                  -- completed/retired/walkover/default/unfinished
    minutes       REAL,
    w_sets INTEGER, l_sets INTEGER,
    w_games INTEGER, l_games INTEGER,
    total_games INTEGER, total_sets INTEGER, game_margin INTEGER,
    tiebreaks INTEGER,
    totals_usable INTEGER,               -- safe as a totals/spread label
    has_stats     INTEGER,
    w_ace REAL, w_df REAL, w_svpt REAL, w_1stIn REAL, w_1stWon REAL,
    w_2ndWon REAL, w_SvGms REAL, w_bpSaved REAL, w_bpFaced REAL,
    l_ace REAL, l_df REAL, l_svpt REAL, l_1stIn REAL, l_1stWon REAL,
    l_2ndWon REAL, l_SvGms REAL, l_bpSaved REAL, l_bpFaced REAL,
    source_file   TEXT,
    FOREIGN KEY (tourney_key) REFERENCES tournaments(tourney_key)
);
CREATE INDEX IF NOT EXISTS ix_m_seq     ON matches(seq);
CREATE INDEX IF NOT EXISTS ix_m_date    ON matches(tourney_date);
CREATE INDEX IF NOT EXISTS ix_m_winner  ON matches(winner_id);
CREATE INDEX IF NOT EXISTS ix_m_loser   ON matches(loser_id);
CREATE INDEX IF NOT EXISTS ix_m_tourney ON matches(tourney_key);

-- Closing//pre-match prices from tennis-data.co.uk. Stored winner-relative:
-- p1 is the match winner, p2 the loser (matches the source layout).
CREATE TABLE IF NOT EXISTS odds (
    match_id   TEXT,
    book       TEXT,           -- B365, PS, Max, Avg, ...
    win_price  REAL,           -- decimal odds on the actual winner
    lose_price REAL,           -- decimal odds on the actual loser
    PRIMARY KEY (match_id, book)
);

-- Live odds captures, one row per (fixture, capture). Deliberately separate
-- from `odds`: these are *unresolved* observations scraped before a match is
-- played, keyed on names rather than match_id, and the same fixture is
-- captured repeatedly so the line's movement is preserved. `odds` stays the
-- resolved, match-keyed table; `resolve_snapshots` promotes the last capture
-- before start into it once a result exists.
--
-- Keeping every capture is the point. A single scrape is an arbitrary moment
-- in a moving market; the last one before play is a closing-line proxy, which
-- is the only thing CLV can honestly be measured against.
CREATE TABLE IF NOT EXISTS odds_snapshots (
    captured_at TEXT NOT NULL,   -- ISO-8601 UTC, when we fetched it
    play_date   INTEGER NOT NULL,-- YYYYMMDD the fixture is listed under
    start_time  TEXT,            -- HH:MM local to the site, as published
    tour        TEXT,            -- atp / wta
    tournament  TEXT,
    p1_name     TEXT NOT NULL,
    p2_name     TEXT NOT NULL,
    p1_odds     REAL,
    p2_odds     REAL,
    source      TEXT NOT NULL,   -- 'tennisexplorer'
    PRIMARY KEY (source, play_date, p1_name, p2_name, captured_at)
);
CREATE INDEX IF NOT EXISTS ix_snap_date ON odds_snapshots(play_date);

-- tennisexplorer's own match records, and the bridge to ours. Kept separate
-- from `matches` because their id space, name format and tour coverage are all
-- theirs; `match_id` is filled in once a row is resolved against our data, and
-- `p1_is_winner` pins orientation ONCE rather than re-deriving it per quote.
CREATE TABLE IF NOT EXISTS te_matches (
    te_id        INTEGER PRIMARY KEY,
    play_date    INTEGER,
    tour         TEXT,
    tournament   TEXT,
    p1_name      TEXT,
    p2_name      TEXT,
    score        TEXT,
    p1_odds      REAL,          -- moneyline from the day page, if shown
    p2_odds      REAL,
    match_id     TEXT,          -- our match_id, NULL until resolved
    p1_is_winner INTEGER,       -- 1 if their p1 is our winner
    detail_done  INTEGER DEFAULT 0,
    fetched_at   TEXT
);
CREATE INDEX IF NOT EXISTS ix_te_date   ON te_matches(play_date);
CREATE INDEX IF NOT EXISTS ix_te_detail ON te_matches(detail_done);
CREATE INDEX IF NOT EXISTS ix_te_match  ON te_matches(match_id);

-- One row per (match, book, market, line, side). This shape is what lets a
-- moneyline, an over/under on games, an Asian handicap and a correct-score
-- quote all live in one table: `odds` cannot hold them because it has no
-- concept of a line. Opening and closing prices sit together, which is what
-- makes real closing-line value computable rather than proxied.
CREATE TABLE IF NOT EXISTS odds_quotes (
    te_id       INTEGER NOT NULL,
    book        TEXT NOT NULL,
    market      TEXT NOT NULL,   -- h2h | totals | handicap | correct_score
    line        REAL,            -- 2.5, -1.5; NULL for h2h
    line_unit   TEXT,            -- sets | games | NULL
    side        TEXT NOT NULL,   -- p1|p2 | over|under | '2:0'
    price_close REAL,
    closed_at   TEXT,
    price_open  REAL,
    opened_at   TEXT,
    -- line_unit is part of the key, not decoration. A -1.5 handicap exists in
    -- both sets and games and they are different markets at different prices
    -- (measured on 12 pages: 2.75 against 1.95). Leaving it out silently
    -- destroyed 3.4% of quotes, all of them games rows -- the ones this whole
    -- exercise is for.
    PRIMARY KEY (te_id, book, market, line, line_unit, side)
);
CREATE INDEX IF NOT EXISTS ix_q_market ON odds_quotes(market, line_unit);
CREATE INDEX IF NOT EXISTS ix_q_te     ON odds_quotes(te_id);

-- Scrape checkpoint. The whole backfill is ~200k requests over days, so it has
-- to survive being killed: every unit of work is recorded the moment it lands,
-- and a restart skips whatever is already here rather than starting over.
CREATE TABLE IF NOT EXISTS te_scrape_log (
    kind       TEXT NOT NULL,    -- 'day' | 'detail'
    key        TEXT NOT NULL,    -- YYYYMMDD | te_id
    status     TEXT NOT NULL,    -- ok | empty | error
    n_rows     INTEGER,
    note       TEXT,
    done_at    TEXT,
    PRIMARY KEY (kind, key)
);

-- Rankings snapshots derived from match rows (rank at time of match).
CREATE TABLE IF NOT EXISTS rankings (
    player_id  TEXT,
    as_of      INTEGER,
    rank       REAL,
    rank_points REAL,
    PRIMARY KEY (player_id, as_of)
);

-- Wide pre-match feature table, one row per match, strictly pre-match values.
CREATE TABLE IF NOT EXISTS features (
    match_id TEXT PRIMARY KEY,
    seq      INTEGER,
    tourney_date INTEGER,
    payload  TEXT              -- JSON blob; canonical copy lives in parquet
);

-- Incremental state so daily updates need not recompute history.
CREATE TABLE IF NOT EXISTS elo_state (
    player_id TEXT,
    scope     TEXT,            -- 'overall' | 'Hard' | 'Clay' | 'Grass' | 'Carpet'
    rating    REAL,
    n_matches INTEGER,
    last_seq  INTEGER,
    PRIMARY KEY (player_id, scope)
);

CREATE TABLE IF NOT EXISTS pipeline_state (
    key   TEXT PRIMARY KEY,
    value TEXT
);

-- Upcoming / entered draws for simulation.
CREATE TABLE IF NOT EXISTS draws (
    draw_id     TEXT,
    tourney_name TEXT,
    surface     TEXT,
    level       TEXT,
    indoor      INTEGER,
    best_of     INTEGER,
    draw_size   INTEGER,
    tourney_date INTEGER,
    slot        INTEGER,        -- 0..draw_size-1, first-round bracket position
    player_id   TEXT,
    player_name TEXT,
    PRIMARY KEY (draw_id, slot)
);

-- Maps a feed tournament id to our own tourney_key. The two sources share no
-- tournament identifier: ours embeds the real ATP id (2026-421 = Canada
-- Masters) while the feed uses internal ids (21346). Resolved once by player
-- overlap, then reused as a straight id lookup.
CREATE TABLE IF NOT EXISTS tourney_xref (
    feed_id      TEXT PRIMARY KEY,
    feed_name    TEXT,
    tourney_key  TEXT,
    season       INTEGER,
    n_overlap    INTEGER,     -- matches supporting the link, for auditing
    resolved_at  TEXT
);

CREATE TABLE IF NOT EXISTS sim_results (
    draw_id   TEXT,
    run_at    TEXT,
    player_id TEXT,
    player_name TEXT,
    round     TEXT,
    prob      REAL,
    PRIMARY KEY (draw_id, run_at, player_id, round)
);

-- Predictions for upcoming (not yet played) matches.
CREATE TABLE IF NOT EXISTS predictions (
    pred_id     TEXT PRIMARY KEY,
    made_at     TEXT,
    match_date  INTEGER,
    tourney_name TEXT,
    surface     TEXT,
    level       TEXT,
    p1_id TEXT, p1_name TEXT,
    p2_id TEXT, p2_name TEXT,
    p1_win_prob REAL,
    pred_total_games REAL,
    pred_spread REAL,
    best_of INTEGER
);

-- Backtest output, one row per scored match per model run.
CREATE TABLE IF NOT EXISTS backtest (
    run_tag   TEXT,
    match_id  TEXT,
    seq       INTEGER,
    tourney_date INTEGER,
    surface   TEXT,
    level     TEXT,
    is_challenger INTEGER,
    tour      TEXT,
    p1_id TEXT, p2_id TEXT,
    y_win     INTEGER,
    p_win     REAL,
    y_total   REAL, pred_total REAL,
    y_spread  REAL, pred_spread REAL,
    close_p1  REAL, close_p2  REAL,
    PRIMARY KEY (run_tag, match_id)
);
CREATE INDEX IF NOT EXISTS ix_bt_date ON backtest(tourney_date);
"""


def connect(path: Path | str = DB_PATH) -> sqlite3.Connection:
    con = sqlite3.connect(str(path), timeout=60)
    con.execute("PRAGMA foreign_keys=ON")
    return con


def init_db(path: Path | str = DB_PATH) -> sqlite3.Connection:
    con = connect(path)
    con.executescript(SCHEMA)
    con.commit()
    return con


def get_state(con: sqlite3.Connection, key: str, default=None):
    row = con.execute("SELECT value FROM pipeline_state WHERE key=?", (key,)).fetchone()
    return row[0] if row else default


def set_state(con: sqlite3.Connection, key: str, value) -> None:
    con.execute(
        "INSERT INTO pipeline_state(key,value) VALUES(?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, str(value)),
    )
