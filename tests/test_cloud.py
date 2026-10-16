"""Tests for the cloud bundles and entrypoints. None needs the real database."""
from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

from tennis.db.schema import SCHEMA, init_db
from tennis.ops import bundle

ROOT = Path(__file__).resolve().parent.parent


def _fake_source(path: Path) -> None:
    """A source database holding rows in serving and research tables alike."""
    con = init_db(path)
    con.execute("INSERT INTO tournaments(tourney_key,tour,name,tourney_date,season) "
                "VALUES('k','atp','Test',20260105,2026)")
    con.execute("INSERT INTO matches(match_id,tourney_key,tour,tourney_date,winner_id,loser_id,seq) "
                "VALUES('m1','k','atp',20260105,'A','B',1)")
    con.execute("INSERT INTO players(player_id,tour,name,name_norm) VALUES('A','atp','A A','a a')")
    con.execute("INSERT INTO odds VALUES('m1','B365',1.5,2.5)")
    con.execute("INSERT INTO odds VALUES('m1','CBO',1.6,2.4)")
    # research data the site must not carry
    con.executemany("INSERT INTO odds_quotes(te_id,book,market,line,line_unit,side,price_close) "
                    "VALUES(?,?,?,?,?,?,?)", [(i, "PS", "h2h", None, None, "p1", 1.9) for i in range(500)])
    con.execute("INSERT INTO te_matches(te_id,play_date) VALUES(1,20260105)")
    con.commit()
    con.close()


def test_slim_db_keeps_serving_tables_and_drops_research_rows(tmp_path):
    src, dst = tmp_path / "full.db", tmp_path / "slim.db"
    _fake_source(src)
    counts = bundle.build_slim_db(src, dst)
    con = sqlite3.connect(dst)
    assert con.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 1
    # odds is only ever counted by the status page, so its rows stay behind
    # (25 MB) and the counts travel in serve_meta.json instead
    assert con.execute("SELECT COUNT(*) FROM odds").fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM odds_quotes").fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM te_matches").fetchone()[0] == 0
    # the schema is whole, so a stray query against a research table still runs
    assert con.execute("SELECT COUNT(*) FROM te_scrape_log").fetchone()[0] == 0
    assert counts["matches"] == 1
    # and the indexes came with it
    idx = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='index'")}
    assert "ix_m_seq" in idx
    con.close()


def test_every_table_the_dashboard_reads_is_in_the_serve_bundle():
    tables = set(re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", SCHEMA))
    files = ["dashboard/app.py", "tennis/models/predict.py", "tennis/sim/draws.py",
             "tennis/sim/bracket.py", "tennis/ingest/draws_api.py",
             "tennis/models/common.py", "tennis/features/pipeline.py"]
    used: dict[str, set] = {}
    for f in files:
        txt = (ROOT / f).read_text()
        for t in re.findall(r"(?:FROM|JOIN|INTO)\s+([a-z_]+)", txt, flags=re.I):
            if t.lower() in tables:
                used.setdefault(t.lower(), set()).add(f)
    # pipeline.py writes elo_state and reads matches; both are served. A table
    # outside the bundle that a page reads would silently show nothing.
    missing = {t: fs for t, fs in used.items() if t not in bundle.SERVE_TABLES}
    assert not missing, f"read by the app but not shipped: {missing}"


def _serve_fixture(tmp_path):
    data, art = tmp_path / "data", tmp_path / "artifacts"
    (data / "raw" / "draws").mkdir(parents=True)
    art.mkdir()
    src = data / "tennis.db"
    _fake_source(src)
    for f in bundle.SERVE_ARTIFACTS:
        (art / f).write_text("x")
    (data / "raw" / "draws" / "draws_atp_1_2026.json").write_text("{}")
    (data / "raw" / "draws" / "explore_today.json").write_text("{}")
    old = data / "raw" / "draws" / "range_atp_old.json"
    old.write_text("{}")
    os.utime(old, (time.time() - 40 * 86400,) * 2)
    return data, art


def test_serve_bundle_roundtrip_and_atomic_swap(tmp_path, monkeypatch):
    data, art = _serve_fixture(tmp_path)
    out = tmp_path / "serve.tar.zst"
    meta = bundle.pack_serve(out, db_path=data / "tennis.db", data=data, artifacts=art)
    assert meta["max_tourney_date"] == 20260105 and out.stat().st_size > 0

    # a fresh machine: nothing local, so it is stale, then a download fills it
    d2, a2 = tmp_path / "d2", tmp_path / "a2"
    assert bundle.is_stale(data=d2)

    class R:
        def __enter__(s): return s
        def __exit__(s, *a): pass
        def raise_for_status(s): pass
        def iter_content(s, n): yield out.read_bytes()

    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: R())
    got = bundle.fetch_serve("http://x/serve.tar.zst", data=d2, artifacts=a2)
    assert got["max_tourney_date"] == 20260105
    assert not bundle.is_stale(data=d2)
    assert (a2 / "models_pooled.pkl").exists()
    assert sqlite3.connect(d2 / "tennis.db").execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 1
    # only live-page cache files ship, and not weeks-old ones
    shipped = {p.name for p in (d2 / "raw" / "draws").iterdir()}
    assert shipped == {"draws_atp_1_2026.json"}


def test_is_stale_after_the_age_limit(tmp_path):
    data = tmp_path
    (data / "tennis.db").write_text("x")
    (data / bundle.SERVE_META).write_text(json.dumps({"built_at": "2020-01-01T00:00:00+00:00"}))
    assert bundle.is_stale(6, data=data)
    now = pd.Timestamp.now("UTC").isoformat()
    (data / bundle.SERVE_META).write_text(json.dumps({"built_at": now}))
    assert not bundle.is_stale(6, data=data)


def test_a_bundle_without_a_database_is_refused(tmp_path, monkeypatch):
    data, art = _serve_fixture(tmp_path)
    bad = tmp_path / "bad.tar.zst"
    with bundle._tar_writer(bad) as tar:
        bundle._add(tar, art / "models_pooled.pkl", "artifacts/models_pooled.pkl")

    class R:
        def __enter__(s): return s
        def __exit__(s, *a): pass
        def raise_for_status(s): pass
        def iter_content(s, n): yield bad.read_bytes()

    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: R())
    with pytest.raises(RuntimeError, match="no tennis.db"):
        bundle.fetch_serve("http://x", data=tmp_path / "d3", artifacts=tmp_path / "a3")
    assert not (tmp_path / "a3" / "models_pooled.pkl").exists()


def test_state_roundtrip_restores_raw_files_and_unregenerable_odds(tmp_path):
    data, art = _serve_fixture(tmp_path)
    raw = data / "raw" / "tennismylife"
    raw.mkdir(parents=True)
    (raw / "2026.csv").write_text("a,b\n1,2\n")
    out = tmp_path / "state.tar.zst"
    counts = bundle.pack_state(out, db_path=data / "tennis.db", data=data, artifacts=art)
    assert counts["odds_extra"] == 1                     # CBO only; B365 is rebuilt
    d2, a2 = tmp_path / "d2", tmp_path / "a2"
    bundle.unpack_state(out, data=d2, artifacts=a2)
    assert (d2 / "raw" / "tennismylife" / "2026.csv").read_text() == "a,b\n1,2\n"
    extra = pd.read_parquet(d2 / bundle.ODDS_EXTRA_FILE)
    assert list(extra["book"]) == ["CBO"]


def test_snapshots_roundtrip_is_idempotent(tmp_path):
    con = init_db(tmp_path / "a.db")
    con.execute("INSERT INTO odds_snapshots(captured_at,play_date,tour,p1_name,p2_name,"
                "p1_odds,p2_odds,source,status) VALUES('t1',20260105,'atp','A','B',1.5,2.5,'te','upcoming')")
    con.commit()
    p = tmp_path / "s.parquet"
    assert bundle.export_snapshots(con, p) == 1
    con2 = init_db(tmp_path / "b.db")
    bundle.import_snapshots(con2, p)
    bundle.import_snapshots(con2, p)
    assert con2.execute("SELECT COUNT(*) FROM odds_snapshots").fetchone()[0] == 1


def test_draw_feed_is_cache_only_on_the_hosted_app(tmp_path, monkeypatch):
    from tennis.ingest import draws_api as d

    monkeypatch.setattr(d, "CACHE", tmp_path)
    monkeypatch.setenv("DATA_BUNDLE_URL", "http://x")
    monkeypatch.delenv("RAPIDAPI_KEY", raising=False)
    assert d.cache_only()
    called = []
    monkeypatch.setattr(d.requests, "get", lambda *a, **k: called.append(1))
    with pytest.raises(d.DrawFeedError, match="live fetching is off"):
        d.tournament_draw(1, 2026)
    # a date-keyed request made after the job ran falls back to the newest file
    (tmp_path / "range_atp_2026-10-01_2026-10-04.json").write_text('{"v": "old"}')
    time.sleep(0.01)
    (tmp_path / "range_atp_2026-10-02_2026-10-05.json").write_text('{"v": "new"}')
    assert d.fixtures_between("2026-10-09", "2026-10-12")["v"] == "new"
    # a cached sheet is served however old it is
    f = tmp_path / "draws_atp_1_2026.json"
    f.write_text('{"ok": 1}')
    os.utime(f, (time.time() - 9 * 86400,) * 2)
    assert d.tournament_draw(1, 2026) == {"ok": 1}
    assert called == []
    # with a key the hosted app may fetch (and is capped by the daily budget)
    monkeypatch.setenv("RAPIDAPI_KEY", "k")
    assert not d.cache_only()


def test_market_lines_block_is_carried_forward_when_the_research_table_is_absent(tmp_path, monkeypatch):
    from tennis.models import evaluate

    monkeypatch.setattr(evaluate, "ARTIFACTS", tmp_path)
    (tmp_path / "backtest_report_atp.json").write_text(json.dumps(
        {"generated_at": "2026-10-04T00:00:00", "market_lines": {"markets": {"totals": 1}}}))
    rep: dict = {}
    evaluate._carry_market_lines(rep, "atp")
    assert rep["market_lines"]["markets"] == {"totals": 1}
    assert rep["market_lines"]["carried_forward_from"] == "2026-10-04T00:00:00"
    # carrying it again keeps the ORIGINAL date, not the intermediate report's
    (tmp_path / "backtest_report_atp.json").write_text(json.dumps(
        {"generated_at": "2026-10-11T00:00:00", "market_lines": rep["market_lines"]}))
    rep2: dict = {}
    evaluate._carry_market_lines(rep2, "atp")
    assert rep2["market_lines"]["carried_forward_from"] == "2026-10-04T00:00:00"
    # a freshly computed block is never overwritten
    fresh = {"market_lines": {"markets": {"x": 2}}}
    evaluate._carry_market_lines(fresh, "atp")
    assert fresh["market_lines"]["markets"] == {"x": 2}


def test_dated_ongoing_copies_keep_every_pull(tmp_path, monkeypatch):
    from datetime import date
    from tennis.ops import cloud

    raw, dated = tmp_path / "raw", tmp_path / "dated"
    raw.mkdir(); dated.mkdir()
    (raw / "ongoing_tourneys.csv").write_text("v1")
    monkeypatch.setattr(cloud, "RAW_TML", raw)
    monkeypatch.setattr(cloud, "RAW_ONGOING_DATED", dated)
    cloud.save_dated_ongoing(date(2026, 10, 10))
    (raw / "ongoing_tourneys.csv").write_text("v2")
    cloud.save_dated_ongoing(date(2026, 10, 11))
    assert (dated / "20261010_ongoing_tourneys.csv").read_text() == "v1"
    assert (dated / "20261011_ongoing_tourneys.csv").read_text() == "v2"


def test_odds_job_runs_against_a_scratch_directory_only(tmp_path):
    """The capture job must work from the snapshots parquet alone, and must
    never touch the laptop database (env-relocated here)."""
    data, art = tmp_path / "data", tmp_path / "art"
    code = f"""
import json
from pathlib import Path
from tennis.ops import cloud
from tennis.db.schema import connect
def fake():
    con = connect()
    con.execute("INSERT INTO odds_snapshots(captured_at,play_date,tour,p1_name,p2_name,p1_odds,p2_odds,source,status) "
                "VALUES('t2',20260106,'atp','C','D',1.8,2.0,'te','upcoming')")
    con.commit(); con.close()
    return {{"captured": 1}}
print(json.dumps(cloud.odds(out=Path({str(tmp_path / 'out')!r}), capture_fn=fake)))
"""
    env = {**os.environ, "TENNIS_DATA_DIR": str(data), "TENNIS_ARTIFACTS_DIR": str(art)}
    # seed one prior snapshot, as the restored parquet would
    data.mkdir()
    con = init_db(tmp_path / "seed.db")
    con.execute("INSERT INTO odds_snapshots(captured_at,play_date,tour,p1_name,p2_name,p1_odds,p2_odds,source,status) "
                "VALUES('t1',20260105,'atp','A','B',1.5,2.5,'te','upcoming')")
    con.commit()
    bundle.export_snapshots(con, data / bundle.SNAPSHOTS_FILE)
    con.close()
    res = subprocess.run([sys.executable, "-I", "-c", "import sys; sys.path.insert(0, %r)\n" % str(ROOT) + code],
                         env=env, capture_output=True, text=True, cwd=tmp_path)
    assert res.returncode == 0, res.stderr[-800:]
    out = json.loads(res.stdout.strip().splitlines()[-1])
    assert out["restored"] == 1 and out["total"] == 2
    assert len(pd.read_parquet(tmp_path / "out" / bundle.SNAPSHOTS_FILE)) == 2
    assert (data / "tennis.db").exists()
