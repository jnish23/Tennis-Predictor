"""Background refresh of the hosted data bundle, and the odds counts that
replaced shipping the odds table."""
from __future__ import annotations

import json
import sqlite3
import threading
import time

import pytest

from tennis.ops import bundle


@pytest.fixture(autouse=True)
def _reset_bg():
    bundle._bg.update({"thread": None, "error": None, "failed_at": 0.0, "seen": None})
    yield
    bundle._bg.update({"thread": None, "error": None, "failed_at": 0.0, "seen": None})


def test_background_fetch_does_not_block_and_runs_once(monkeypatch):
    gate, calls = threading.Event(), []
    monkeypatch.setattr(bundle, "fetch_serve", lambda *a, **k: (calls.append(1), gate.wait(5)))
    t0 = time.time()
    assert bundle.start_background_fetch("http://x") == "started"
    assert time.time() - t0 < 1.0
    assert bundle.start_background_fetch("http://x") == "running"
    gate.set()
    bundle._bg["thread"].join(5)
    assert calls == [1]


def test_failed_refresh_backs_off_and_reports(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("network down")
    monkeypatch.setattr(bundle, "fetch_serve", boom)
    assert bundle.start_background_fetch("http://x") == "started"
    bundle._bg["thread"].join(5)
    assert bundle.background_error() == "network down"
    assert bundle.start_background_fetch("http://x") == "backoff"
    bundle._bg["failed_at"] -= bundle.BG_RETRY_S + 1
    assert bundle.start_background_fetch("http://x") == "started"
    bundle._bg["thread"].join(5)


def test_bundle_changed_fires_once_per_new_build(tmp_path):
    meta = tmp_path / bundle.SERVE_META
    meta.write_text(json.dumps({"built_at": "2026-01-01T00:00:00+00:00"}))
    assert bundle.bundle_changed(tmp_path) is False       # baseline only
    assert bundle.bundle_changed(tmp_path) is False
    meta.write_text(json.dumps({"built_at": "2026-01-01T06:00:00+00:00"}))
    assert bundle.bundle_changed(tmp_path) is True
    assert bundle.bundle_changed(tmp_path) is False


def test_odds_table_is_not_shipped_but_its_counts_are():
    assert "odds" not in bundle.SERVE_TABLES
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE odds(match_id TEXT, book TEXT, win_price REAL, lose_price REAL)")
    con.executemany("INSERT INTO odds VALUES(?,?,?,?)",
                    [("a", "B365", 1.5, 2.5), ("a", "PS", 1.5, 2.6), ("b", "PS", 2.0, 1.9)])
    assert bundle.odds_summary(con) == {"rows": 3, "matches": 2}
    assert bundle.odds_summary(sqlite3.connect(":memory:")) == {"rows": 0, "matches": 0}
