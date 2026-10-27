"""Shared test configuration.

Most of this suite is pure logic -- parsers, simulator invariants, the odds-key
and bracket-geometry regressions -- and needs nothing but the source tree. A
minority genuinely exercise the 256 MB SQLite database or the trained model
artifacts, neither of which is in version control and neither of which CI
should try to reconstruct.

Those are marked `needs_db` and skip themselves when the data is absent, so the
same `pytest` command is correct both on a laptop with the full pipeline and on
a clean CI checkout. Skipping is the honest behaviour here: a green run on CI
means "the logic tests pass", and the skip count says plainly how many were not
exercised rather than pretending they were.
"""
from __future__ import annotations

import pytest

from tennis.config import ARTIFACTS, DB_PATH


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "needs_db: requires the local SQLite database or artifacts")
    config.addinivalue_line(
        "markers", "features: additionally requires artifacts/features_atp.parquet")


def pytest_collection_modifyitems(config, items):
    have_db = DB_PATH.exists()
    have_feats = (ARTIFACTS / "features_atp.parquet").exists()
    for item in items:
        if "needs_db" not in item.keywords:
            continue
        if not have_db:
            item.add_marker(pytest.mark.skip(reason="no local database"))
        elif "features" in item.keywords and not have_feats:
            item.add_marker(pytest.mark.skip(reason="no features_atp.parquet"))


@pytest.fixture(autouse=True)
def _no_reviewed_aliases(monkeypatch, tmp_path):
    """The reviewed-alias file is data, not logic. Tests of the duplicate-id
    rules must see only the ids they construct, however many real merges the
    file has accumulated."""
    import tennis.ingest.load as load
    monkeypatch.setattr(load, "ALIAS_CSV", tmp_path / "no_aliases.csv")
