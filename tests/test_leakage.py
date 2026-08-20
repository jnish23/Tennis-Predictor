"""Leakage guards.

CLAUDE.md calls data leakage a standing risk rather than a one-time check, so
these run as part of the normal test suite and should be re-run whenever the
feature engine changes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tennis.db.schema import connect
from tennis.features.build import FeatureEngine, build_features
from tennis.features.pipeline import load_matches


@pytest.fixture(scope="module")
def matches():
    con = connect()
    df = load_matches(con)
    con.close()
    return df


@pytest.mark.needs_db
def test_prefix_invariance(matches):
    """A match's features must not change when later matches are added.

    This is the strongest single leakage check: build features over the first N
    matches, then over the first 2N, and assert the first N rows are identical.
    If any feature peeked forward, the two runs would disagree.
    """
    n = 20_000
    head = matches.iloc[:n]
    both = matches.iloc[: 2 * n]

    f_head, _ = build_features(head, FeatureEngine())
    f_both, _ = build_features(both, FeatureEngine())
    f_both = f_both.iloc[:n]

    assert list(f_head["match_id"]) == list(f_both["match_id"])
    num = [c for c in f_head.columns if pd.api.types.is_numeric_dtype(f_head[c])]
    for c in num:
        a, b = f_head[c].to_numpy(float), f_both[c].to_numpy(float)
        assert np.allclose(a, b, equal_nan=True), f"feature {c} changed with future data"


@pytest.mark.needs_db
def test_first_appearance_has_no_history(matches):
    """A player's very first match must carry no form/serve history."""
    sub = matches.iloc[:50_000]
    feats, _ = build_features(sub, FeatureEngine())
    seen: set[str] = set()
    checked = 0
    for row in sub.sort_values("seq").itertuples():
        for pid in (row.winner_id, row.loser_id):
            if pid in seen:
                continue
            fr = feats[feats["match_id"] == row.match_id]
            if fr.empty:
                continue
            side = "p1" if fr.iloc[0]["p1_id"] == pid else "p2"
            for col in ("winrate_10", "spw", "career_winrate", "avg_total_games"):
                assert pd.isna(fr.iloc[0][f"{side}_{col}"]), (
                    f"debut match for {pid} has {col}"
                )
            assert fr.iloc[0][f"{side}_career_n"] == 0
            checked += 1
        seen.update([row.winner_id, row.loser_id])
        if checked > 300:
            break
    assert checked > 100


@pytest.mark.needs_db
@pytest.mark.features
def test_no_same_match_stats_in_features():
    """No feature column may be a raw same-match stat column."""
    from tennis.features.pipeline import FEATURES_PATH

    feats = pd.read_parquet(FEATURES_PATH)
    banned = {
        "score", "minutes", "w_ace", "l_ace", "w_svpt", "l_svpt", "w_1stIn",
        "l_1stIn", "w_1stWon", "l_1stWon", "w_2ndWon", "l_2ndWon", "w_SvGms",
        "l_SvGms", "w_bpSaved", "l_bpSaved", "w_bpFaced", "l_bpFaced",
        "winner_id", "loser_id", "winner_name", "loser_name", "status",
        "w_games", "l_games", "w_sets", "l_sets", "tiebreaks",
    }
    from tennis.models.common import FEATURE_COLS

    overlap = banned.intersection(FEATURE_COLS)
    assert not overlap, f"same-match stats present in model features: {overlap}"
    # total_games / game_margin survive in the frame as labels, but must never
    # be listed as inputs.
    assert "total_games" not in FEATURE_COLS
    assert "game_margin" not in FEATURE_COLS
    assert "y_win" not in FEATURE_COLS
