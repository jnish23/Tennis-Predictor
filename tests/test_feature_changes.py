"""Tests for the feature-engine changes: real tiebreak counts, shrinkage,
tournament speed, opponent-adjusted and surface serve rates, the season-opener
snapshot, and the label merge that used to lose `best_of`.

All synthetic: nothing here needs the database. The whole-history leakage
check (every numeric column, including the new ones, is unchanged when later
matches are appended) is `test_prefix_invariance` in test_leakage.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tennis.features.build import (FeatureEngine, SHRINK_K, _shrunk,
                                   build_features)
from tennis.features.elo import BASE_RATING
from tennis.features.pipeline import LABEL_COLS, _attach_labels
from tennis.ingest.parse import tiebreaks_won


def mk(i, w, l, date, *, name="Foo Open", surface="Hard", score="6-4 6-4",
       aces=(10, 5), svpt=(80, 80), won=(55, 40), best_of=3):
    """One played match as the engine's input row."""
    return {
        "match_id": f"m{i}", "seq": i, "tourney_date": date, "winner_id": w,
        "loser_id": l, "surface": surface, "level": "atp250",
        "is_challenger": 0, "status": "completed", "best_of": best_of,
        "round_idx": 1, "draw_size": 32, "indoor": 0, "minutes": 90.0,
        "has_stats": 1, "totals_usable": 1, "total_games": 20, "total_sets": 2,
        "game_margin": 4, "w_sets": 2, "l_sets": 0, "tiebreaks": 0,
        "winner_rank": 10.0, "loser_rank": 20.0, "winner_rank_points": 1.0,
        "loser_rank_points": 1.0, "winner_age": 25.0, "loser_age": 25.0,
        "winner_ht": 185.0, "loser_ht": 185.0, "winner_seed": np.nan,
        "loser_seed": np.nan, "winner_hand": "R", "loser_hand": "R",
        "score": score, "tourney_name": name,
        "w_ace": aces[0], "l_ace": aces[1], "w_df": 2, "l_df": 2,
        "w_svpt": svpt[0], "l_svpt": svpt[1],
        "w_1stIn": 50, "l_1stIn": 50, "w_1stWon": won[0] - 10, "l_1stWon": won[1] - 10,
        "w_2ndWon": 10, "l_2ndWon": 10, "w_SvGms": 10, "l_SvGms": 10,
        "w_bpSaved": 3, "l_bpSaved": 2, "w_bpFaced": 4, "l_bpFaced": 6,
    }


def play(engine, row):
    """Read features, then fold the result in -- the order training uses."""
    f = engine.features_for(row)
    engine.update(row)
    return f


# -- tiebreaks ---------------------------------------------------------------
def test_tiebreaks_won_reads_the_set_score():
    assert tiebreaks_won("7-6(4) 6-7(5) 7-6(2)") == (2, 1)
    assert tiebreaks_won("6-7(3) 6-4 7-6(5)") == (1, 1)
    assert tiebreaks_won("6-4 3-6 [10-8]") == (1, 0)
    assert tiebreaks_won("6-4 3-6 [8-10]") == (0, 1)
    assert tiebreaks_won("6-3 6-4") == (0, 0)
    assert tiebreaks_won("6-4 6-6 RET") == (0, 0)      # no tiebreak completed
    assert tiebreaks_won(None) == (0, 0)


def test_tb_winrate_counts_tiebreaks_not_matches():
    eng = FeatureEngine()
    # A wins the match 7-6 6-7 7-6: two tiebreaks won, one lost.
    play(eng, mk(1, "A", "B", 20200101, score="7-6(4) 6-7(5) 7-6(2)"))
    a, b = eng.players["A"], eng.players["B"]
    assert (a.tb_w, a.tb_n) == (2, 3)
    assert (b.tb_w, b.tb_n) == (1, 3)


def test_retirement_still_counts_a_completed_tiebreak():
    eng = FeatureEngine()
    row = mk(1, "A", "B", 20200101, score="7-6(3) 2-1 RET")
    row["status"], row["totals_usable"] = "retired", 0
    eng.update(row)
    assert (eng.players["A"].tb_w, eng.players["A"].tb_n) == (1, 1)


def test_shrinkage_pulls_small_samples_to_half():
    assert _shrunk(0, 0, 10) == 0.5
    assert _shrunk(1, 1, SHRINK_K["tb"]) == pytest.approx(0.5 + 0.5 / 11)
    assert _shrunk(80, 100, 10) > _shrunk(8, 10, 10)
    eng = FeatureEngine()
    f = play(eng, mk(1, "A", "B", 20200101, score="7-6(4) 6-3"))
    f = eng.features_for(mk(2, "A", "B", 20200108))
    p = "p1" if f["p1_id"] == "A" else "p2"
    assert f[f"{p}_tb_wr_s"] == pytest.approx(_shrunk(1, 1, SHRINK_K["tb"]))


# -- tournament speed --------------------------------------------------------
def test_speed_uses_prior_seasons_only():
    eng = FeatureEngine()
    play(eng, mk(1, "A", "B", 20200105, aces=(20, 20)))      # 40 aces / 160 pts
    first_2021 = play(eng, mk(2, "C", "D", 20210104, aces=(1, 1)))
    # The 2021 index is 2020's: 40/160. 2021's own result is not in it.
    assert first_2021["tspeed_ace"] == pytest.approx(40 / 160)
    assert first_2021["tspeed_n"] == 1
    later_2021 = play(eng, mk(3, "E", "F", 20210104, aces=(50, 50)))
    # Same event, same season: still 2020's index, whatever 2021 has produced.
    assert later_2021["tspeed_ace"] == first_2021["tspeed_ace"]
    assert eng.features_for(mk(4, "G", "H", 20220103))["tspeed_ace"] != first_2021["tspeed_ace"]


def test_speed_is_per_tournament_and_surface():
    eng = FeatureEngine()
    play(eng, mk(1, "A", "B", 20200105, name="Foo Open", aces=(20, 20)))
    other = eng.features_for(mk(2, "C", "D", 20210104, name="Bar Cup"))
    assert pd.isna(other["tspeed_ace"]) and other["tspeed_n"] == 0
    clay = eng.features_for(mk(3, "C", "D", 20210104, surface="Clay"))
    assert pd.isna(clay["tspeed_ace"])


def test_speed_without_a_name_is_missing_not_an_error():
    eng = FeatureEngine()
    row = mk(1, "A", "B", 20200105)
    del row["tourney_name"]
    f = play(eng, row)
    assert pd.isna(f["tspeed_ace"])


# -- opponent-adjusted serve/return -----------------------------------------
def test_adjusted_rates_read_the_opponents_state_before_the_match():
    eng = FeatureEngine()
    play(eng, mk(1, "B", "C", 20200105))          # gives B an ew_rpw / ew_spw history
    b = eng.players["B"]
    b_ret, b_srv = b.ew_rpw["slow"].get(), b.ew_spw["slow"].get()
    assert b_ret == b_ret and b_srv == b_srv
    # Fresh player A meets B. A's adjusted serve = A's match SPW - (1 - B's
    # return rate *going in*), and B's own update in this match must not leak
    # into that baseline.
    row = mk(2, "A", "B", 20200112, won=(60, 40), svpt=(80, 80))
    play(eng, row)
    a = eng.players["A"]
    expect_spw = (60 / 80) - (1.0 - b_ret)
    expect_rpw = (1.0 - 40 / 80) - (1.0 - b_srv)
    assert a.ew_adj_spw["slow"].get() == pytest.approx(expect_spw)
    assert a.ew_adj_rpw["slow"].get() == pytest.approx(expect_rpw)


def test_adjusted_rates_wait_for_an_opponent_with_history():
    eng = FeatureEngine()
    play(eng, mk(1, "A", "B", 20200105))          # neither has history beforehand
    assert np.isnan(eng.players["A"].ew_adj_spw["slow"].get())
    assert eng.players["A"].ew_spw["slow"].get() == eng.players["A"].ew_spw["slow"].get()


# -- surface serve/return ----------------------------------------------------
def test_surface_rates_blend_toward_the_overall_rate():
    eng = FeatureEngine()
    play(eng, mk(1, "A", "B", 20200105, surface="Hard", won=(60, 40)))
    play(eng, mk(2, "A", "C", 20200112, surface="Clay", won=(40, 40)))
    f = eng.features_for(mk(3, "A", "D", 20200119, surface="Clay"))
    p = "p1" if f["p1_id"] == "A" else "p2"
    overall, surf = f[f"{p}_ew_spw_slow"], eng.players["A"].surf_spw["Clay"].get()
    w = 1 / (1 + 15.0)
    assert f[f"{p}_spw_surf"] == pytest.approx(w * surf + (1 - w) * overall)
    # a surface never played falls back to the overall figure outright
    g = eng.features_for(mk(4, "A", "D", 20200126, surface="Grass"))
    q = "p1" if g["p1_id"] == "A" else "p2"
    assert g[f"{q}_spw_surf"] == pytest.approx(g[f"{q}_ew_spw_slow"])


# -- symmetric serve features ------------------------------------------------
def test_symmetric_serve_features_do_not_depend_on_orientation():
    eng = FeatureEngine()
    play(eng, mk(1, "A", "B", 20200105))
    play(eng, mk(2, "C", "A", 20200112))
    r = mk(3, "A", "C", 20200119)
    f1 = eng.features_for(r, force_winner_first=True)
    f2 = eng.features_for(r, force_winner_first=False)
    for c in ("spw_sum", "hold_prod", "exp_serve_sum"):
        assert f1[c] == pytest.approx(f2[c])
    assert f1["exp_serve_p1"] == pytest.approx(f2["exp_serve_p2"])


# -- season-opener snapshot --------------------------------------------------
def test_snapshot_is_the_rating_before_the_first_match_of_the_season():
    rows = [mk(1, "A", "B", 20241201), mk(2, "A", "C", 20241208),
            mk(3, "B", "C", 20241225), mk(4, "A", "B", 20250105)]
    df = pd.DataFrame(rows)
    _, eng = build_features(df, snapshot_start=20241225)
    # Replay only the matches before the opener, as the dashboard used to.
    _, before = build_features(df.iloc[:2])
    from tennis.features.build import _date_to_ord
    day0 = _date_to_ord(20241225)
    snap = eng.season_snap
    assert snap["start"] == 20241225
    for pid in ("A", "B", "C"):
        assert snap["ratings"][pid] == pytest.approx(before.elo.get(pid, "overall", day0))
    # and the live ratings have moved on from it
    assert eng.elo.get("A", "overall", day0) != snap["ratings"]["A"]


def test_snapshot_survives_incremental_extension():
    df = pd.DataFrame([mk(1, "A", "B", 20241201), mk(2, "B", "C", 20241225)])
    _, eng = build_features(df, snapshot_start=20241225)
    first = dict(eng.season_snap["ratings"])
    more = pd.DataFrame([mk(3, "A", "C", 20250105)])
    _, eng = build_features(more, eng, snapshot_start=20241225)
    assert eng.season_snap["ratings"] == first
    assert "D" not in first and BASE_RATING == pytest.approx(1500.0, abs=500)


# -- labels ------------------------------------------------------------------
def test_labels_come_from_the_match_frame_and_best_of_survives():
    rows = [mk(1, "A", "B", 20200105, best_of=5), mk(2, "A", "B", 20200112, best_of=3)]
    matches = pd.DataFrame(rows)
    feats, _ = build_features(matches)
    out = _attach_labels(feats, matches)
    assert "best_of" in out.columns
    assert not {"best_of_x", "best_of_y"} & set(out.columns)
    assert out["best_of"].tolist() == [5, 3]
    assert out["best_of"].notna().all()
    for c in LABEL_COLS:
        assert c in out.columns
    assert "best_of" not in LABEL_COLS
    assert out["y_total_games"].tolist() == [20, 20]
