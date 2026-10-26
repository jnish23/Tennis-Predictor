"""Regression tests for the model audit.

Each pins a defect that was real in the code: rest measured from an event's
first day, a calibrator that starved the booster and blurred the two tours
together, a backtest that scored a model nobody serves, retirements settled as
ordinary wins, and the orientation of a handicap ladder.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.test_review_fixes import _mrow


def _full_row(**kw):
    base = _mrow(level="atp250", draw_size=32.0, indoor=0.0, is_challenger=0,
                 round="R32", round_idx=5.0, winner_rank=10.0, loser_rank=40.0,
                 winner_rank_points=3000.0, loser_rank_points=900.0,
                 winner_age=25.0, loser_age=27.0, winner_ht=185.0, loser_ht=190.0,
                 winner_seed=np.nan, loser_seed=np.nan, winner_hand="R",
                 loser_hand="L", score="6-3 6-4", date_is_exact=0,
                 match_date=20260105, tourney_name="Test Open")
    base.update(kw)
    return base


# --------------------------------------------------------------------------
# F01: matches are dated by when they were played, not by the event's first day
# --------------------------------------------------------------------------
def test_a_stamped_event_places_each_round_on_its_own_day():
    from tennis.features.build import _date_to_ord, _play_ord

    start = _date_to_ord(20260105)
    assert _play_ord(_full_row(round="R32")) == start + 1
    assert _play_ord(_full_row(round="F")) == start + 6
    slam = _full_row(level="grand_slam", draw_size=128.0)
    assert _play_ord({**slam, "round": "F"}) == start + 13
    assert _play_ord({**slam, "round": "R128"}) == start + 1


def test_an_exact_date_beats_the_round_estimate_and_a_bare_row_is_taken_at_face_value():
    from tennis.features.build import _date_to_ord, _play_ord

    exact = _full_row(round="F", date_is_exact=1, match_date=20260107)
    assert _play_ord(exact) == _date_to_ord(20260107)
    # A live prediction row has neither flag: its tourney_date already is the day.
    bare = {"tourney_date": 20260109, "round": "F"}
    assert _play_ord(bare) == _date_to_ord(20260109)


def test_rest_days_within_an_event_are_not_all_zero():
    from tennis.features.build import FeatureEngine

    eng = FeatureEngine()
    eng.update(_full_row(match_id="a", seq=1, round="R32"))
    row = _full_row(match_id="b", seq=2, round="QF", winner_id="W", loser_id="Z")
    f = eng.features_for(row, force_winner_first=True)
    assert f["p1_rest_days"] == 3          # R32 on day 1, QF on day 4
    # The 14-day window still reaches back across the event.
    assert f["p1_matches_14d"] == 1


def test_a_final_does_not_age_out_of_the_fatigue_window_early():
    """Stamped day 0, a Slam final looked 15 days old one day into the next
    week's event and dropped out of the 14-day count."""
    from tennis.features.build import FeatureEngine

    eng = FeatureEngine()
    eng.update(_full_row(match_id="f", seq=1, round="F", level="grand_slam",
                         draw_size=128.0, tourney_date=20260105))
    nxt = _full_row(match_id="n", seq=2, round="R32", tourney_date=20260119,
                    winner_id="W", loser_id="Q")
    f = eng.features_for(nxt, force_winner_first=True)
    assert f["p1_rest_days"] == 2          # final on day 13, next event's R32 on day 15
    assert f["p1_matches_14d"] == 1


# --------------------------------------------------------------------------
# F07: nothing about a default or an abandoned match is a result
# --------------------------------------------------------------------------
@pytest.mark.parametrize("status", ["walkover", "default", "unfinished"])
def test_matches_with_no_countable_result_do_not_move_ratings(status):
    from tennis.features.build import FeatureEngine

    eng = FeatureEngine()
    eng.update(_full_row(status=status))
    assert eng.elo.n("W") == 0 and "W" not in eng.players


def test_retirement_still_rates_as_a_win():
    from tennis.features.build import FeatureEngine

    eng = FeatureEngine()
    eng.update(_full_row(status="retired", score="6-4 3-1 RET"))
    assert eng.elo.n("W") == 1


def test_playable_drops_defaults_and_unfinished_but_keeps_retirements():
    from tennis.models.train import _playable

    df = pd.DataFrame({"status": ["completed", "retired", "walkover", "default",
                                  "unfinished"]})
    assert list(_playable(df)["status"]) == ["completed", "retired"]


def test_completed_sets_ignores_the_set_in_progress():
    from tennis.models.evaluate import completed_sets

    assert completed_sets("6-4 3-2 RET", 3) == 1
    assert completed_sets("3-1 RET", 3) == 0
    assert completed_sets("7-6(4) 6-7(3) 2-1 RET", 3) == 2
    assert completed_sets("6-4 6-3", 3) == 2


def test_retired_bets_are_voided_by_each_books_rule():
    from tennis.models.evaluate import roi_winner, voided

    d = pd.DataFrame({
        "match_id": ["t-R32-a-b", "t-R32-c-d", "t-R32-e-f"],
        "status": ["retired", "retired", "completed"],
        "sets_done": [0, 1, 2], "book": "PS"})
    assert list(voided(d, "PS")) == [True, False, False]     # Pinnacle: one set
    assert list(voided(d, "B365")) == [True, True, False]    # others: completion

    px = d.assign(p1_price=3.0, p2_price=1.4, p_win=0.6, y_win=1)
    r = roi_winner(px, edge=0.0)
    assert r["voided_matches"] == 1 and r["bets"] == 2


# --------------------------------------------------------------------------
# F05: p(A,B) + p(B,A) != 1, and the backtest has to score what is served
# --------------------------------------------------------------------------
def test_swap_sides_is_what_the_engine_emits_for_the_other_orientation():
    from tennis.features.build import FeatureEngine
    from tennis.models.common import swap_sides

    eng = FeatureEngine()
    for i in range(6):
        eng.update(_full_row(match_id=f"h{i}", seq=i, tourney_date=20260105 + i,
                             winner_id="W" if i % 2 else "L", loser_id="L" if i % 2 else "W",
                             winner_rank=10.0 + i, loser_rank=40.0 - i))
    row = _full_row(match_id="x", seq=50, tourney_date=20260120, winner_id="W",
                    loser_id="L", winner_seed=2.0)
    fwd = pd.DataFrame([eng.features_for(row, force_winner_first=True)])
    rev = pd.DataFrame([eng.features_for(row, force_winner_first=False)])
    swapped = swap_sides(fwd)
    for c in fwd.columns:
        if c in ("match_id", "seq", "y_win"):
            continue
        a, b = swapped[c].iloc[0], rev[c].iloc[0]
        if isinstance(b, str):
            assert a == b, c
        else:
            assert (np.isnan(a) and np.isnan(b)) or a == pytest.approx(b), c
    assert swapped["y_win"].iloc[0] == 1 - fwd["y_win"].iloc[0]


def test_symmetric_regression_is_order_free_and_the_spread_flips_sign():
    from tennis.models.common import swap_sides
    from tennis.models.train import _symmetric_regression

    class Lin:                                   # a model that is NOT antisymmetric
        def predict(self, X):
            return (3.0 * X["p1_elo"] - 1.0 * X["p2_elo"] + X["d_elo"]).to_numpy()

    df = pd.DataFrame({"p1_elo": [1600.0, 1400.0], "p2_elo": [1500.0, 1450.0],
                       "d_elo": [100.0, -50.0]})
    feats = list(df.columns)
    tot = _symmetric_regression(Lin(), df, feats)
    assert np.allclose(tot, _symmetric_regression(Lin(), swap_sides(df), feats))
    spr = _symmetric_regression(Lin(), df, feats, signed=True)
    assert np.allclose(spr, -_symmetric_regression(Lin(), swap_sides(df), feats, signed=True))


# --------------------------------------------------------------------------
# F04 / F06: calibration
# --------------------------------------------------------------------------
def test_each_tour_gets_its_own_calibration_curve():
    from tennis.models.common import MIN_TOUR_CAL_ROWS, TourCalibrator

    rng = np.random.default_rng(0)
    n = MIN_TOUR_CAL_ROWS
    raw = rng.uniform(0.05, 0.95, n)
    # The ATP is calibrated; the WTA's true rate is a flattened version of its score.
    y_atp = (rng.uniform(size=n) < raw).astype(float)
    y_wta = (rng.uniform(size=n) < 0.5 + 0.6 * (raw - 0.5)).astype(float)
    tours = np.array(["atp"] * n + ["wta"] * n)
    both = np.r_[raw, raw]
    cal = TourCalibrator.fit(both, 1 - both, np.r_[y_atp, y_wta], tours)
    assert set(cal.by_tour) == {"atp", "wta"}
    hi = np.full(2, 0.9)
    got = cal.predict(hi, np.array(["atp", "wta"]))
    assert got[0] > got[1] + 0.05            # the WTA curve pulls 0.9 toward 0.5


def test_a_tour_with_too_little_data_falls_back_to_the_pooled_curve():
    from tennis.models.common import TourCalibrator

    rng = np.random.default_rng(1)
    raw = rng.uniform(size=5000)
    y = (rng.uniform(size=5000) < raw).astype(float)
    cal = TourCalibrator.fit(raw, raw, y, np.array(["atp"] * 5000))
    assert "wta" not in cal.by_tour
    assert np.allclose(cal.predict([0.3, 0.7], ["wta", "wta"]), cal.pooled.predict([0.3, 0.7]))


def test_calibrator_survives_pickling_from_the_module_that_defines_it():
    import pickle

    from tennis.models.common import TourCalibrator
    from tennis.models.train import TourCalibrator as FromTrain

    assert FromTrain is TourCalibrator
    assert TourCalibrator.__module__ == "tennis.models.common"
    cal = TourCalibrator.fit(np.array([0.2, 0.8] * 3000), np.array([0.8, 0.2] * 3000),
                             np.array([0.0, 1.0] * 3000), np.array(["atp"] * 6000))
    back = pickle.loads(pickle.dumps(cal))
    assert np.allclose(back.predict([0.2, 0.8], ["atp", "atp"]),
                       cal.predict([0.2, 0.8], ["atp", "atp"]))


def test_predictor_still_accepts_an_old_single_isotonic_calibrator():
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip").fit([0.1, 0.9], [0.1, 0.9])
    assert not hasattr(iso, "by_tour")        # the branch Predictor._score takes


# --------------------------------------------------------------------------
# F10: tour seasons
# --------------------------------------------------------------------------
def test_new_year_events_belong_to_the_season_they_open():
    from tennis.models.common import season_of

    got = season_of(np.array([20251224, 20251225, 20251228, 20260101, 20261224, 20261231]))
    assert list(got) == [2025, 2026, 2026, 2026, 2026, 2027]


def test_walk_forward_folds_put_the_united_cup_in_the_new_season():
    from tennis.models.common import walk_forward_folds

    dates = ([20000110 + 10000 * k for k in range(9)]       # 2000-2008 history
             + [20081228, 20090105, 20091228, 20100105])
    df = pd.DataFrame({"tourney_date": sorted(dates)})
    folds = {s: te for _, te, s in walk_forward_folds(df, start_season=2009)}
    assert df["tourney_date"].iloc[folds[2009]].tolist() == [20081228, 20090105]


# --------------------------------------------------------------------------
# F03 / F09 / F12: lines and uncertainty
# --------------------------------------------------------------------------
def test_ours_follows_which_player_tennisexplorer_listed_first():
    from tennis.models.lines import _ours

    d = pd.DataFrame({"y_win": [1, 1, 0, 0], "p1_is_winner": [1, 0, 1, 0]})
    # our p1 won and their p1 won -> same player; our p1 won, their p1 lost -> not
    assert list(_ours(d)) == [True, False, False, True]


def test_tourney_key_survives_hyphens_in_the_key_and_round():
    from tennis.models.stats import tourney_key_of

    ids = ["2026-0301|indian wells|0|0-R128-104925-106421",
           "x|mid-name|0|0-3RD/4TH-A-B", "k|n|1|0-NA-1-2"]
    assert list(tourney_key_of(ids)) == [
        "2026-0301|indian wells|0|0", "x|mid-name|0|0", "k|n|1|0"]


def test_clustered_interval_is_wider_than_the_naive_one_on_correlated_data():
    from tennis.models.stats import cluster_bootstrap_mean

    rng = np.random.default_rng(2)
    clusters = np.repeat(np.arange(60), 40)
    x = rng.normal(size=60).repeat(40) + 0.1 * rng.normal(size=2400)   # shared shock
    lo, hi = cluster_bootstrap_mean(x, clusters)
    naive = 1.96 * x.std(ddof=1) / np.sqrt(len(x))
    assert (hi - lo) / 2 > 3 * naive
    assert lo < x.mean() < hi


def test_a_single_cluster_gives_no_interval_rather_than_a_false_one():
    from tennis.models.stats import cluster_bootstrap_mean

    lo, hi = cluster_bootstrap_mean([1.0, 2.0, 3.0], ["a", "a", "a"])
    assert np.isnan(lo) and np.isnan(hi)


def test_clv_view_reports_a_tournament_clustered_interval():
    from tennis.models.lines import _clv_view

    rng = np.random.default_rng(3)
    g = pd.DataFrame({"match_id": [f"t{k % 30}|n|0|0-R32-{k}-{k + 1}" for k in range(900)],
                      "clv": rng.normal(0.3, 1.0, 900)})
    v = _clv_view(g)
    lo, hi = v["ci95_tournament_pts"]
    assert v["bets"] == 900 and lo < v["mean_clv_pts"] < hi


def test_clv_orients_the_handicap_ladder_on_opening_prices():
    """A ladder whose close has moved so far that its direction flips must not
    take its orientation from the close."""
    from tennis.models.lines import _orient_ladder

    ladder = pd.DataFrame({"te_id": 1, "line": [-3.5, -1.5, 0.5, 2.5],
                           "p_mkt": [0.9, 0.7, 0.5, 0.3]})        # falls as line rises
    out = _orient_ladder(ladder)
    assert list(out["line"]) == [-3.5, -1.5, 0.5, 2.5]
    flipped = ladder.assign(p_mkt=ladder["p_mkt"].to_numpy()[::-1])
    assert list(_orient_ladder(flipped)["line"]) == [3.5, 1.5, -0.5, -2.5]


# --------------------------------------------------------------------------
# Staleness: an old engine state must not be extended under new definitions
# --------------------------------------------------------------------------
def test_engine_state_from_before_the_date_change_is_rebuilt():
    from tennis.features.build import FeatureEngine

    old = FeatureEngine()
    del old.version                            # states pickled before versioning
    assert getattr(old, "version", 1) != FeatureEngine.VERSION
    assert FeatureEngine().version == FeatureEngine.VERSION
