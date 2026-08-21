"""Hierarchical Markov chain and the serve ratings that feed it."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tennis.models.markov import p_game, p_match, p_set, p_tiebreak


@pytest.mark.parametrize("p,expected", [
    (0.50, 0.5000), (0.60, 0.7357), (0.65, 0.8296),
    (0.70, 0.9008), (0.75, 0.9492),
])
def test_p_game_matches_published_hold_probabilities(p, expected):
    """The game formula is exact, so it must reproduce the standard table."""
    assert p_game(p) == pytest.approx(expected, abs=5e-4)


def test_equal_players_are_exactly_even_at_every_level():
    """Any symmetric input must give 0.5 -- the sharpest test of the DPs.

    A serve-order or indexing error in the tiebreak or set recursion shows up
    here immediately, because nothing else forces the two halves to balance.
    """
    for p in (0.55, 0.62, 0.70):
        assert p_tiebreak(p, p) == pytest.approx(0.5, abs=1e-12)
        assert p_set(p, p) == pytest.approx(0.5, abs=1e-12)
        for bo in (3, 5):
            assert p_match(p, p, bo) == pytest.approx(0.5, abs=1e-12)


def test_match_probabilities_are_complementary():
    """P(A beats B) + P(B beats A) == 1, or probability is leaking somewhere."""
    for a, b in ((0.66, 0.61), (0.70, 0.58), (0.62, 0.64)):
        for bo in (3, 5):
            assert p_match(a, b, bo) + p_match(b, a, bo) == pytest.approx(1.0, abs=1e-12)


def test_longer_format_favours_the_better_player():
    """Best-of-five converts the same edge into a bigger probability."""
    assert p_match(0.67, 0.63, 5) > p_match(0.67, 0.63, 3) > 0.5
    assert p_match(0.61, 0.63, 5) < p_match(0.61, 0.63, 3) < 0.5


def test_chain_amplifies_a_small_serve_edge():
    """The reason to use the chain at all: the scoring system is an amplifier.

    A couple of points of serve is worth many points of match probability, and
    that mapping is what the chain supplies exactly instead of learning it.
    """
    edge = p_match(0.66, 0.64, 3) - 0.5
    assert edge > 0.08, "2 points of serve should move the match a lot more"


def test_probabilities_stay_in_range_across_the_grid():
    g = np.linspace(0.30, 0.90, 13)
    a, b = np.meshgrid(g, g)
    for v in (p_set(a, b), p_match(a, b, 3), p_match(a, b, 5)):
        assert np.isfinite(v).all()
        assert ((v >= 0.0) & (v <= 1.0)).all()


def _match_row(mid, w, l, **kw):
    row = {"match_id": mid, "seq": 0, "winner_id": w, "loser_id": l,
           "best_of": 3, "surface": "Hard",
           "w_svpt": 80.0, "w_1stWon": 35.0, "w_2ndWon": 15.0,
           "l_svpt": 80.0, "l_1stWon": 30.0, "l_2ndWon": 12.0}
    row.update(kw)
    return row


def test_serve_ratings_survive_a_missing_counter():
    """A NaN counter must be skipped, never folded into a rating.

    `NaN or 0` returns NaN, because NaN is truthy. That guard silently turned
    the entire rating pool into NaN: one match with a missing 1stWon poisoned
    that player, and the poison spread to every opponent they later met.
    """
    from tennis.models import serve_ratings as SR

    rows = [_match_row(f"m{i}", "A", "B") for i in range(6)]
    rows[2]["w_1stWon"] = np.nan          # one missing counter
    rows.append(_match_row("m6", "A", "C"))
    out = SR.fit(pd.DataFrame(rows), 0.62)

    assert out[["w_serve", "w_ret", "l_serve", "l_ret"]].notna().all().all()
    # A won every point exchange, so their serve rating must be positive by
    # the end -- proof the good matches still updated around the bad one.
    assert out.iloc[-1]["w_serve"] > 0


def test_serve_ratings_are_pre_match():
    """The first match a player appears in must carry a zero rating.

    Ratings are emitted before the update, so a match is never rated using its
    own result -- the same discipline as the Elo state.
    """
    from tennis.models import serve_ratings as SR

    out = SR.fit(pd.DataFrame([_match_row("m0", "A", "B"),
                               _match_row("m1", "A", "B")]), 0.62)
    assert out.iloc[0][["w_serve", "w_ret", "l_serve", "l_ret"]].eq(0).all()
    assert out.iloc[1]["w_serve"] != 0      # now it has history


def test_serve_ratings_shrink_thin_histories():
    """Two matches must not produce a large rating, however lopsided.

    The chain is steeply convex, so an unshrunk rating from a thin sample is
    exactly what produced the slope-0.34 overconfidence.
    """
    from tennis.models import serve_ratings as SR

    lopsided = [_match_row(f"m{i}", "A", "B", w_1stWon=60.0, w_2ndWon=15.0)
                for i in range(2)]
    out = SR.fit(pd.DataFrame(lopsided), 0.62)
    assert abs(out.iloc[-1]["w_serve"]) < 0.05
