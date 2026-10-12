"""Per-set labels from score strings."""
from tennis.ingest.parse import set_winners


def test_set_winners_orients_on_the_match_winner():
    assert set_winners("2-6 7-6(3) 6-1") == [0, 1, 1]
    assert set_winners("7-6(5) 6-7(4) 6-4") == [1, 0, 1]


def test_interrupted_set_is_not_a_result():
    # a retirement mid-set must not invent a set-2 label from 2-1
    assert set_winners("6-3 2-1 RET") == [1]
    assert set_winners("6-4 6-3 RET") == [1, 1]


def test_degenerate_scores():
    assert set_winners("W/O") == []
    assert set_winners(None) == []
    assert set_winners(float("nan")) == []
