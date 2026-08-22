"""Parser, simulator and prediction-coherence tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tennis.ingest.parse import (
    COMPLETED,
    DEFAULT,
    RETIRED,
    UNFINISHED,
    WALKOVER,
    parse_score,
)


# --------------------------------------------------------------------------
# score parsing
# --------------------------------------------------------------------------
def test_completed_straight_sets():
    p = parse_score("6-4 6-3", 3)
    assert p.status == COMPLETED and p.usable_for_totals
    assert (p.w_sets, p.l_sets) == (2, 0)
    assert p.total_games == 19 and p.game_margin == 5


def test_tiebreaks_counted():
    p = parse_score("7-6(5) 6-7(4) 7-6(8)", 3)
    assert p.tiebreaks == 3 and p.usable_for_totals
    assert p.total_games == 39  # 13 games in each of three tiebreak sets
    assert (p.w_games, p.l_games) == (20, 19)


def test_five_setter():
    p = parse_score("4-6 7-5 6-2 3-6 6-4", 5)
    assert p.usable_for_totals and (p.w_sets, p.l_sets) == (3, 2)
    assert p.total_games == 49


@pytest.mark.parametrize("s,status", [
    ("6-0 3-1 RET", RETIRED),
    ("W/O", WALKOVER),
    ("6-2 6-7(5) DEF", DEFAULT),
    ("", UNFINISHED),
    (None, UNFINISHED),
    (float("nan"), UNFINISHED),
])
def test_non_completions_are_labelled_not_dropped(s, status):
    p = parse_score(s, 3)
    assert p.status == status
    # The whole point: these are classified, and excluded from totals only.
    assert not p.usable_for_totals


def test_retirement_keeps_partial_score_but_not_as_a_total():
    p = parse_score("6-0 3-1 RET", 3)
    assert p.total_games == 10       # partial games still readable
    assert not p.usable_for_totals   # but never used as a totals label


def test_best_of_mislabel_still_parses():
    """A 3-set win recorded as best_of=5 is a source error, not an unfinished match."""
    p = parse_score("6-4 6-4 6-4", 3)
    assert p.usable_for_totals and p.w_sets == 3


def test_garbage_rejected():
    assert not parse_score("60-0 xx", 3).usable_for_totals


# --------------------------------------------------------------------------
# simulator invariants
# --------------------------------------------------------------------------
class _FakePredictor:
    """Deterministic stand-in so simulator tests don't need trained models."""

    def __init__(self, n):
        self.n = n

    def win_prob_matrix(self, ids, ctx):
        n = len(ids)
        m = np.full((n, n), 0.5)
        for i in range(n):
            for j in range(n):
                if i != j:
                    # earlier index = stronger
                    m[i, j] = 1 / (1 + 10 ** ((i - j) / 4))
        np.fill_diagonal(m, 0.5)
        return m


def _draw(n):
    from tennis.models.predict import MatchContext
    from tennis.sim.bracket import Draw

    ids = [f"P{i}" for i in range(n)]
    return Draw(name="t", slots=ids, ctx=MatchContext(),
                player_names={i: i for i in ids})


def test_title_probabilities_sum_to_one():
    from tennis.sim.bracket import simulate

    d = _draw(16)
    res = simulate(d, _FakePredictor(16), n_sims=4000, seed=1)
    assert res["Champion"].sum() == pytest.approx(1.0, abs=1e-9)


def test_each_round_sums_to_its_slot_count():
    from tennis.sim.bracket import simulate

    d = _draw(32)
    res = simulate(d, _FakePredictor(32), n_sims=4000, seed=2)
    for label, expected in [("R16", 16), ("QF", 8), ("SF", 4),
                            ("Final", 2), ("Champion", 1)]:
        assert res[label].sum() == pytest.approx(expected, abs=1e-9), label


def test_stronger_player_wins_more_often():
    from tennis.sim.bracket import simulate

    res = simulate(_draw(16), _FakePredictor(16), n_sims=8000, seed=3)
    res = res.set_index("player_id")
    assert res.loc["P0", "Champion"] > res.loc["P15", "Champion"]


def test_byes_pad_non_power_of_two_draw():
    d = _draw(12)
    assert d.size == 16
    assert d.slots.count("__BYE__") == 4


def test_simulation_is_not_deterministic_chaining():
    """A Monte Carlo run must give the favourite <100% title probability.

    Guards against regressing to a deterministic round-by-round chain, which
    CLAUDE.md rules out: that would hand the top seed a 1.0.
    """
    from tennis.sim.bracket import simulate

    res = simulate(_draw(16), _FakePredictor(16), n_sims=4000, seed=4)
    assert 0.0 < res["Champion"].max() < 0.95


# --------------------------------------------------------------------------
# prediction coherence
# --------------------------------------------------------------------------
@pytest.mark.skipif(
    not (__import__("tennis.config", fromlist=["ARTIFACTS"]).ARTIFACTS / "models.pkl").exists(),
    reason="production models not trained",
)
@pytest.mark.needs_db
def test_predictions_are_orientation_symmetric():
    from tennis.db.schema import connect
    from tennis.models.predict import MatchContext, Predictor

    con = connect()
    ids = pd.read_sql(
        "SELECT player_id FROM elo_state WHERE scope='overall' AND n_matches>50 "
        "ORDER BY rating DESC LIMIT 6", con)["player_id"].tolist()
    con.close()

    p = Predictor()
    ctx = MatchContext(surface="Hard", level="atp250", best_of=3)
    for a, b in [(ids[0], ids[1]), (ids[2], ids[5])]:
        fwd = p.predict_many([(a, b)], ctx).iloc[0]
        rev = p.predict_many([(b, a)], ctx).iloc[0]
        assert fwd["p1_win_prob"] + rev["p1_win_prob"] == pytest.approx(1.0, abs=1e-9)
        assert fwd["pred_total_games"] == pytest.approx(rev["pred_total_games"], abs=1e-9)
        assert fwd["pred_spread"] == pytest.approx(-rev["pred_spread"], abs=1e-9)


# --------------------------------------------------------------------------
# incremental update path
# --------------------------------------------------------------------------
@pytest.mark.needs_db
def test_incremental_matches_full_recompute(tmp_path):
    """Resuming from pickled state must equal a from-scratch rebuild.

    This is the property the nightly job depends on: it replays only new
    matches against saved state instead of recomputing 200k rows, and that is
    only safe if the two paths agree exactly.
    """
    from tennis.db.schema import connect
    from tennis.features.build import FeatureEngine, build_features
    from tennis.features.pipeline import load_matches

    con = connect()
    m = load_matches(con)
    con.close()
    m = m[m["seq"] < 40_000]
    split = 30_000

    full, _ = build_features(m, FeatureEngine())
    head, tail = m[m["seq"] < split], m[m["seq"] >= split]
    f1, eng = build_features(head, FeatureEngine())

    state = tmp_path / "state.pkl"
    eng.save(state)
    eng2 = FeatureEngine.load(state)

    f2, _ = build_features(tail, eng2, emit_from_seq=eng2.last_seq)
    inc = pd.concat([f1, f2], ignore_index=True).sort_values("seq").reset_index(drop=True)
    full = full.sort_values("seq").reset_index(drop=True)

    assert list(full["match_id"]) == list(inc["match_id"])
    for c in full.columns:
        if pd.api.types.is_numeric_dtype(full[c]):
            assert np.allclose(full[c].to_numpy(float), inc[c].to_numpy(float),
                               equal_nan=True), f"{c} diverged"


# --------------------------------------------------------------------------
# draw reconstruction
# --------------------------------------------------------------------------
@pytest.mark.needs_db
def test_bracket_reconstruction_matches_known_result():
    """Roland Garros 2024: bracket, finalists and champion must come out right."""
    from tennis.db.schema import connect
    from tennis.sim.draws import actual_progression, build_draw

    con = connect()
    try:
        tk = pd.read_sql(
            "SELECT tourney_key FROM tournaments WHERE name='Roland Garros' "
            "AND season=2024", con)
        if tk.empty:
            pytest.skip("Roland Garros 2024 not loaded")
        built = build_draw(con, tk.iloc[0, 0])
    finally:
        con.close()
    assert built is not None
    draw, g = built

    assert draw.size == 128 and draw.n_rounds == 7
    assert draw.ctx.best_of == 5 and draw.ctx.surface == "Clay"
    # every slot filled by a distinct real player
    real = [s for s in draw.slots if s != "__BYE__"]
    assert len(real) == len(set(real)) == 128

    ap = actual_progression(g, draw.slots)
    champs = ap.loc[ap["Champion"] == 1, "player_id"]
    assert len(champs) == 1
    assert draw.player_names[champs.iloc[0]] == "Carlos Alcaraz"
    finalists = {draw.player_names[p] for p in ap.loc[ap["Final"] == 1, "player_id"]}
    assert finalists == {"Carlos Alcaraz", "Alexander Zverev"}
    assert (ap["SF"] == 1).sum() == 4 and (ap["QF"] == 1).sum() == 8


@pytest.mark.needs_db
def test_reconstruction_rate_is_high():
    """The recovery rate across recent seasons must not regress."""
    from tennis.db.schema import connect
    from tennis.sim.draws import TOURNEY_SQL, reconstruct_slots

    con = connect()
    try:
        g = pd.read_sql(TOURNEY_SQL + " AND t.season >= 2023", con)
    finally:
        con.close()
    total = ok = 0
    for _, sub in g.groupby("tourney_key"):
        if len(sub["round_idx"].dropna().unique()) < 2:
            continue
        total += 1
        if reconstruct_slots(sub) is not None:
            ok += 1
    assert total > 200
    assert ok / total > 0.95, f"reconstruction rate fell to {ok/total:.1%}"


@pytest.mark.needs_db
def test_actual_progression_credits_byes():
    """A player who receives a first-round bye still counts as advancing."""
    from tennis.db.schema import connect
    from tennis.sim.draws import actual_progression, build_draw, list_replayable

    con = connect()
    try:
        idx = list_replayable(con, min_season=2023)
        # a draw padded with byes has fewer players than slots
        withbyes = idx[idx["n_players"] < idx["draw_size"]]
        if withbyes.empty:
            pytest.skip("no bye-padded draws available")
        built = build_draw(con, withbyes.iloc[0]["tourney_key"])
    finally:
        con.close()
    assert built is not None
    draw, g = built
    ap = actual_progression(g, draw.slots)
    # exactly one champion, and the round counts halve as the bracket narrows
    assert (ap["Champion"] == 1).sum() == 1
    assert (ap["Final"] == 1).sum() == 2


@pytest.mark.needs_db
def test_feed_name_resolution_handles_both_formats():
    """'Jannik Sinner' and 'Sinner J.' must resolve to the same player.

    The feed's name format is not documented, so the resolver has to cope with
    either. Getting this wrong would silently simulate the wrong player.
    """
    from tennis.ingest.draws_api import resolve_players

    r = resolve_players([
        "Jannik Sinner", "Sinner J.", "Carlos Alcaraz", "Alcaraz C.",
        "Felix Auger-Aliassime", "Auger-Aliassime F.", "Nonexistent Person",
    ])
    assert r["Jannik Sinner"] and r["Jannik Sinner"] == r["Sinner J."]
    assert r["Carlos Alcaraz"] and r["Carlos Alcaraz"] == r["Alcaraz C."]
    assert r["Felix Auger-Aliassime"] == r["Auger-Aliassime F."]
    # unmatched names surface as None rather than a wrong guess
    assert r["Nonexistent Person"] is None


# --------------------------------------------------------------------------
# draw sheets and byes
# --------------------------------------------------------------------------
def _sheet_payload(n_byes: int = 2, n_matches: int = 2) -> dict:
    """Synthetic draw sheet in the feed's shape: byes marked result='bye'."""
    singles, pos = [], 0
    for i in range(n_byes):
        pos += 1
        singles.append({
            "roundId": 4, "draw": pos, "result": "bye", "seed1": str(i + 1),
            "player1Id": 100 + i, "player2Id": 3700,
            "player1": {"id": 100 + i, "name": f"Seed Player {i}", "seed": str(i + 1)},
            "player2": {"id": 3700, "name": "Unknown Player"},
        })
    for j in range(n_matches):
        pos += 1
        singles.append({
            "roundId": 4, "draw": pos, "result": "", "seed1": None,
            "player1Id": 200 + j, "player2Id": 300 + j,
            "player1": {"id": 200 + j, "name": f"Unseeded A{j}"},
            "player2": {"id": 300 + j, "name": f"Unseeded B{j}"},
        })
    # later-round placeholders repeat the same players and must be ignored
    singles.append({
        "roundId": 5, "draw": 1, "result": "", "player1Id": 100, "player2Id": 3700,
        "player1": {"id": 100, "name": "Seed Player 0"},
        "player2": {"id": 3700, "name": "Unknown Player"},
    })
    return {"data": {"singles": singles, "doubles": [], "qualifying": []}}


def test_draw_sheet_keeps_bye_players():
    """A player with a first-round bye must survive into the bracket.

    The fixtures feed omits them entirely (it only publishes a match once both
    players are known), which silently deleted all 32 seeds from a 96-draw
    Masters and inflated everyone else's title chances.
    """
    from tennis.ingest.draws_api import parse_draw_sheet

    sheet = parse_draw_sheet(_sheet_payload(n_byes=2, n_matches=2))
    assert len(sheet) == 4                      # opening round only
    assert set(sheet["round_id"]) == {4}        # round-5 placeholder dropped
    assert int(sheet["is_bye"].sum()) == 2
    byes = sheet[sheet["is_bye"]]
    assert list(byes["p1_name"]) == ["Seed Player 0", "Seed Player 1"]
    assert byes["p2_name"].isna().all()         # sentinel opponent not kept
    assert list(sheet["draw_pos"]) == [1, 2, 3, 4]


@pytest.mark.needs_db
def test_draw_sheet_bracket_has_bye_slots():
    """Bye players occupy a real slot paired with BYE, preserving draw order."""
    from tennis.ingest.draws_api import build_draw_from_sheet, parse_draw_sheet
    from tennis.sim.bracket import BYE

    sheet = parse_draw_sheet(_sheet_payload(n_byes=2, n_matches=2))
    # Unknown names resolve to None, which must still hold a slot rather than
    # collapsing the bracket.
    draw, unresolved = build_draw_from_sheet(
        sheet, name="t", surface="Hard", level="masters")
    assert draw.size == 8
    assert len(unresolved) == 6  # synthetic players are not in our database
    # slot layout: (p, BYE), (p, BYE), (a, b), (a, b)
    assert draw.slots[1] == BYE and draw.slots[3] == BYE


def test_bye_player_advances_first_round_with_certainty():
    """A bye must give a 100% chance of clearing the opening round."""
    from tennis.models.predict import MatchContext
    from tennis.sim.bracket import BYE, Draw, round_names, simulate

    slots = ["P0", BYE, "P1", BYE, "P2", "P3", "P4", "P5"]
    draw = Draw(name="t", slots=slots, ctx=MatchContext(),
                player_names={f"P{i}": f"P{i}" for i in range(6)})
    res = simulate(draw, _FakePredictor(6), n_sims=4000, seed=7).set_index("player_id")
    # An 8-slot bracket has three rounds, so its opening round is labelled "SF".
    opening = round_names(3)[0]
    assert res.loc["P0", opening] == pytest.approx(1.0)
    assert res.loc["P1", opening] == pytest.approx(1.0)
    assert res.loc["P2", opening] < 1.0          # had to actually play
    assert res["Champion"].sum() == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------
# conditional simulation (rounds already played)
# --------------------------------------------------------------------------
def test_resolved_matches_are_not_simulated():
    """A pinned result must hold in every playthrough."""
    from tennis.sim.bracket import round_names, simulate

    d = _draw(8)
    # P7 is the weakest; pinning the upset must give it certainty, not ~35%
    free = simulate(d, _FakePredictor(8), n_sims=3000, seed=1).set_index("player_id")
    pinned = simulate(d, _FakePredictor(8), n_sims=3000, seed=1,
                      resolved={0: {3: "P7"}}).set_index("player_id")
    opening = round_names(3)[0]
    assert 0.0 < free.loc["P7", opening] < 1.0
    assert pinned.loc["P7", opening] == pytest.approx(1.0)
    assert pinned.loc["P6", opening] == pytest.approx(0.0)
    assert pinned["Champion"].sum() == pytest.approx(1.0, abs=1e-9)


@pytest.mark.needs_db
def test_walk_bracket_pins_completed_rounds_only():
    """Rounds are pinned up to the first incomplete one, and no further."""
    from tennis.db.schema import connect
    from tennis.sim.draws import build_draw, walk_bracket

    con = connect()
    try:
        tk = pd.read_sql(
            "SELECT tourney_key FROM tournaments WHERE name='Roland Garros' "
            "AND season=2024", con)
        if tk.empty:
            pytest.skip("Roland Garros 2024 not loaded")
        built = build_draw(con, tk.iloc[0, 0])
    finally:
        con.close()
    draw, g = built

    # full history: every round resolvable
    resolved, _, completed = walk_bracket(draw, g)
    assert completed == draw.n_rounds
    # 128-draw: 64+32+16+8+4+2+1 = 127 matches
    assert sum(len(v) for v in resolved.values()) == 127

    # truncated: only the first three rounds pinned
    r3, _, done3 = walk_bracket(draw, g, through_round=3)
    assert done3 == 3
    assert sum(len(v) for v in r3.values()) == 64 + 32 + 16

    # drop one first-round result: nothing beyond round 0 can be pinned
    partial = g[~((g["round_idx"] == g["round_idx"].min())
                  & (g["match_num"] == g["match_num"].min()))]
    _, _, done_p = walk_bracket(draw, partial)
    assert done_p == 0


@pytest.mark.needs_db
def test_replay_from_round_state_precedes_that_round():
    """Feature state for round N must stop before round N's first match."""
    from tennis.db.schema import connect
    from tennis.sim.draws import replay_from_round

    con = connect()
    try:
        tk = pd.read_sql(
            "SELECT tourney_key FROM tournaments WHERE name='Roland Garros' "
            "AND season=2024", con)
        if tk.empty:
            pytest.skip("Roland Garros 2024 not loaded")
        rep = replay_from_round(con, tk.iloc[0, 0], from_round=2)
    finally:
        con.close()
    g = rep["matches"]
    rounds = sorted(g["round_idx"].dropna().unique())
    r2_first = int(g[g["round_idx"] == rounds[2]]["seq"].min())
    assert rep["state_seq"] == r2_first
    # every pinned match happened strictly before that point
    earlier = g[g["round_idx"].isin(rounds[:2])]
    assert earlier["seq"].max() < rep["state_seq"]


@pytest.mark.needs_db
def test_tourney_xref_prefers_the_right_event_not_the_biggest():
    """Linking must not be won by whichever tournament has more matches.

    A first version ranked candidates by raw count of matches played between
    the draw's players, and mapped the Montreal draw to Roland Garros — the
    same top players appear at both, and RG simply has more matches. A date
    window plus Jaccard similarity fixes it.
    """
    from tennis.db.lock import exclusive_write
    from tennis.db.schema import connect
    from tennis.ingest.draws_api import resolve_tourney_key

    # This test writes to the real database, so it loses the write lock to the
    # multi-day odds backfill and fails with "database is locked" for as long
    # as that runs -- which is most of the time. Claim the database the same
    # way the nightly job does and the backfill yields at its next batch.
    con = connect()
    con.execute("PRAGMA busy_timeout=60000")
    try:
        with exclusive_write("test_tourney_xref"):
            canada = pd.read_sql(
                "SELECT tourney_key, tourney_date FROM tournaments "
                "WHERE season=2026 AND name LIKE '%Canada%'", con)
            if canada.empty:
                pytest.skip("Canada Masters 2026 not loaded")
            key = canada.iloc[0]["tourney_key"]
            players = pd.read_sql(
                "SELECT winner_id AS p FROM matches WHERE tourney_key=? "
                "UNION SELECT loser_id FROM matches WHERE tourney_key=?",
                con, params=(key, key))["p"].tolist()
            con.execute("DELETE FROM tourney_xref WHERE feed_id='test-xref'")
            con.commit()

            got = resolve_tourney_key("test-xref", "National Bank Open - Montreal",
                                      players, 2026, "2026-08-03", con)
            assert got == key, f"resolved to {got}, expected {key}"

            # second call must come from the cache, not a rescan
            cached = con.execute(
                "SELECT tourney_key FROM tourney_xref WHERE feed_id='test-xref'"
            ).fetchone()
            assert cached and cached[0] == key

            # an unrelated field must not resolve to anything
            assert resolve_tourney_key("test-xref-2", "Nowhere Open",
                                       ["NOPE1", "NOPE2"], 2026, "2026-08-03", con) is None
    finally:
        con.execute("DELETE FROM tourney_xref WHERE feed_id LIKE 'test-xref%'")
        con.commit()
        con.close()


@pytest.mark.needs_db
def test_winner_convention_counts_cannot_go_negative():
    """A pair that met twice must count once, as unknown."""
    from tennis.ingest.draws_api import verify_winner_convention

    sheet = pd.DataFrame([{
        "round_id": 4, "draw_pos": 1, "p1_name": "Jannik Sinner",
        "p2_name": "Carlos Alcaraz", "seed1": "1", "seed2": "2",
        "is_bye": False, "result": "6-4 6-4",
    }])
    # same pair recorded in both directions, as if they met twice
    ours = pd.DataFrame([{"winner_id": "S0AG", "loser_id": "A0E2"},
                         {"winner_id": "A0E2", "loser_id": "S0AG"}])
    chk = verify_winner_convention(sheet, ours)
    assert chk["unknown"] >= 0
    assert chk["confirmed"] + chk["contradicted"] + chk["unknown"] == chk["checked"]


# --------------------------------------------------------------------------
# progressive bracket
# --------------------------------------------------------------------------
class _FlatPredictor:
    """Prices every tie at 50/50, so bracket tests do not need trained models."""

    def predict_many(self, pairs, ctx):
        return pd.DataFrame({
            "p1_id": [a for a, _ in pairs],
            "p2_id": [b for _, b in pairs],
            "p1_win_prob": [0.5] * len(pairs),
            "pred_total_games": [22.0] * len(pairs),
            "pred_total_sets": [2.4] * len(pairs),
            "pred_spread": [0.0] * len(pairs),
        })


def test_bracket_prices_only_decided_ties():
    """Round 1 is priced; later rounds stay TBD until real results fill them."""
    from tennis.sim.bracket import bracket_state

    rounds = bracket_state(_draw(8), _FlatPredictor(), resolved=None)
    # Eight players play a QF, an SF and a final. This previously asserted
    # ["SF", "Final", "Champion"], which was the reached-stage vocabulary
    # applied to the round being played -- every label off by one step.
    assert [rd["round"] for rd in rounds] == ["QF", "SF", "F"]
    first = rounds[0]["ties"]
    assert all(t["state"] == "live" for t in first)
    assert all(t.get("p1_win_prob") is not None for t in first)
    # nothing beyond round 1 has known players yet
    for rd in rounds[1:]:
        assert all(t["state"] == "pending" for t in rd["ties"])
        assert all(t.get("p1_win_prob") is None for t in rd["ties"])


def test_bracket_advances_as_results_arrive():
    """Finishing round 1 makes round 2 priceable, and no further."""
    from tennis.sim.bracket import bracket_state

    d = _draw(8)
    # resolve all four opening ties
    resolved = {0: {0: "P0", 1: "P2", 2: "P4", 3: "P6"}}
    rounds = bracket_state(d, _FlatPredictor(), resolved=resolved)

    assert all(t["state"] == "played" for t in rounds[0]["ties"])
    # round 2 pairings are now determined, so they get priced
    assert all(t["state"] == "live" for t in rounds[1]["ties"])
    assert {t["p1"] for t in rounds[1]["ties"]} == {"P0", "P4"}
    # the final still has nobody in it
    assert all(t["state"] == "pending" for t in rounds[2]["ties"])


def test_bracket_partial_round_leaves_next_round_tbd():
    """One unfinished tie must leave the dependent slot TBD, not guessed."""
    from tennis.sim.bracket import bracket_state

    d = _draw(8)
    resolved = {0: {0: "P0", 1: "P2"}}       # ties 2 and 3 still outstanding
    rounds = bracket_state(d, _FlatPredictor(), resolved=resolved)

    r2 = rounds[1]["ties"]
    # first R2 tie has both feeders decided -> priced
    assert r2[0]["state"] == "live"
    # second has neither -> pending, with no invented opponent
    assert r2[1]["state"] == "pending"
    assert r2[1]["p1"] is None and r2[1]["p2"] is None


def test_bracket_frame_is_renderable():
    from tennis.sim.bracket import bracket_frame, bracket_state

    d = _draw(8)
    rounds = bracket_state(d, _FlatPredictor(), resolved={0: {0: "P0"}})
    df = bracket_frame(d, rounds)
    assert set(df["State"]) <= {"played", "live", "pending"}
    assert "TBD" in set(df["Player 1"]) | set(df["Player 2"])
    assert df[df["State"] == "live"]["P1 win %"].notna().all()


def test_played_ties_keep_their_prediction_and_grade_it():
    """A finished match still shows what the model said, and how wrong it was.

    `winner_prob` is the probability given to the player who actually won, so
    one number separates a coin-flip miss from a genuine upset.
    """
    from tennis.sim.bracket import bracket_state

    class _Lopsided:
        def predict_many(self, pairs, ctx):
            # first listed player is always the heavy favourite
            return pd.DataFrame({
                "p1_id": [a for a, _ in pairs],
                "p2_id": [b for _, b in pairs],
                "p1_win_prob": [0.8] * len(pairs),
                "pred_total_games": [22.0] * len(pairs),
                "pred_total_sets": [2.4] * len(pairs),
                "pred_spread": [3.0] * len(pairs),
            })

    d = _draw(8)
    # tie 0: favourite P0 wins (called).  tie 1: underdog P3 wins (upset).
    rounds = bracket_state(d, _Lopsided(), resolved={0: {0: "P0", 1: "P3"}})
    ties = {t["pair"]: t for t in rounds[0]["ties"]}

    called, upset = ties[0], ties[1]
    assert called["state"] == "played" and upset["state"] == "played"
    # prediction survives the result
    assert called["p1_win_prob"] == pytest.approx(0.8)
    assert upset["p1_win_prob"] == pytest.approx(0.8)
    # graded against who actually won
    assert called["winner_prob"] == pytest.approx(0.8) and called["correct"]
    assert upset["winner_prob"] == pytest.approx(0.2) and not upset["correct"]


def test_near_coin_flip_is_not_graded_as_an_upset():
    """A 49% miss and a 5% miss must be distinguishable, not both just 'wrong'."""
    from tennis.sim.bracket import bracket_state

    class _CoinFlip:
        def predict_many(self, pairs, ctx):
            return pd.DataFrame({
                "p1_id": [a for a, _ in pairs],
                "p2_id": [b for _, b in pairs],
                "p1_win_prob": [0.51] * len(pairs),
                "pred_total_games": [22.0] * len(pairs),
                "pred_total_sets": [2.4] * len(pairs),
                "pred_spread": [0.1] * len(pairs),
            })

    rounds = bracket_state(_draw(8), _CoinFlip(), resolved={0: {0: "P1"}})
    tie = rounds[0]["ties"][0]
    assert not tie["correct"]
    # wrong, but only just -- the caller can shade this differently
    assert 0.45 < tie["winner_prob"] < 0.5


# --------------------------------------------------------------------------
# withdrawals and replacement players
# --------------------------------------------------------------------------
def _sheet_with_withdrawal() -> dict:
    """Feed shape for a seed who pulled out after the draw was made.

    Position 4 still reads "Withdrawn Seed, bye", but round 2 position 2 shows
    the replacement playing. The replacement has no draw position of its own.
    """
    return {"data": {"singles": [
        {"roundId": 4, "draw": 1, "result": "6-4 6-4", "player1Id": 1, "player2Id": 2,
         "player1": {"id": 1, "name": "Alpha"}, "player2": {"id": 2, "name": "Bravo"}},
        {"roundId": 4, "draw": 2, "result": "6-2 6-2", "player1Id": 3, "player2Id": 4,
         "player1": {"id": 3, "name": "Charlie"}, "player2": {"id": 4, "name": "Delta"}},
        {"roundId": 4, "draw": 3, "result": "6-1 6-1", "player1Id": 5, "player2Id": 6,
         "player1": {"id": 5, "name": "Echo"}, "player2": {"id": 6, "name": "Foxtrot"}},
        {"roundId": 4, "draw": 4, "result": "bye", "seed1": "1",
         "player1Id": 7, "player2Id": 3700,
         "player1": {"id": 7, "name": "Withdrawn Seed"},
         "player2": {"id": 3700, "name": "Unknown Player"}},
        # the replacement: no position, listed as a lucky loser
        {"roundId": 4, "draw": None, "result": None, "seed1": "LL",
         "player1Id": 8, "player2Id": None,
         "player1": {"id": 8, "name": "Lucky Loser"}, "player2": None},
        # round 2: the feed's own later round names the replacement
        {"roundId": 5, "draw": 1, "result": "", "player1Id": 1, "player2Id": 3,
         "player1": {"id": 1, "name": "Alpha"}, "player2": {"id": 3, "name": "Charlie"}},
        {"roundId": 5, "draw": 2, "result": "", "player1Id": 5, "player2Id": 8,
         "player1": {"id": 5, "name": "Echo"}, "player2": {"id": 8, "name": "Lucky Loser"}},
    ]}}


def test_replacement_takes_the_withdrawn_players_slot():
    """The alternate must land in the withdrawn seed's position, not be dropped.

    Two wrong answers were shipped before this. Keeping the unplaced row gave a
    96-player event 65 first-round ties, padding the bracket from 128 slots to
    256 and letting the stray player walk a half-empty draw to the title.
    Dropping it kept the bracket valid but ran a player who had withdrawn and
    omitted one who was actually playing.
    """
    from tennis.ingest.draws_api import parse_draw_sheet

    sheet = parse_draw_sheet(_sheet_with_withdrawal())
    assert len(sheet) == 4, "must stay a power of two, no stray tie"
    slot4 = sheet[sheet["draw_pos"] == 4].iloc[0]
    assert slot4["p1_name"] == "Lucky Loser"
    assert slot4["is_bye"]                      # inherits the bye
    assert "Withdrawn Seed" not in set(sheet["p1_name"]) | set(sheet["p2_name"])
    # and the replacement appears exactly once
    everyone = list(sheet["p1_name"]) + [x for x in sheet["p2_name"] if isinstance(x, str)]
    assert everyone.count("Lucky Loser") == 1


@pytest.mark.needs_db
def test_reconciled_draw_is_a_clean_power_of_two():
    from tennis.ingest.draws_api import build_draw_from_sheet, parse_draw_sheet

    sheet = parse_draw_sheet(_sheet_with_withdrawal())
    draw, unresolved = build_draw_from_sheet(
        sheet, name="t", surface="Hard", level="masters")
    assert draw.size == 8                       # 4 ties, no padding
    assert draw.slots.count("__BYE__") == 1     # only the genuine bye


def test_unmatched_alternate_is_dropped_not_appended():
    """An alternate the later rounds never mention cannot be placed, so it goes.

    Appending it would add a 5th tie and pad the bracket to 16 slots.
    """
    from tennis.ingest.draws_api import parse_draw_sheet

    payload = _sheet_with_withdrawal()
    # strip the round-2 row that identifies where the alternate belongs
    payload["data"]["singles"] = [
        x for x in payload["data"]["singles"]
        if not (x.get("roundId") == 5 and x.get("draw") == 2)]
    sheet = parse_draw_sheet(payload)
    assert len(sheet) == 4
    assert "Lucky Loser" not in set(sheet["p1_name"])
    # the withdrawn seed stays, since nothing identified a replacement
    assert "Withdrawn Seed" in set(sheet["p1_name"])


# --------------------------------------------------------------------------
# ROI accounting
# --------------------------------------------------------------------------
def _book(prices, model_p, outcomes):
    """Minimal frame for roi_winner: p1 always the side the model likes."""
    return pd.DataFrame({
        "p1_price": prices,
        "p2_price": [1.01] * len(prices),   # never the value side
        "p_win": model_p,
        "y_win": outcomes,
        "tourney_date": [20200101 + i for i in range(len(prices))],
    })


def test_breakeven_hit_rate_is_mean_of_inverse_price():
    """Break-even must be mean(1/price), not 1/mean(price).

    The two agree only on a flat book. On a skewed one they diverge sharply,
    and using the wrong one makes a losing model look profitable.
    """
    from tennis.models.evaluate import roi_winner

    prices = [2.0, 2.0, 10.0, 10.0]
    d = _book(prices, [0.9] * 4, [1, 0, 1, 0])
    r = roi_winner(d, edge=0.0)
    assert r["bets"] == 4
    assert r["breakeven_hit_rate"] == pytest.approx(np.mean([1 / p for p in prices]), abs=1e-4)
    # the naive figure is 1/6 = 0.167 against a true 0.30 -- not interchangeable
    assert r["breakeven_hit_rate"] != pytest.approx(1 / np.mean(prices), abs=0.05)


def test_hitting_exactly_breakeven_returns_the_stake():
    """A book won at exactly its break-even rate must show ~0% ROI."""
    from tennis.models.evaluate import roi_winner

    # 4 bets at 4.0: break-even is 25%, so exactly one winner returns the stake
    d = _book([4.0] * 4, [0.9] * 4, [1, 0, 0, 0])
    r = roi_winner(d, edge=0.0)
    assert r["hit_rate"] == pytest.approx(r["breakeven_hit_rate"], abs=1e-6)
    assert r["roi_pct"] == pytest.approx(0.0, abs=1e-6)


def test_roi_by_season_partitions_the_bets():
    """Seasons must sum back to the pooled figure -- no dropped or doubled rows."""
    from tennis.models.evaluate import roi_by_season, roi_winner

    d = _book([3.0] * 6, [0.9] * 6, [1, 0, 1, 0, 0, 1])
    d["tourney_date"] = [20200101, 20200601, 20210101, 20210601, 20220101, 20220601]
    seasons = roi_by_season(d, edge=0.0)
    pooled = roi_winner(d, edge=0.0)
    assert [s["season"] for s in seasons] == [2020, 2021, 2022]
    assert sum(s["bets"] for s in seasons) == pooled["bets"]
    assert sum(s["profit"] for s in seasons) == pytest.approx(pooled["profit"], abs=1e-6)


# --------------------------------------------------------------------------
# Elo: surface prior and layoff decay
# --------------------------------------------------------------------------
def test_unplayed_surface_reads_as_the_overall_rating():
    """A surface never played must read at the player's overall level, not 1500.

    This was the defect the surface prior fixes: a 2300-rated player's first
    clay match rated them 1500, identical to a qualifier, and the surface
    probability came out worse than ignoring surface entirely.
    """
    from tennis.features.elo import BASE_RATING, EloBook

    book = EloBook()
    for _ in range(40):
        book.update("strong", "weak", "overall")
    overall = book.get("strong", "overall")
    assert overall > BASE_RATING + 100
    assert book.get("strong", "Clay") == pytest.approx(overall)


def test_surface_blend_moves_toward_the_surface_value_with_matches():
    """Weight on the surface book must rise monotonically with its match count."""
    from tennis.features.elo import SURF_BLEND_N, EloBook

    book = EloBook()
    for _ in range(60):
        book.update("p", "filler", "overall")     # strong overall
    seen = []
    for i in range(1, 120):
        book.update("loser", "p", "Clay")         # p loses on clay every time
        seen.append(book.get("p", "Clay"))
    # Losing only on clay must drag the clay rating below overall, and further
    # below as the blend hands more weight to the surface book.
    assert seen[-1] < seen[0] < book.get("p", "overall")
    n = book.n("p", "Clay")
    raw = book.ratings[EloBook._key("p", "Clay")]
    w = n / (n + SURF_BLEND_N)
    expect = w * raw + (1 - w) * book.get("p", "overall")
    assert book.get("p", "Clay") == pytest.approx(expect, abs=1e-9)


def test_layoff_decay_spares_a_normal_schedule():
    """Inside the grace window a rating must not move at all."""
    from tennis.features.elo import IDLE_GRACE_DAYS, EloBook

    book = EloBook()
    for _ in range(30):
        book.update("p", "q", "overall", day=0)
    book.touch("p", 0)
    book.touch("q", 0)
    held = book.get("p", "overall", day=0)

    assert book.get("p", "overall", day=int(IDLE_GRACE_DAYS)) == pytest.approx(held)
    # A year out, though, it must have given ground toward the baseline.
    stale = book.get("p", "overall", day=int(IDLE_GRACE_DAYS) + 365)
    assert stale < held


def test_surface_decay_follows_the_overall_clock():
    """A player active on hard courts must not decay on clay.

    Keyed to per-surface gaps instead, the eight-month wait between clay
    seasons reads as an injury -- which measured away the entire surface gain.
    """
    from tennis.features.elo import EloBook

    book = EloBook()
    for d in range(0, 400, 10):                 # busy, but only on hard
        book.update("p", "q", "overall", day=d)
        book.update("p", "q", "Hard", day=d)
        book.touch("p", d)
        book.touch("q", d)
    for _ in range(30):                          # a clay history, long ago
        book.update("p", "q", "Clay", day=0)

    active = book.get("p", "Clay", day=400)      # 400 days since last clay match
    book_idle = EloBook()
    book_idle.ratings = dict(book.ratings)
    book_idle.counts = dict(book.counts)
    book_idle.last_day = {"p": 0, "q": 0}        # same ratings, but idle throughout
    idle = book_idle.get("p", "Clay", day=400)
    assert idle < active


def test_season_boundary_splits_next_gen_from_the_openers():
    """The tour year is not the calendar year, and December straddles two of them.

    The Next Gen Finals *close* a season (18 Dec 2024, 22 Dec 2025) while Doha,
    Brisbane, Hong Kong and the United Cup *open* the next (26-31 Dec). The cut
    has to land between them. An earlier version cut at 1 December and filed
    both Next Gen editions into the season they actually end.
    """
    import importlib.util  # noqa: F401

    # app.py runs Streamlit calls at import, so pull the function out by source
    src = open("dashboard/app.py").read()
    start = src.index("SEASON_CUT = ")
    end = src.index("@st.cache_data", start)
    ns: dict = {}
    exec(src[start:end], ns)
    season_start, season_label = ns["season_start"], ns["season_label"]

    # season-ENDING events stay with the season they close
    assert season_start(20241218) == 20231225   # Next Gen Finals 2024
    assert season_start(20251222) == 20241225   # Next Gen ATP Finals 2025
    assert season_start(20211219) == 20201225   # Rio challenger
    assert season_start(20251125) == 20241225   # ATP Finals week

    # season-OPENING events start the next one
    assert season_start(20241229) == 20241225   # Brisbane / Hong Kong / United Cup
    assert season_start(20051226) == 20051225   # earliest opener in the data
    assert season_start(20260102) == 20251225   # United Cup 2026
    assert season_start(20260807) == 20251225   # mid-season

    assert season_start(20251225) == 20251225   # the cut itself is inclusive
    assert season_label(20251225) == 2026       # December 2025 opens season 2026

    # every season start must itself be a season start
    for d in (20200115, 20211231, 20230601, 20241218):
        assert season_start(season_start(d)) == season_start(d)


def _dash_helpers():
    """Pull the pure helpers out of app.py, which runs Streamlit at import."""
    src = open("dashboard/app.py").read()
    ns: dict = {"pd": pd}
    start = src.index("LEVEL_BAND = {")
    exec(src[start:src.index("def _refine_from_history")]
         .replace("@st.cache_data(show_spinner=False)\n", ""), ns)
    body = src.index("def _refine_from_history")
    exec(src[body:src.index("\n\ndef ", body)], ns)
    return ns


def test_history_never_moves_an_event_between_tiers():
    """A city can host several tiers; the lookup must stay inside the feed's.

    Hamburg runs an ATP 500, a retired Masters and a Challenger. Matching on
    the city alone returned whichever had the longest history, which relabelled
    the Hamburg Challenger as an ATP 500 and fed the model the wrong level.
    Buenos Aires and Barcelona carry 12-21 Challengers beside a main-tour week
    and failed the same way.
    """
    ns = _dash_helpers()
    hist = pd.DataFrame([
        # name, surface, level, indoor, n, last_held
        ("Hamburg", "Clay", "atp500", 0.0, 18, 20260518),
        ("Hamburg", "Clay", "masters", 0.0, 9, 20080511),
        ("Hamburg", "Carpet", "challenger", 1.0, 4, 20030127),
        ("Hamburg", "Hard", "challenger", 0.0, 4, 20240317),
        ("Hamburg", "Clay", "challenger", 0.0, 1, 20251020),
        ("Washington", "Hard", "atp500", 0.0, 11, 20260801),
    ], columns=["name", "surface", "level", "indoor", "n", "last_held"])
    ns["_history_context"] = lambda: hist
    refine = ns["_refine_from_history"]

    # the bug: no band, and the biggest history wins whatever tier it is
    assert refine("Hamburg Challenger", None)["level"] == "atp500"
    # scoped to the tier the feed reports, it stays a Challenger
    assert refine("Hamburg Challenger", "challenger")["level"] == "challenger"

    # the job the lookup exists for still works: the feed cannot split 250/500
    assert refine("Citi Open - Washington", "tour")["level"] == "atp500"

    # A decorated main-tour name finds nothing -- "Hamburg European Open"
    # reduces to "Hamburg European", which is no city we hold. That is a
    # pre-existing gap in the name matching, not a tier problem, and it fails
    # safe: with no history the feed's own level stands and the field stays
    # editable. Pinned so a future change to the strip list is a visible one.
    assert refine("Hamburg European Open", "tour") == {}

    # most recent edition wins, not the most frequent -- Hamburg's challenger
    # carpet and hard editions tie at 4, and carpet left the tour in 2009
    assert refine("Hamburg Challenger", "challenger")["surface"] == "Clay"

    # a tier we hold no history for yields nothing rather than a wrong guess
    assert refine("Hamburg Challenger", "slam") == {}


def test_live_list_orders_by_tier_then_size():
    """A Masters outranks a Challenger that happens to have more matches left."""
    ns = _dash_helpers()
    rank = ns["LEVEL_RANK"]
    assert rank["grand_slam"] < rank["masters"] < rank["atp500"]
    assert rank["atp500"] < rank["atp250"] < rank["challenger"]
    rows = [("Hamburg Challenger", "challenger", 29),
            ("National Bank Open", "masters", 8),
            ("Todi Challenger", "challenger", 27)]
    order = sorted(rows, key=lambda r: (rank.get(r[1], 99), -r[2]))
    assert [r[0] for r in order] == [
        "National Bank Open", "Hamburg Challenger", "Todi Challenger"]


def test_bracket_connectors_only_span_consecutive_rounds():
    """An elbow across a hidden round would claim a progression that skips it.

    The round filter is a free multiselect, so R64+R16 with R32 dropped is
    reachable in two clicks. Connectors are suppressed there; the quarter rules
    still apply, since those describe the draw rather than the path through it.
    """
    src = open("dashboard/app.py").read()
    assert 'shown == list(range(shown[0], shown[-1] + 1))' in src, \
        "contiguity check moved; update this test"

    order = ["R64", "R32", "R16", "QF", "SF", "Final"]

    def linked(show):
        shown = [i for i, r in enumerate(order) if r in show]
        return len(shown) < 2 or shown == list(range(shown[0], shown[-1] + 1))

    assert linked({"R64", "R32", "R16", "QF"})     # the default view
    assert linked({"R16", "QF", "SF", "Final"})
    assert linked({"QF"})                          # nothing to join
    assert not linked({"R64", "R16"})              # R32 dropped from the middle
    assert not linked({"R64", "QF", "Final"})


def test_bracket_ties_reserve_the_meta_row():
    """Every tie must be the same height or the connecting elbows drift.

    A bye carries no prediction to grade, so its meta row was omitted and those
    ties came out 24px shorter than the rest -- enough to pull the R64 elbows
    6px off their stubs. Byes now say so outright and the other states reserve
    the row, which is what makes the geometry exact.
    """
    src = open("dashboard/app.py").read()
    assert '<div class="meta muted">no match played</div>' in src
    assert src.count('<div class="meta empty"></div>') >= 2
    assert '.tie .meta.empty::before {content:"\\00a0";}' in src


# --------------------------------------------------------------------------
# checkbestodds scrape
# --------------------------------------------------------------------------
_CBO_ROWS = (
    '<tr> <td class="l2 match"> <span ts="1360060200" class="time hM">11:30</span> '
    '<a href="/tennis-odds/challenger-bergamo/marco-cecchinato-ivan-sergeyev-2013-02-05/42248">'
    ' Marco Cecchinato -  Ivan Sergeyev</a></td> '
    '<td class="r"> <b>3.47</b></td> <td class="r"> <b>26.00</b></td></tr>'
    '<tr> <td class="l2 match"> <span ts="1360065600" class="time hM">13:00</span> '
    '<a href="/tennis-odds/challenger-bergamo/amir-weintraub-niels-desein-2013-02-05/42225">'
    ' Amir Weintraub -  Niels Desein</a></td> '
    '<td class="r"> <b>1.50</b></td> <td class="r"> <b>3.25</b></td></tr>'
)


def test_cbo_parses_players_prices_and_date():
    from tennis.ingest.odds_cbo import parse_page

    df = parse_page(_CBO_ROWS, "historical-odds-challenger-bergamo")
    assert len(df) == 2
    r = df.iloc[1]
    assert (r.p1_name, r.p2_name) == ("Amir Weintraub", "Niels Desein")
    assert (r.p1_odds, r.p2_odds) == (1.50, 3.25)
    assert r.tourney_date == 20130205 and r.tier == "challenger"


def test_cbo_rejects_transposed_bookmaker_rows():
    """The listing's "best odds" takes the max per side across books.

    One bookmaker with its two sides swapped therefore poisons a column. The
    real case: Cecchinato-Sergeyev 2013-02-05 listed 3.47/26.00 because
    Titanbet alone had 1.01/26.00 while eleven other books said ~3.20/~1.29.
    That pair implies a total probability of 0.33 -- a standing 3x arbitrage,
    which does not exist. Believing it prices a 1.29 favourite at 26.00 and
    manufactures an ROI that was never available, so the overround filter is
    load-bearing rather than cosmetic.

    The floor is 0.96, not 1.0. Best odds is a maximum taken across books, so
    it legitimately lands below 1.0 -- that is a real cross-book arbitrage. The
    second row here (1.50/3.25, overround 0.974) is exactly that and must
    survive: cutting at 1.0 discards the best-priced rows, which is precisely
    the data any ROI estimate depends on.
    """
    from tennis.ingest.odds_cbo import OVR_MAX, OVR_MIN, parse_page, sane

    df = parse_page(_CBO_ROWS, "historical-odds-challenger-bergamo")
    assert 1 / 3.47 + 1 / 26.00 < 1.0          # the impossible one
    kept = sane(df)
    assert len(kept) == 1
    assert kept.iloc[0].p1_name == "Amir Weintraub"
    o = 1 / kept.iloc[0].p1_odds + 1 / kept.iloc[0].p2_odds
    assert OVR_MIN <= o <= OVR_MAX


# --------------------------------------------------------------------------
# live odds capture
# --------------------------------------------------------------------------
def _te_row(price_class_1: str, price_class_2: str) -> str:
    """One tennisexplorer match: two rows, prices on the first."""
    return (
        '<table class="result">'
        '<tr class="head flags"><td class="t-name">Hamburg challenger</td></tr>'
        '<tr class="one fRow bott"><td class="first time">14:30 Live streams 1xBet</td>'
        '<td class="t-name">Tseng C. (5)</td><td class="h2h">0</td>'
        f'<td class="{price_class_1}">1.74</td><td class="{price_class_2}">2.01</td></tr>'
        '<tr class="one"><td class="t-name">Sachko V.</td></tr>'
        '</table>')


def test_live_prices_read_in_both_upcoming_and_finished_states():
    """The winner's price cell is relabelled once a result exists.

    An upcoming match carries two `course` cells; after a winner is known the
    site relabels that side `coursew`. Treating `coursew` as "player one"
    therefore reads only *finished* matches -- backwards for a live collector,
    and the reason an earlier version captured two completed main-tour matches
    and none of the 39 upcoming Challengers on the same page.
    """
    from datetime import date

    from tennis.ingest.odds_live import parse

    for c1, c2 in [("course", "course"),      # upcoming
                   ("coursew", "course"),     # p1 won
                   ("course", "coursew")]:    # p2 won
        df = parse(_te_row(c1, c2), "atp", date(2026, 8, 10))
        assert len(df) == 1, (c1, c2)
        r = df.iloc[0]
        assert (r.p1_odds, r.p2_odds) == (1.74, 2.01), (c1, c2)
        assert (r.p1_name, r.p2_name) == ("Tseng C.", "Sachko V.")
        assert r.start_time == "14:30"        # ad copy stripped from the cell
        assert r.tournament == "Hamburg challenger"


def test_live_capture_skips_tiers_below_challenger():
    from datetime import date

    from tennis.ingest.odds_live import parse

    html = _te_row("course", "course").replace(
        "Hamburg challenger", "UTR Pro Tennis Series 3")
    assert parse(html, "atp", date(2026, 8, 10)).empty


def test_surname_key_joins_both_name_styles():
    """tennisexplorer writes 'Sinner J.'; our players table has 'Jannik Sinner'."""
    from tennis.ingest.odds_live import surname_key

    assert surname_key("Sinner J.") == surname_key("Jannik Sinner")
    assert surname_key("Van De Zandschulp B.") == "zandschulp|b"
    assert surname_key("Tseng C.") != surname_key("Tsitsipas C.")


# --------------------------------------------------------------------------
# tennisexplorer historical odds
# --------------------------------------------------------------------------
_TE_ROW = (
    '<div id="oddsMenu-3-data">'
    '<table><tr><th colspan="4">Asian Handicap -1.5 sets</th></tr>'
    '<tr class="one"><td class="first tl"><span class="t">10Bet</span></td>'
    '<td class="k1"><div class="odds-in">2.75</div></td>'
    '<td class="k2"><div class="odds-in">1.40</div></td></tr></table>'
    '<table><tr><th colspan="4">Asian Handicap -1.5 games</th></tr>'
    '<tr class="one"><td class="first tl"><span class="t">10Bet</span></td>'
    '<td class="k1"><div class="odds-in odown">1.95'
    '<div class="odds-change-div"><table>'
    '<tr><td>01.08. 22:54</td><td class="bold">1.95</td><td>-0.03</td></tr>'
    '<tr><td class="title" colspan="3">Opening odds</td></tr>'
    '<tr><td>01.08. 02:44</td><td class="bold">1.98</td><td> </td></tr>'
    '</table></div></div></td>'
    '<td class="k2"><div class="odds-in">1.75</div></td></tr></table>'
    '</div>')


def test_te_line_unit_separates_sets_from_games():
    """A -1.5 handicap exists in both sets and games at different prices.

    They are different markets. Keying quotes without `line_unit` collapses
    them and one silently overwrites the other -- measured on 12 real pages,
    that destroyed 3.4% of all quotes, and every lost row was a *games* row,
    which is the market this source was scraped for. The primary key in
    `db/schema.py` must include line_unit; this pins the parser side of it.
    """
    from tennis.ingest.odds_te_hist import parse_detail

    q = parse_detail(_TE_ROW, te_id=1)
    key_no_unit = {(r["book"], r["market"], r["line"], r["side"]) for r in q}
    key_with_unit = {(r["book"], r["market"], r["line"], r["line_unit"], r["side"])
                     for r in q}
    assert len(key_with_unit) == len(q)          # every quote survives
    assert len(key_no_unit) < len(key_with_unit)  # ...only because of the unit

    units = {r["line_unit"] for r in q}
    assert units == {"sets", "games"}
    sets_p1 = next(r for r in q if r["line_unit"] == "sets" and r["side"] == "p1")
    games_p1 = next(r for r in q if r["line_unit"] == "games" and r["side"] == "p1")
    assert sets_p1["price_close"] == 2.75
    assert games_p1["price_close"] == 1.95


def test_te_parses_opening_and_closing_prices():
    """The hover table holds the close on row 1 and the open on row 3."""
    from tennis.ingest.odds_te_hist import parse_detail

    q = parse_detail(_TE_ROW, te_id=1)
    moved = next(r for r in q if r["line_unit"] == "games" and r["side"] == "p1")
    assert moved["price_close"] == 1.95 and moved["closed_at"] == "01.08. 22:54"
    assert moved["price_open"] == 1.98 and moved["opened_at"] == "01.08. 02:44"
    # A cell with no hover table has a close and no history -- not a zero.
    flat = next(r for r in q if r["line_unit"] == "games" and r["side"] == "p2")
    assert flat["price_close"] == 1.75 and flat["price_open"] is None


# --------------------------------------------------------------------------
# devigging
# --------------------------------------------------------------------------
def test_devig_methods_return_valid_probabilities():
    import numpy as np

    from tennis.models.devig import power, proportional, shin

    prices = np.array([[1.20, 5.00], [1.90, 1.95], [1.05, 12.0], [2.50, 1.55]])
    for name, p in (("proportional", proportional(prices)),
                    ("shin", shin(prices)[0]), ("power", power(prices)[0])):
        assert np.allclose(p.sum(axis=1), 1.0), name
        assert (p > 0).all() and (p < 1).all(), name


def test_devig_is_a_no_op_on_a_zero_margin_book():
    """With no margin there is nothing to remove, whatever the method."""
    import numpy as np

    from tennis.models.devig import power, proportional, shin

    fair = np.array([[2.0, 2.0], [4.0, 4.0 / 3.0]])
    for p in (proportional(fair), shin(fair)[0], power(fair)[0]):
        assert np.allclose(p[:, 0], [0.5, 0.25], atol=1e-6)


def test_shin_moves_probability_toward_the_favourite():
    """The whole point: books load the margin onto the longshot.

    Proportional devigging assumes an even split and therefore leaves the
    favourite-longshot bias in place -- measured on our own data as backing
    every favourite returning -5.4% against a 6.7% vig while backing every
    underdog returned -11.5%. Shin shifts probability from the longshot to the
    favourite, which is the correction. Scored on 523,879 quotes it beat
    proportional in 16 of 17 books, the exception being an exchange with no
    bookmaker margin to misallocate.
    """
    import numpy as np

    from tennis.models.devig import proportional, shin

    prices = np.array([[1.20, 5.00], [1.05, 12.0], [2.50, 1.55]])
    pr, sh = proportional(prices)[:, 0], shin(prices)[0][:, 0]
    fav = pr > 0.5
    assert (sh[fav] > pr[fav]).all()       # favourites get more
    assert (sh[~fav] < pr[~fav]).all()     # longshots get less
    # and the insider fraction it implies should be small but non-zero
    z = shin(prices)[1]
    assert ((z > 0) & (z < 0.25)).all()


def test_dedupe_fixtures_keys_on_the_pairing_not_the_date():
    """One row per pairing, however many times the site re-lists the match.

    A postponed fixture reappears under a new play_date, and the earlier key of
    (play_date, p1, p2) let it through once per listing -- three rows for one
    match on the betting page, which is a tripled stake on a single result.
    The site's p1/p2 order is not stable between captures either, so a swapped
    re-listing has to collapse onto the same key.
    """
    import pandas as pd

    # app.py runs Streamlit calls at import, so pull the function out by source
    src = open("dashboard/app.py").read()
    # From _pair_key, which dedupe_fixtures now shares with drop_finished.
    start = src.index("def _pair_key")
    end = src.index("@st.cache_data", start)
    ns: dict = {"pd": pd}
    exec(src[start:end], ns)
    dedupe = ns["dedupe_fixtures"]

    df = pd.DataFrame({
        "p1_name": ["Blanchet U.", "Blanchet U.", "Sakamoto R.", "Kozlov S."],
        "p2_name": ["Sakamoto R.", "Sakamoto R.", "Blanchet U.", "Mayo A."],
        "play_date": [20260810, 20260812, 20260813, 20260812],
        "captured_at": ["2026-08-10T04:00", "2026-08-12T04:00",
                        "2026-08-13T04:00", "2026-08-12T04:00"],
        "p1_odds": [3.84, 3.84, 1.28, 2.44],
    })
    out = dedupe(df)
    assert len(out) == 2, "three listings of one match must collapse to one row"
    # The freshest capture survives, including the one with the names swapped.
    kept = out.set_index("p1_name").play_date.to_dict()
    assert kept == {"Sakamoto R.": 20260813, "Kozlov S.": 20260812}
    assert dedupe(df.iloc[:0]).empty


def test_draw_sheet_finds_the_seed_when_the_sentinel_holds_player1():
    """A bye row puts the sentinel on whichever side the real player is not.

    Cincinnati 2026 used both orientations: Zverev and Paul sat in player1 with
    "Unknown Player" below them, while Etcheverry and Vacherot sat in player2
    with the sentinel *above*. Reading p1 blindly kept the sentinel as the bye
    holder and dropped the real seed, so 16 of the 32 seeds in a 96-draw
    vanished from the bracket -- and the ones left behind showed as
    "Unknown Player, bye".
    """
    from tennis.ingest.draws_api import parse_draw_sheet

    payload = {"data": {"singles": [
        {"roundId": 4, "draw": 1, "result": "bye", "seed1": "1", "seed2": None,
         "player1Id": 100, "player2Id": 3700,
         "player1": {"id": 100, "name": "Top Seed", "seed": "1"},
         "player2": {"id": 3700, "name": "Unknown Player"}},
        # same bye, mirrored: the seed is in player2
        {"roundId": 4, "draw": 2, "result": "bye", "seed1": None, "seed2": "26",
         "player1Id": 3700, "player2Id": 200,
         "player1": {"id": 3700, "name": "Unknown Player"},
         "player2": {"id": 200, "name": "Mirrored Seed", "seed": "26"}},
    ], "doubles": [], "qualifying": []}}

    sheet = parse_draw_sheet(payload)
    assert int(sheet["is_bye"].sum()) == 2
    # Both seeds survive, and neither slot is held by the sentinel.
    assert list(sheet["p1_name"]) == ["Top Seed", "Mirrored Seed"]
    assert "Unknown Player" not in set(sheet["p1_name"])
    assert sheet["p2_name"].isna().all()
    # The seed travels with the player it belongs to, not with the slot.
    assert list(sheet["seed1"]) == ["1", "26"]


def test_bracket_labels_the_round_being_played_not_the_stage_reached():
    """A 128-draw opens with the R128, and the tour names it that way.

    `round_names` deliberately names the stage a player *reached* by winning
    round r, which is what `simulate` counts and what `_actual_progression`
    joins against. Using those labels on a bracket shifts every column by one:
    Cincinnati's opening round showed as "R64" and the real R16 was captioned
    "QF". The bracket needs the round itself.
    """
    from tennis.sim.bracket import playing_round_names, round_names

    assert playing_round_names(7) == ["R128", "R64", "R32", "R16", "QF", "SF", "F"]
    assert playing_round_names(6) == ["R64", "R32", "R16", "QF", "SF", "F"]
    assert playing_round_names(3) == ["QF", "SF", "F"]
    assert playing_round_names(1) == ["F"]
    # The two vocabularies must stay distinct and offset by exactly one step.
    assert round_names(7)[0] == "R64" and playing_round_names(7)[0] == "R128"
    assert playing_round_names(7)[1:-1] == round_names(7)[:-2]
    # Matches the `round` vocabulary the matches table uses.
    assert set(playing_round_names(7)) <= {"R128", "R64", "R32", "R16",
                                           "QF", "SF", "F"}


def test_single_instance_refuses_a_second_live_run(tmp_path, monkeypatch):
    """A second copy of the scraper must refuse to start, not silently double up.

    Every write the scraper makes is idempotent by primary key, so a duplicate
    run corrupts nothing and is therefore invisible -- two `--details` runs
    overlapped for nine hours before anyone noticed. What it does cost is a
    doubled request rate against a site we deliberately rate-limit, and a pile
    of re-fetched pages.
    """
    import json
    import os
    import subprocess

    from tennis.db import lock as lockmod

    monkeypatch.setattr(lockmod, "DATA", tmp_path)
    pf = tmp_path / ".running-job"

    # A live foreign pid holds the job: we must be refused.
    proc = subprocess.Popen(["sleep", "30"])
    try:
        pf.write_text(json.dumps({"pid": proc.pid, "name": "job",
                                  "since": "2026-08-19T13:07:22"}))
        with pytest.raises(lockmod.AlreadyRunning) as exc:
            with lockmod.single_instance("job"):
                pass
        assert str(proc.pid) in str(exc.value)
    finally:
        proc.kill()
        proc.wait()

    # The pid is dead now, so the job may take over -- a crash must not lock
    # the job out permanently.
    with lockmod.single_instance("job"):
        assert json.loads(pf.read_text())["pid"] == os.getpid()
    assert not pf.exists()          # cleaned up on the way out

    # A stale file for a dead pid is cleared rather than honoured.
    pf.write_text(json.dumps({"pid": proc.pid, "name": "job", "since": "x"}))
    with lockmod.single_instance("job"):
        pass
    assert not pf.exists()


def test_tourney_meta_trusts_a_challenger_name_over_the_history():
    """Level resolution needs both sources, and in this order.

    Events currently in play are often absent from `tournaments` entirely --
    that table is built from *results*, so a tournament being played right now
    has no row yet. A name carrying "challenger" states its own level and is
    trusted outright. Matching by name alone also mis-resolves: "Cincinnati
    WTA" normalises onto the ATP Cincinnati Masters row.
    """
    import sqlite3

    import pandas as pd

    src = open("dashboard/app.py").read()
    start = src.index("LEVEL_ORDER = {")
    end = src.index("# Short TTL", start)
    ns: dict = {"pd": pd}
    exec(src[start:end], ns)

    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE tournaments (name TEXT, level TEXT, "
                "surface TEXT, season INT)")
    con.executemany("INSERT INTO tournaments VALUES (?,?,?,?)", [
        ("Cincinnati Masters", "masters", "Hard", 2026),
        ("Winston Salem", "atp250", "Hard", 2025),
        ("Prague 2", "challenger", "Clay", 2024),
    ])
    out = ns["_tourney_meta"](pd.Series([
        "Cincinnati", "Winston Salem", "Prague 2 challenger",
        "Sion challenger",          # absent from history entirely
    ]), con).set_index("tournament")

    assert out.loc["Cincinnati", "level"] == "masters"
    assert out.loc["Winston Salem", "level"] == "atp250"
    # Present in history *and* named challenger -- surface still comes through.
    assert out.loc["Prague 2 challenger", "level"] == "challenger"
    assert out.loc["Prague 2 challenger", "surface"] == "Clay"
    # Absent from history: the name still settles the level.
    assert out.loc["Sion challenger", "level"] == "challenger"

    order = ns["LEVEL_ORDER"]
    assert order["grand_slam"] < order["masters"] < order["atp250"] < order["challenger"]


def test_no_undefined_names_anywhere():
    """Catch the bug class that silently broke the nightly job for nine days.

    A guard was added to `load_all` without importing it, so every run raised
    `NameError: name 'exclusive_write' is not defined` -- after the download
    step, so the logs looked busy and the failure was invisible until someone
    asked why the data was stale. Nothing in the suite could catch it: the
    function needs a database, and an import-time check does not evaluate
    function bodies.

    Only undefined names fail here. Unused imports are reported by pyflakes too
    but are cosmetic, and failing on them would make this test noise that gets
    disabled rather than a guard that gets trusted.
    """
    import subprocess
    import sys

    pyflakes = pytest.importorskip("pyflakes")     # noqa: F841
    proc = subprocess.run(
        [sys.executable, "-m", "pyflakes",
         "tennis", "dashboard", "scripts", "tests"],
        capture_output=True, text=True)
    bad = [ln for ln in proc.stdout.splitlines() if "undefined name" in ln]
    assert not bad, "undefined names:\n" + "\n".join(bad)


def test_drop_finished_judges_the_fixture_not_the_row():
    """A fixture seen finished once is finished, whatever other rows say.

    Rows captured before the `status` column existed carry NULL. Treating NULL
    as 'upcoming' row-by-row is what let played matches through: a finished
    fixture still has its older NULL-status captures sitting in the table, and
    those are precisely the rows a row-level filter keeps. Cincinnati's
    Tiafoe-Musetti stayed on the betting screen at its pre-match price fourteen
    hours after it ended.
    """
    import pandas as pd

    src = open("dashboard/app.py").read()
    start = src.index("def _pair_key")
    end = src.index("def dedupe_fixtures", start)
    ns: dict = {"pd": pd}
    exec(src[start:end], ns)
    drop_finished = ns["drop_finished"]

    df = pd.DataFrame({
        "p1_name": ["Tiafoe F.", "Musetti L.", "Fils A.", "Royer V."],
        "p2_name": ["Musetti L.", "Tiafoe F.", "Cobolli F.", "Martinez P."],
        # the older capture predates the column; the newer one knows
        "status": [None, "finished", "upcoming", None],
        "captured_at": ["2026-08-22T13:58", "2026-08-22T14:37",
                        "2026-08-22T14:37", "2026-08-22T13:58"],
    })
    out = drop_finished(df)
    # Both orientations of the finished fixture go, including the NULL row.
    assert "Tiafoe F." not in set(out.p1_name) | set(out.p2_name)
    # Genuinely upcoming fixtures survive, and so does an unknown-status one.
    assert set(out.p1_name) == {"Fils A.", "Royer V."}
    # No status column at all (an older database) must not blow up.
    assert len(drop_finished(df.drop(columns="status"))) == 4
