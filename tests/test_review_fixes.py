"""Regressions for the repo-review findings.

Each test pins a failure that was real: the id that moved when upstream
renumbered, the WTA ids read as floats, the resolver that missed family-name-
first feed names, the reload that could leave `matches` empty. All of them run
without the production database.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _raw(rows: list[dict]) -> pd.DataFrame:
    base = {
        "tourney_id": "2026-0001", "tourney_name": "Testville",
        "surface": "Hard", "tourney_level": "250", "indoor": "O",
        "draw_size": 32, "best_of": 3, "round": "R32", "match_num": 1,
        "winner_id": "A001", "loser_id": "B002", "score": "6-4 6-4",
        "tourney_date": 20260105, "winner_rank": 10, "loser_rank": 20,
        "is_challenger_file": 0, "tour": "atp", "source_file": "2026.csv",
        "source_priority": 0,
        "winner_name": "Win Ner", "loser_name": "Lo Ser",
        "winner_hand": "R", "loser_hand": "L", "winner_ht": 185, "loser_ht": 180,
        "winner_ioc": "USA", "loser_ioc": "ESP",
    }
    return pd.DataFrame([{**base, **r} for r in rows])


# --------------------------------------------------------------------------
# ids and ingestion
# --------------------------------------------------------------------------
def test_match_id_survives_match_num_renumbering():
    """The same match in the ongoing file and the yearly file is one match."""
    from tennis.ingest.load import clean

    df = clean(_raw([
        {"match_num": 2, "source_file": "ongoing_tourneys.csv", "source_priority": 1},
        {"match_num": 8, "source_file": "2026.csv", "source_priority": 0},
    ]))
    assert len(df) == 1
    # the settled (yearly) copy wins the tie
    assert df["source_file"].iloc[0] == "2026.csv"
    assert "-2-" not in df["match_id"].iloc[0] and "-8-" not in df["match_id"].iloc[0]


def test_match_id_is_independent_of_match_num():
    from tennis.ingest.load import clean

    a = clean(_raw([{"match_num": 2}]))["match_id"].iloc[0]
    b = clean(_raw([{"match_num": 99}]))["match_id"].iloc[0]
    assert a == b


def test_round_robin_is_ordered_before_the_knockout_rounds():
    """RR sits last in ROUND_ORDER; it must still be *played* first."""
    from tennis.ingest.load import clean

    df = clean(_raw([
        {"round": "F", "winner_id": "A001", "loser_id": "B002", "match_num": 1},
        {"round": "SF", "winner_id": "A001", "loser_id": "C003", "match_num": 2},
        {"round": "RR", "winner_id": "B002", "loser_id": "C003", "match_num": 3},
    ]))
    assert list(df.sort_values("seq")["round"]) == ["RR", "SF", "F"]


def test_ids_stay_strings_when_a_column_has_blanks(tmp_path, monkeypatch):
    """A blank id makes pandas read the column as float: '216016' -> '216016.0',
    which split 280 WTA players into two identities."""
    from tennis.ingest import load

    d = tmp_path / "raw"
    d.mkdir()
    (d / "2026_wta.csv").write_text(
        "tourney_id,winner_id,loser_id,tourney_date\n"
        "2026-1,216016,300001,20260105\n"
        "2026-1,,300002,20260105\n")
    monkeypatch.setattr(load, "RAW_TML", d)
    monkeypatch.setattr(load, "RAW_ONGOING_ARCHIVE", tmp_path / "none")
    raw = load.read_raw()
    assert "216016" in set(raw["winner_id"].dropna())
    assert not any(str(x).endswith(".0") for x in raw["winner_id"].dropna())
    assert (raw["tour"] == "wta").all()


def test_clean_strips_float_suffix_from_ids():
    from tennis.ingest.load import clean

    df = clean(_raw([{"winner_id": "216016.0", "loser_id": "300001"}]))
    assert df["winner_id"].iloc[0] == "216016"


def test_archived_ongoing_tournament_survives_leaving_the_ongoing_file(tmp_path, monkeypatch):
    """A tournament that dropped out of the ongoing file before the yearly one
    caught up used to vanish from the database."""
    from tennis.ingest import load
    from tennis.ingest.download import archive_ongoing

    raw_dir, arch = tmp_path / "raw", tmp_path / "arch"
    raw_dir.mkdir()
    monkeypatch.setattr(load, "RAW_TML", raw_dir)
    monkeypatch.setattr(load, "RAW_ONGOING_ARCHIVE", arch)
    import tennis.ingest.download as dl
    monkeypatch.setattr(dl, "RAW_ONGOING_ARCHIVE", arch)

    csv = ("tourney_id,winner_id,loser_id,tourney_date\n"
           "2026-A,W1,L1,20260105\n2026-B,W2,L2,20260105\n")
    archive_ongoing("ongoing_tourneys.csv", csv.encode())
    # next pull: tournament A is gone from the live file
    (raw_dir / "ongoing_tourneys.csv").write_text(
        "tourney_id,winner_id,loser_id,tourney_date\n2026-B,W2,L2,20260105\n")
    raw = load.read_raw()
    assert set(raw["tourney_id"]) == {"2026-A", "2026-B"}
    # and once the yearly file carries A, the stale snapshot is not merged in
    (raw_dir / "2026.csv").write_text(
        "tourney_id,winner_id,loser_id,tourney_date\n2026-A,W1,L1,20260105\n")
    raw = load.read_raw()
    assert (raw["tourney_id"] == "2026-A").sum() == 1


def test_reload_is_atomic_and_remaps_legacy_ids(tmp_path, monkeypatch):
    """A failure partway through the reload must leave the old data intact; and
    odds rows keyed by the old `match_num` ids must follow the match."""
    from tennis.db.schema import init_db
    from tennis.ingest import load

    rows = [{"round": r, "match_num": i, "winner_id": f"W{i}", "loser_id": f"L{i}"}
            for i, r in enumerate(["R32", "R16", "QF", "SF", "F"])]
    monkeypatch.setattr(load, "read_raw", lambda: _raw(rows))
    db = tmp_path / "t.db"
    con = init_db(db)
    # an odds row under the pre-fix id format
    con.execute("INSERT INTO odds VALUES(?,?,?,?)",
                ("2026-0001|testville|0|0-3-W3-L3", "B365", 1.5, 2.5))
    con.commit()
    con.close()

    stats = load.load_all(db)
    assert stats["matches"] == 5 and stats["ids_remapped"] == 1
    con = init_db(db)
    mid = con.execute("SELECT match_id FROM odds").fetchone()[0]
    assert con.execute("SELECT COUNT(*) FROM matches WHERE match_id=?", (mid,)).fetchone()[0] == 1
    con.close()

    def boom(*a, **k):
        if a[1] == "matches":
            raise RuntimeError("disk full")
        return real(*a, **k)
    real = load._insert
    monkeypatch.setattr(load, "_insert", boom)
    with pytest.raises(RuntimeError):
        load.load_all(db)
    con = init_db(db)
    assert con.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 5
    assert con.execute("SELECT COUNT(*) FROM players").fetchone()[0] > 0
    con.close()


def test_reload_refuses_a_collapsed_source(tmp_path, monkeypatch):
    from tennis.db.schema import init_db
    from tennis.ingest import load

    many = [{"round": "R32", "match_num": i, "winner_id": f"W{i}", "loser_id": f"L{i}"}
            for i in range(20)]
    db = tmp_path / "t.db"
    init_db(db).close()
    monkeypatch.setattr(load, "read_raw", lambda: _raw(many))
    load.load_all(db)
    monkeypatch.setattr(load, "read_raw", lambda: _raw(many[:3]))
    with pytest.raises(RuntimeError, match="refusing"):
        load.load_all(db)


# --------------------------------------------------------------------------
# incremental feature updates
# --------------------------------------------------------------------------
def _folded(ids, dates):
    return pd.DataFrame({"match_id": ids, "tourney_date": dates})


def _current(ids, dates):
    return pd.DataFrame({"match_id": ids, "tourney_date": dates,
                         "seq": range(len(ids))})


def test_incremental_ignores_renumbered_seq():
    """New rows are found by match_id, so a renumbered `seq` cannot hide them."""
    from tennis.features.pipeline import _full_rebuild_reason

    prior = _folded(["a", "b"], [20260105, 20260105])
    # every seq shifted, one genuinely new match at the end
    cur = _current(["x0", "a", "b", "c"], [20260105] * 3 + [20260112])
    assert _full_rebuild_reason(prior, cur.drop(0)) is None


def test_incremental_falls_back_when_a_folded_match_disappears():
    from tennis.features.pipeline import _full_rebuild_reason

    prior = _folded(["a", "b"], [20260105, 20260105])
    assert "no longer exist" in _full_rebuild_reason(prior, _current(["a", "c"], [20260105, 20260112]))


def test_incremental_falls_back_for_a_match_older_than_state():
    from tennis.features.pipeline import _full_rebuild_reason

    prior = _folded(["a"], [20260112])
    assert "pre-date" in _full_rebuild_reason(prior, _current(["a", "z"], [20260112, 20260105]))


def test_incremental_run_matches_full_rebuild_across_a_reload(tmp_path, monkeypatch):
    """The scenario the old test could not see: between two runs the database is
    reloaded, so every seq changes and an unrelated tour's rows slot in ahead."""
    from tennis.db.schema import init_db
    from tennis.features import pipeline
    from tennis.ingest import load

    def rows(n):
        out = []
        for i in range(n):
            out.append({"tourney_id": f"2026-{i // 4:03d}", "tourney_name": f"T{i // 4}",
                        "tourney_date": int((pd.Timestamp("2026-01-05")
                                            + pd.Timedelta(weeks=i // 4)).strftime("%Y%m%d")),
                        "round": "R32",
                        "match_num": i, "winner_id": f"P{i % 5}", "loser_id": f"P{(i + 2) % 5}",
                        "w_svpt": 60, "l_svpt": 60, "w_1stIn": 40, "l_1stIn": 40,
                        "w_1stWon": 25, "l_1stWon": 25, "w_2ndWon": 12, "l_2ndWon": 12,
                        "w_ace": 3, "l_ace": 2, "w_df": 1, "l_df": 1, "w_SvGms": 10,
                        "l_SvGms": 9, "w_bpSaved": 3, "l_bpSaved": 2, "w_bpFaced": 4,
                        "l_bpFaced": 5})
        return rows_to_raw(out)

    def rows_to_raw(out):
        return _raw(out)

    monkeypatch.setattr(pipeline, "ARTIFACTS", tmp_path)
    db = tmp_path / "t.db"
    monkeypatch.setattr(pipeline, "connect", lambda: init_db(db))
    init_db(db).close()

    monkeypatch.setattr(load, "read_raw", lambda: rows(16))
    load.load_all(db)
    pipeline.run(full=True, tour="atp")

    # more matches arrive AND a WTA tour appears, renumbering every seq
    more = rows(24)
    wta = _raw([{"tourney_id": "2026-900", "tourney_name": "W", "tourney_date": 20260105,
                 "tour": "wta", "winner_id": "9001", "loser_id": "9002", "source_file": "2026_wta.csv"}])
    monkeypatch.setattr(load, "read_raw", lambda: pd.concat([wta, more], ignore_index=True))
    load.load_all(db)
    inc = pipeline.run(full=False, tour="atp")
    assert inc["total_rows"] == 24

    inc_df = pd.read_parquet(pipeline.features_path("atp")).sort_values("match_id").reset_index(drop=True)
    pipeline.run(full=True, tour="atp")
    full_df = pd.read_parquet(pipeline.features_path("atp")).sort_values("match_id").reset_index(drop=True)
    assert list(inc_df["match_id"]) == list(full_df["match_id"])
    for c in full_df.columns:
        if pd.api.types.is_numeric_dtype(full_df[c]):
            assert np.allclose(full_df[c].to_numpy(float), inc_df[c].to_numpy(float),
                               equal_nan=True), f"{c} diverged"


# --------------------------------------------------------------------------
# feed name resolution
# --------------------------------------------------------------------------
@pytest.fixture()
def player_db(tmp_path):
    from tennis.db.schema import init_db

    con = init_db(tmp_path / "p.db")
    rows = [
        ("Y09V", "atp", "Yunchaokete Bu", "CHN", 20261001),
        ("Z371", "atp", "Zhizhen Zhang", "CHN", 20261001),
        ("S0RE", "atp", "Juncheng Shang", "CHN", 20261001),
        ("A111", "atp", "Anna Lee", "USA", 20200101),     # same key as a WTA player
        ("W222", "wta", "Alice Lee", "KOR", 20261001),
        ("M001", "atp", "Adam Martin", "FRA", 20200101),
        ("M002", "atp", "Alex Martin", "USA", 20261001),
    ]
    from tennis.ingest.load import norm_name
    con.executemany(
        "INSERT INTO players(player_id,tour,name,name_norm,ioc,last_seen) VALUES(?,?,?,?,?,?)",
        [(i, t, n, norm_name(n), c, d) for i, t, n, c, d in rows])
    con.commit()
    yield con
    con.close()


def test_resolver_handles_family_name_first(player_db):
    """'Bu Yunchaokete' (feed) is 'Yunchaokete Bu' (ours): the Shanghai bug."""
    from tennis.ingest.draws_api import resolve_players

    r = resolve_players(["Bu Yunchaokete", "Yunchaokete Bu", "Zhizhen Zhang",
                         "Zhang Zhizhen", "Shang Juncheng", "Nobody Atall"],
                        player_db)
    assert r["Bu Yunchaokete"] == r["Yunchaokete Bu"] == "Y09V"
    assert r["Zhizhen Zhang"] == r["Zhang Zhizhen"] == "Z371"
    assert r["Shang Juncheng"] == "S0RE"
    assert r["Nobody Atall"] is None


def test_resolver_is_scoped_to_the_tour(player_db):
    from tennis.ingest.draws_api import resolve_players

    assert resolve_players(["A. Lee"], player_db, tour="atp")["A. Lee"] == "A111"
    assert resolve_players(["A. Lee"], player_db, tour="wta")["A. Lee"] == "W222"


def test_resolver_uses_country_to_break_ties(player_db):
    from tennis.ingest.draws_api import resolve_players

    # Martin A.: two ATP players share the key; the country settles it even
    # though the other played more recently.
    r = resolve_players(["Martin A."], player_db,
                        meta={"Martin A.": {"feed_id": 5, "ioc": "FRA"}})
    assert r["Martin A."] == "M001"
    # the country made it unambiguous, so it is recorded against the feed's id
    assert player_db.execute(
        "SELECT player_id FROM feed_player_xref WHERE feed_id='5'").fetchone()[0] == "M001"
    # without a country the tie is a guess, and a guess is never recorded
    resolve_players(["Martin A."], player_db, meta={"Martin A.": {"feed_id": 6}})
    assert player_db.execute(
        "SELECT COUNT(*) FROM feed_player_xref WHERE feed_id='6'").fetchone()[0] == 0


def test_resolver_learns_feed_ids(player_db):
    from tennis.ingest.draws_api import resolve_players

    resolve_players(["Bu Yunchaokete"], player_db,
                    meta={"Bu Yunchaokete": {"feed_id": 61667, "ioc": "CHN"}})
    # a later spelling the name matcher could never get still resolves by id
    r = resolve_players(["B. Yunchaokete-Variant"], player_db,
                        meta={"B. Yunchaokete-Variant": {"feed_id": 61667}})
    assert r["B. Yunchaokete-Variant"] == "Y09V"


def test_unresolved_played_result_keeps_the_tie_pinnable(player_db):
    from tennis.ingest.draws_api import played_results, unknown_id

    sheet = pd.DataFrame([{
        "p1_name": "Zhizhen Zhang", "p2_name": "Unlisted Person",
        "p1_id": 1, "p2_id": 2, "p1_ioc": "CHN", "p2_ioc": "ARG",
        "is_bye": False, "result": "6-1 6-1"}])
    res = played_results(sheet, player_db)
    assert len(res) == 1
    assert res.iloc[0]["winner_id"] == "Z371"
    assert res.iloc[0]["loser_id"] == unknown_id("Unlisted Person")


# --------------------------------------------------------------------------
# prediction context
# --------------------------------------------------------------------------
def _bare_predictor(tour):
    from tennis.models.predict import Predictor

    p = object.__new__(Predictor)
    p.tour = tour
    return p


def test_context_defers_to_the_predictors_tour():
    from tennis.models.predict import MatchContext

    assert _bare_predictor("wta").resolve(MatchContext()).tour == "wta"


def test_context_for_another_tour_is_an_error():
    from tennis.models.predict import MatchContext

    with pytest.raises(ValueError):
        _bare_predictor("atp").resolve(MatchContext(tour="wta"))


def test_default_date_is_this_weeks_monday_not_a_frozen_one():
    from datetime import date
    from tennis.models.predict import MatchContext, event_week_start

    assert event_week_start(date(2026, 10, 11)) == 20261005     # a Sunday
    assert event_week_start(date(2026, 10, 5)) == 20261005      # the Monday itself
    got = _bare_predictor("atp").resolve(MatchContext()).tourney_date
    assert got == event_week_start() and got != 20260803


def test_matrix_cache_distinguishes_round_and_tour_and_is_bounded():
    from tennis.models import predict

    class P(predict.Predictor):
        def __init__(self, tour):
            self.tour = tour
            self._matrix_cache = predict.OrderedDict()
            self.calls = 0

        def predict_many(self, pairs, ctx, **kw):
            self.calls += 1
            return pd.DataFrame({"p1_win_prob": [0.5 + ctx.round_idx / 100] * len(pairs)})

    p = P("atp")
    ids = ["a", "b", "c"]
    m5 = p.win_prob_matrix(ids, predict.MatchContext(round_idx=5.0))
    m6 = p.win_prob_matrix(ids, predict.MatchContext(round_idx=6.0))
    assert p.calls == 2 and not np.allclose(m5, m6)
    for i in range(predict._MATRIX_CACHE_MAX + 5):
        p.win_prob_matrix(ids, predict.MatchContext(round_idx=float(i + 10)))
    assert len(p._matrix_cache) <= predict._MATRIX_CACHE_MAX


# --------------------------------------------------------------------------
# features and parsing
# --------------------------------------------------------------------------
def test_match_tiebreak_is_one_game_not_a_seventeen_game_set():
    from tennis.ingest.parse import parse_score

    p = parse_score("6-4 5-7 [10-7]", 3)
    assert (p.w_sets, p.l_sets) == (2, 1)
    assert p.total_games == 6 + 5 + 1 + 4 + 7
    assert p.usable_for_totals


def test_best_of_five_with_two_sets_is_not_a_complete_match():
    from tennis.ingest.parse import parse_score

    assert not parse_score("6-3 6-4", 5).usable_for_totals
    assert parse_score("6-3 6-4 6-2", 5).usable_for_totals
    assert parse_score("6-3 6-4", 3).usable_for_totals
    # three sets won under a declared best-of-3 is a mislabel, still complete
    assert parse_score("6-4 6-4 6-4", 3).usable_for_totals


def _mrow(**kw):
    base = dict(match_id="m1", seq=1, tourney_date=20260105, winner_id="W", loser_id="L",
                surface="Hard", status="completed", has_stats=1, totals_usable=1,
                total_games=19, total_sets=2, game_margin=5, l_sets=0, best_of=3,
                tiebreaks=0, minutes=90.0, w_svpt=60, l_svpt=60, w_1stIn=40, l_1stIn=40,
                w_1stWon=30, l_1stWon=28, w_2ndWon=10, l_2ndWon=9, w_ace=3, l_ace=2,
                w_df=1, l_df=1, w_SvGms=10, l_SvGms=9, w_bpSaved=3, l_bpSaved=2,
                w_bpFaced=4, l_bpFaced=5)
    base.update(kw)
    return base


def test_missing_break_point_counts_do_not_become_extreme_rates():
    from tennis.features.build import FeatureEngine

    eng = FeatureEngine()
    eng.update(_mrow(w_bpFaced=np.nan, w_bpSaved=np.nan, l_bpFaced=np.nan, l_bpSaved=np.nan))
    for pid in ("W", "L"):
        assert len(eng.players[pid].hold) == 0 and len(eng.players[pid].brk) == 0


def test_opponent_elo_is_read_before_the_match_updates_it():
    from tennis.features.build import FeatureEngine

    eng = FeatureEngine()
    eng.update(_mrow())                       # W: ~1500 -> up, L: down
    w_after = eng.elo.get("W")
    eng.update(_mrow(match_id="m2", seq=2, winner_id="X", loser_id="W", tourney_date=20260106))
    # X's recorded opponent strength is W as W stood BEFORE losing to X
    assert eng.players["X"].opp_elo[-1] == pytest.approx(w_after)


def test_surface_rating_is_seeded_from_the_pre_match_overall():
    from tennis.features.build import FeatureEngine
    from tennis.features.elo import expected, k_factor

    eng = FeatureEngine()
    for i in range(5):
        eng.update(_mrow(match_id=f"h{i}", seq=i, surface="Clay"))
    w0, l0 = eng.elo.get("W", "overall"), eng.elo.get("L", "overall")
    eng.update(_mrow(match_id="g", seq=9, surface="Grass", tourney_date=20260110))
    # First grass update starts from the overall rating as it stood going in,
    # not from the value this very match had just raised it to.
    want = w0 + k_factor(0) * (1 - expected(w0, l0))
    assert eng.elo.ratings[eng.elo._key("W", "Grass")] == pytest.approx(want, rel=1e-4)


def test_layoff_decay_reaches_surfaces_the_player_did_not_just_play():
    from tennis.features.elo import BASE_RATING, EloBook

    b = EloBook()
    for _ in range(30):
        b.update("P", "Q", "Clay", day=1000)
        b.update("P", "Q", "overall", day=1000)
        b.touch("P", 1000)
        b.touch("Q", 1000)
    clay_before = b.ratings[b._key("P", "Clay")]
    assert clay_before > BASE_RATING + 50
    # two years away, then a grass match: clay must carry the decay it was owed
    b.advance("P", 1000 + 730)
    b.touch("P", 1000 + 730)
    clay_after = b.ratings[b._key("P", "Clay")]
    assert BASE_RATING < clay_after < clay_before
    assert b.get("P", "Clay", 1000 + 730) == pytest.approx(b.get("P", "Clay", 1000 + 731), abs=1.0)


def test_reference_line_uses_only_strictly_earlier_seasons():
    from tennis.models.evaluate import _reference_line

    n = 120
    df = pd.DataFrame({"surface": "Hard", "is_challenger": 0,
                       "tourney_date": [20200301] * n + [20210301] * n,
                       "y": [10.0] * n + [30.0] * n})
    line = _reference_line(df, "y")
    assert line.iloc[:n].isna().all()               # no earlier season: no line
    assert (line.iloc[n:] == 10.0).all()            # never sees its own season


def test_verdict_follows_the_numbers():
    from tennis.models.evaluate import _verdict

    rep = {"market_subset": {"n": 10, "model": {"log_loss": 0.60},
                             "market_closing": {"log_loss": 0.62}},
           "winner": {"log_loss": 0.60}, "winner_elo_baseline": {"log_loss": 0.65},
           "roi_winner": {"edge_0.03": {"bets": 50, "roi_pct": 4.0}}}
    good = _verdict(rep, 4.0, True)
    assert "loses money" not in good and "beats" in good and "+4.00%" in good
    rep["roi_winner"]["edge_0.03"]["roi_pct"] = -3.0
    bad = _verdict(rep, -3.0, False)
    assert "loses money" in bad and "less accurate" in bad


# --------------------------------------------------------------------------
# locking and downloads
# --------------------------------------------------------------------------
def test_write_claim_is_exclusive_and_only_released_by_its_owner(tmp_path, monkeypatch):
    from tennis.db import lock

    monkeypatch.setattr(lock, "CLAIM", tmp_path / "claim")
    with lock.exclusive_write("a"):
        with pytest.raises(lock.LockTimeout):
            with lock.exclusive_write("b", wait=0.3, poll=0.1):
                pass
        assert lock.CLAIM.exists()          # the failed waiter did not free it
    assert not lock.CLAIM.exists()


def test_a_live_claim_is_not_expired_by_age(tmp_path, monkeypatch):
    import json, os, time
    from tennis.db import lock

    monkeypatch.setattr(lock, "CLAIM", tmp_path / "claim")
    lock.CLAIM.write_text(json.dumps({"pid": os.getpid(), "at": time.time() - 7200}))
    assert lock._active_claim() is not None       # two hours old, owner alive


def test_download_does_not_retry_a_404(monkeypatch):
    from tennis.ingest import download

    calls = []

    class R:
        status_code = 404
        content = b""
        def raise_for_status(self):
            raise AssertionError("should have stopped before this")

    monkeypatch.setattr(download.requests, "get", lambda *a, **k: calls.append(1) or R())
    monkeypatch.setattr(download.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="404"):
        download._get("http://x/y")
    assert len(calls) == 1


def test_cache_writes_are_atomic(tmp_path):
    from tennis.ingest.download import _write_atomic

    dest = tmp_path / "f.csv"
    dest.write_bytes(b"old")
    _write_atomic(dest, b"new")
    assert dest.read_bytes() == b"new" and [p.name for p in tmp_path.iterdir()] == ["f.csv"]


# --------------------------------------------------------------------------
# duplicate upstream player ids
# --------------------------------------------------------------------------
def _dup_raw(tour, same_event=False, ioc2="USA"):
    n = [{"tour": tour, "winner_id": "100", "loser_id": f"x{i}", "loser_name": f"Opp {i}", "winner_name": "Ann Lee",
          "winner_ioc": "USA", "tourney_id": f"t{i}", "match_num": i} for i in range(5)]
    new = {"tour": tour, "winner_id": "900", "loser_id": "y1", "loser_name": "Opp Y", "winner_name": "Ann Lee",
           "winner_ioc": ioc2, "tourney_id": "t0" if same_event else "t9", "match_num": 50}
    return _raw(n + [new])


def test_wta_duplicate_ids_are_merged_into_the_busier_one():
    from tennis.ingest.load import clean, find_aliases

    raw = _dup_raw("wta")
    applied, cand = find_aliases(raw)
    assert applied.iloc[0]["alias_id"] == "900" and applied.iloc[0]["canonical_id"] == "100"
    df = clean(raw)
    assert set(df["winner_id"]) == {"100"}


def test_atp_duplicates_are_reported_not_merged():
    from tennis.ingest.load import clean, find_aliases

    raw = _dup_raw("atp")
    applied, cand = find_aliases(raw)
    assert applied.empty and len(cand) == 1
    assert "900" in set(clean(raw)["winner_id"])


def test_same_name_in_one_event_or_other_country_is_not_a_duplicate():
    from tennis.ingest.load import find_aliases

    assert find_aliases(_dup_raw("wta", same_event=True))[0].empty
    assert find_aliases(_dup_raw("wta", ioc2="KOR"))[0].empty


def test_merged_ids_pull_legacy_match_ids_along(tmp_path, monkeypatch):
    from tennis.db.schema import init_db
    from tennis.ingest import load

    monkeypatch.setattr(load, "ARTIFACTS", tmp_path)
    raw = _dup_raw("wta")
    monkeypatch.setattr(load, "read_raw", lambda: raw)
    db = tmp_path / "t.db"
    con = init_db(db)
    # an odds row keyed on the match as it was before the merge
    con.execute("INSERT INTO odds VALUES(?,?,?,?)",
                ("t9|testville|0|0-R32-900-y1", "B365", 1.5, 2.5))
    con.commit(); con.close()
    stats = load.load_all(db)
    con = init_db(db)
    assert stats["aliases_applied"] == 1
    mid = con.execute("SELECT match_id FROM odds").fetchone()[0]
    assert "-100-y1" in mid
    assert con.execute("SELECT COUNT(*) FROM matches WHERE match_id=?", (mid,)).fetchone()[0] == 1
    assert con.execute("SELECT canonical_id FROM player_alias").fetchone()[0] == "100"
    con.close()


def test_a_single_mislabelled_row_does_not_create_a_duplicate():
    from tennis.ingest.load import find_aliases

    raw = _dup_raw("wta")
    # id 900 is really "Bea Other" except for one row carrying Ann Lee's name
    raw["winner_name"] = raw["winner_name"].where(raw["winner_id"] != "900", "Bea Other")
    raw.loc[raw.index[-1], "winner_name"] = "Ann Lee"
    extra = raw.iloc[[-1]].copy(); extra["winner_name"] = "Bea Other"; extra["tourney_id"] = "t8"
    raw = pd.concat([raw, extra, extra.assign(tourney_id="t7")], ignore_index=True)
    assert find_aliases(raw)[0].empty
