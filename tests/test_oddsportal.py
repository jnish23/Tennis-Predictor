"""OddsPortal set-price parsing and the join back to our matches."""
from __future__ import annotations

import json

import pytest

from tennis.db.schema import init_db
from tennis.ingest import odds_oddsportal as op

# Shapes copied from a real OddsHarvester run (Shanghai 2024, 1st-set market).
REC_PRICED = {
    "match_date": "2024-10-13 08:40:00 UTC", "season": 2024,
    "match_link": "https://www.oddsportal.com/tennis/h2h/djokovic-novak-AZg49Et9/"
                  "sinner-jannik-6HdC3z4H/#6NAvawNp",
    "home_team": "Sinner J.", "away_team": "Djokovic N.",
    "league_name": "ATP Shanghai 2024", "home_score": 2, "away_score": 0,
    "partial_results": "(7:6 4 , 6:3)",
    "match_winner_market": [{"player_1": "1.50", "player_2": "2.63",
                             "bookmaker_name": "bet365.us", "period": "FirstSet",
                             "submarket_name": "Home/Away"}],
}
# The CSV flavour stringifies the list; and some matches carry no prices at all.
REC_CSV_STYLE = dict(REC_PRICED, match_link=REC_PRICED["match_link"].replace(
    "6NAvawNp", "XXXX1111"),
    match_winner_market=str(REC_PRICED["match_winner_market"]))
REC_EMPTY = dict(REC_PRICED, match_link=REC_PRICED["match_link"].replace(
    "6NAvawNp", "EMPTY222"), match_winner_market=[])


def test_parse_partial_handles_tiebreak_on_either_side():
    assert op.parse_partial("(7:6 4 , 1:6, 4:6)") == [(7, 6), (1, 6), (4, 6)]
    assert op.parse_partial("(6 5 :7, 5:7)") == [(6, 7), (5, 7)]
    assert op.parse_partial(None) == []


def test_parse_records_prices_and_ids():
    ms, qs = op.parse_records([REC_PRICED, REC_CSV_STYLE, REC_EMPTY])
    assert [m["op_id"] for m in ms] == ["6NAvawNp", "XXXX1111", "EMPTY222"]
    assert ms[0]["play_date"] == 20241013 and ms[0]["home_sets"] == 2
    assert len(qs) == 2                     # the unpriced match yields no quote
    assert qs[0] == {"op_id": "6NAvawNp", "book": "bet365.us", "period": 1,
                     "home_price": 1.5, "away_price": 2.63}


def test_ingest_is_idempotent_and_resolution_pins_orientation(tmp_path, monkeypatch):
    monkeypatch.setattr(op, "init_db", lambda: init_db(tmp_path / "t.db"))
    con = init_db(tmp_path / "t.db")
    con.execute("INSERT INTO tournaments(tourney_key, name, tourney_date) "
                "VALUES('k','Shanghai',20241006)")
    con.executemany("INSERT INTO players(player_id, name) VALUES(?,?)",
                    [("a", "Jannik Sinner"), ("b", "Novak Djokovic")])
    con.execute("INSERT INTO matches(match_id, tourney_key, tour, winner_id, loser_id,"
                " tourney_date, score) VALUES('m1','k','atp','a','b',20241006,"
                "'7-6(4) 6-3')")
    con.commit()
    f = tmp_path / "period1.json"
    f.write_text(json.dumps([REC_PRICED]))
    for _ in range(2):
        r = op.ingest_file(f, con)
    assert con.execute("SELECT COUNT(*) FROM set_quotes").fetchone()[0] == 1
    assert r["matches_with_prices"] == 1
    res = op.resolve_matches(con)
    assert res["resolved"] == 1 and res["set1_agree_pct"] == 100.0
    row = con.execute("SELECT match_id, home_is_winner, set1_agree FROM op_matches").fetchone()
    assert row == ("m1", 1, 1)


def test_set1_disagreement_is_flagged(tmp_path):
    con = init_db(tmp_path / "t.db")
    con.execute("INSERT INTO tournaments(tourney_key, name, tourney_date) "
                "VALUES('k','Shanghai',20241006)")
    con.executemany("INSERT INTO players(player_id, name) VALUES(?,?)",
                    [("a", "Jannik Sinner"), ("b", "Novak Djokovic")])
    # our record says the winner dropped set 1 -- theirs says home (the winner) took it
    con.execute("INSERT INTO matches(match_id, tourney_key, tour, winner_id, loser_id,"
                " tourney_date, score) VALUES('m1','k','atp','a','b',20241006,"
                "'3-6 6-3 6-4')")
    con.commit()
    ms, _ = op.parse_records([REC_PRICED])
    con.execute("INSERT INTO op_matches(op_id,play_date,home_name,away_name,partial) "
                "VALUES(?,?,?,?,?)", (ms[0]["op_id"], ms[0]["play_date"],
                                      ms[0]["home_name"], ms[0]["away_name"],
                                      ms[0]["partial"]))
    op.resolve_matches(con)
    assert con.execute("SELECT set1_agree FROM op_matches").fetchone()[0] == 0
