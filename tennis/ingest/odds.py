"""Load tennis-data.co.uk odds and join them to the match table.

Two things the source does *not* give us, both of which shape what the
backtest can claim:

1. Only match-winner (moneyline) prices exist -- there are no totals or
   handicap lines in any season. ROI against a real closing line is therefore
   possible for the winner model only; see `tennis.backtest` for how the other
   two targets are evaluated instead.
2. Coverage is ATP main tour only; challenger matches have no prices.

The join is by name, because the two sources share no match key. tennis-data
writes "Dimitrov G." while TennisMyLife writes "Grigor Dimitrov", so we key on
(season, last name token, first initial) for both players. tennis-data *does*
carry the true per-match date, so a successful join also upgrades the match's
weekly `tourney_date` stamp to an exact `match_date`.
"""
from __future__ import annotations

import logging
import re
import warnings

import numpy as np
import pandas as pd

from tennis.config import RAW_TD
from tennis.db.schema import connect
from tennis.ingest.load import norm_name

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# Bookmaker column pairs, newest naming first. Avg/Max are the consensus and
# best-of-market closing prices and are the ones we use for ROI.
BOOK_PAIRS = {
    "B365": ("B365W", "B365L"),
    "PS":   ("PSW", "PSL"),
    "Max":  ("MaxW", "MaxL"),
    "Avg":  ("AvgW", "AvgL"),
    "EX":   ("EXW", "EXL"),
    "LB":   ("LBW", "LBL"),
    "SJ":   ("SJW", "SJL"),
    "CB":   ("CBW", "CBL"),
    "GB":   ("GBW", "GBL"),
    "IW":   ("IWW", "IWL"),
    "SB":   ("SBW", "SBL"),
    "BFE":  ("BFEW", "BFEL"),
}

TD_ROUND_MAP = {
    "1st round": "R128/R64/R32", "2nd round": "R64/R32/R16",
    "3rd round": "R32/R16", "4th round": "R16",
    "quarterfinals": "QF", "semifinals": "SF", "the final": "F",
    "round robin": "RR",
}


def _key_from_td(name) -> str:
    """'Auger-Aliassime F.' -> 'aliassime|f'   'Del Potro J.M.' -> 'potro|j'"""
    n = norm_name(name)
    if not n:
        return ""
    toks = n.split()
    initials = [t for t in toks if len(t) == 1]
    surname = [t for t in toks if len(t) > 1]
    if not surname:
        return ""
    ini = initials[0][0] if initials else ""
    return f"{surname[-1]}|{ini}"


def _key_from_full(name) -> str:
    """'Grigor Dimitrov' -> 'dimitrov|g'   'Juan Martin del Potro' -> 'potro|j'"""
    n = norm_name(name)
    if not n:
        return ""
    toks = n.split()
    if len(toks) < 2:
        return f"{toks[0]}|" if toks else ""
    return f"{toks[-1]}|{toks[0][0]}"


def read_odds_files() -> pd.DataFrame:
    frames = []
    for path in sorted(RAW_TD.glob("*.xls*")):
        if path.name.startswith((".", "~")):
            continue  # editor lock/temp files
        try:
            df = pd.read_excel(path)
        except Exception as exc:  # noqa: BLE001
            log.warning("could not read %s: %s", path.name, exc)
            continue
        df["season"] = int(re.match(r"(\d{4})", path.name).group(1))
        frames.append(df)
    if not frames:
        raise RuntimeError(f"no odds workbooks in {RAW_TD}")
    out = pd.concat(frames, ignore_index=True)
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["w_key"] = out["Winner"].map(_key_from_td)
    out["l_key"] = out["Loser"].map(_key_from_td)
    return out


def join_odds(con) -> tuple[pd.DataFrame, dict]:
    td = read_odds_files()
    matches = pd.read_sql(
        "SELECT m.match_id, m.tourney_date, m.round, m.best_of, m.winner_id,"
        "       m.loser_id, m.is_challenger_flag, pw.name AS w_name, pl.name AS l_name "
        "FROM (SELECT m.*, t.is_challenger AS is_challenger_flag FROM matches m "
        "      JOIN tournaments t USING(tourney_key)) m "
        "JOIN players pw ON pw.player_id = m.winner_id "
        "JOIN players pl ON pl.player_id = m.loser_id",
        con,
    )
    main = matches[matches["is_challenger_flag"] == 0].copy()
    main["season"] = main["tourney_date"] // 10000
    main["w_key"] = main["w_name"].map(_key_from_full)
    main["l_key"] = main["l_name"].map(_key_from_full)
    main["td"] = pd.to_datetime(main["tourney_date"].astype(str), format="%Y%m%d")

    # Candidate join on (season, both player keys). Then disambiguate repeat
    # meetings inside a season by taking the pairing whose real date sits
    # closest to (and generally within ~2 weeks of) the tournament week stamp.
    cand = main.merge(
        td[["season", "w_key", "l_key", "Date", "Round", "Surface", "Best of",
            *[c for pair in BOOK_PAIRS.values() for c in pair if c in td.columns]]],
        on=["season", "w_key", "l_key"], how="inner", suffixes=("", "_td"),
    )
    cand["gap"] = (cand["Date"] - cand["td"]).dt.days
    cand = cand[cand["gap"].between(-3, 21)]
    cand = cand.sort_values("gap").drop_duplicates("match_id", keep="first")
    # One tennis-data row must not be reused for two different matches.
    cand["td_row"] = (
        cand["season"].astype(str) + cand["w_key"] + cand["l_key"]
        + cand["Date"].astype(str)
    )
    cand = cand.drop_duplicates("td_row", keep="first")

    rows = []
    for book, (wc, lc) in BOOK_PAIRS.items():
        if wc not in cand.columns:
            continue
        sub = cand[["match_id", wc, lc]].copy()
        # A few cells carry text placeholders instead of a price.
        sub[wc] = pd.to_numeric(sub[wc], errors="coerce")
        sub[lc] = pd.to_numeric(sub[lc], errors="coerce")
        sub = sub.dropna()
        sub = sub[(sub[wc] > 1.0) & (sub[lc] > 1.0)]
        rows.append(pd.DataFrame({
            "match_id": sub["match_id"], "book": book,
            "win_price": sub[wc], "lose_price": sub[lc],
        }))
    odds = pd.concat(rows, ignore_index=True).drop_duplicates(["match_id", "book"])

    # Scoped to the books this module owns. An unscoped `DELETE FROM odds`
    # silently destroys every other source's rows -- and it did: the
    # checkbestodds Challenger prices vanished on the next run of this
    # function, because `odds` is now shared with `odds_cbo` (CBO) and
    # `odds_live` (TE_CLOSE), neither of which existed when this was written.
    con.executemany("DELETE FROM odds WHERE book = ?",
                    [(b,) for b in BOOK_PAIRS])
    odds.to_sql("odds", con, if_exists="append", index=False)

    # Upgrade weekly stamps to true match dates where the join succeeded.
    exact = cand[["match_id", "Date"]].dropna()
    exact["match_date"] = exact["Date"].dt.strftime("%Y%m%d").astype(int)
    con.executemany(
        "UPDATE matches SET match_date=?, date_is_exact=1 WHERE match_id=?",
        list(zip(exact["match_date"].tolist(), exact["match_id"].tolist())),
    )
    con.commit()

    stats = {
        "td_rows": len(td),
        "main_tour_matches": len(main),
        "matched": int(cand["match_id"].nunique()),
        "match_rate": round(cand["match_id"].nunique() / max(len(main), 1), 4),
        "odds_rows": len(odds),
        "books": odds["book"].value_counts().to_dict(),
        "dates_made_exact": len(exact),
    }
    return cand, stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    con = connect()
    _, st = join_odds(con)
    for k, v in st.items():
        print(f"{k}: {v}")
    con.close()
