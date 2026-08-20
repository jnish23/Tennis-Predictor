"""Can the model win against a softer book, or against an opening price?

Every model-vs-market number in this project so far has been against Pinnacle's
*closing* line, which is the hardest target available: the sharpest book at its
sharpest moment. Losing to it says little about whether the model could beat a
recreational book, or beat a line before the money arrives. This tests both.

Two comparisons, each with a trap the naive version walks into:

* **Across books.** Books cover different matches -- B365 prices 59,893 of our
  matches and Max only 39,624 -- so a plain per-book ROI confounds the price
  with the sample. Everything here is computed on the **common subset** priced
  by every book being compared, so the only thing varying is the price.

* **Open against close.** tennisexplorer carries both. The opening price is the
  softer number, and if the model has any edge at all this is where it should
  appear. Same control: only matches carrying both prices count, so the two are
  scored on identical matches.

**On reading the results.** Fourteen books times four thresholds is 56 tests,
and at that count something clears zero by luck alone. Confidence intervals are
bootstrapped **by match** (a match priced by 14 books is one observation, not
14), and the summary reports how many books beat zero against how many would be
expected to by chance. Treat a lone positive cell as noise unless it survives
that.

Run:  python scripts/backtest_books.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tennis.db.schema import connect  # noqa: E402
from tennis.models.devig import METHODS  # noqa: E402
from tennis.models.evaluate import load_backtest, roi_winner  # noqa: E402

RNG = np.random.default_rng(0)
EDGES = (0.0, 0.03, 0.05, 0.10)
MIN_BETS = 200


def boot_roi(d: pd.DataFrame, n: int = 400) -> tuple[float, float]:
    """95% CI for flat-stake ROI, resampling matches rather than bets."""
    if d.empty:
        return (np.nan, np.nan)
    profit = np.where(d.won, d.price - 1, -1.0)
    idx = RNG.integers(0, len(profit), (n, len(profit)))
    boot = profit[idx].mean(axis=1) * 100
    return (round(float(np.percentile(boot, 2.5)), 2),
            round(float(np.percentile(boot, 97.5)), 2))


def bets_at(df: pd.DataFrame, edge: float) -> pd.DataFrame:
    """The bets an edge threshold selects, with price and outcome attached."""
    d = df.dropna(subset=["p1_price", "p2_price", "p_win"]).copy()
    if d.empty:
        return d
    ev1 = d.p_win * d.p1_price - 1
    ev2 = (1 - d.p_win) * d.p2_price - 1
    side1 = ev1 >= ev2
    d["ev"] = np.maximum(ev1, ev2)
    d["price"] = np.where(side1, d.p1_price, d.p2_price)
    d["won"] = np.where(side1, d.y_win == 1, d.y_win == 0)
    return d[d.ev > edge]


def book_frame(bt: pd.DataFrame, book: str, con) -> pd.DataFrame:
    """Backtest rows with one book's closing prices oriented onto p1/p2."""
    o = pd.read_sql("SELECT match_id, win_price, lose_price FROM odds "
                    "WHERE book=?", con, params=(book,))
    out = bt.merge(o, on="match_id", how="inner")
    out["p1_price"] = np.where(out.y_win == 1, out.win_price, out.lose_price)
    out["p2_price"] = np.where(out.y_win == 1, out.lose_price, out.win_price)
    return out.dropna(subset=["p1_price", "p2_price"])


def te_open_close(bt: pd.DataFrame, con) -> pd.DataFrame:
    """Opening and closing moneyline from tennisexplorer, oriented onto p1/p2.

    `side` is in tennisexplorer's own k1/k2 frame and k1 is the winner in 99.8%
    of resolved matches, so the price must be moved into our p1/p2 frame the
    same way `attach_odds` does -- via who actually won. The price itself was
    set pre-match, so using the result to *locate* it leaks nothing.
    """
    q = pd.read_sql("""
        SELECT m.match_id, q.book,
               MAX(CASE WHEN q.side='p1' THEN q.price_open  END) k1_open,
               MAX(CASE WHEN q.side='p2' THEN q.price_open  END) k2_open,
               MAX(CASE WHEN q.side='p1' THEN q.price_close END) k1_close,
               MAX(CASE WHEN q.side='p2' THEN q.price_close END) k2_close,
               m.p1_is_winner
        FROM odds_quotes q JOIN te_matches m USING(te_id)
        WHERE q.market='h2h' AND m.match_id IS NOT NULL
              AND m.p1_is_winner IS NOT NULL
        GROUP BY m.match_id, q.book""", con)
    q = q.dropna(subset=["k1_open", "k2_open", "k1_close", "k2_close"])
    q = q[(q.k1_open > 1.001) & (q.k2_open > 1.001)
          & (q.k1_close > 1.001) & (q.k2_close > 1.001)]
    # k1/k2 -> winner/loser
    w = q.p1_is_winner == 1
    for tag in ("open", "close"):
        q[f"win_{tag}"] = np.where(w, q[f"k1_{tag}"], q[f"k2_{tag}"])
        q[f"lose_{tag}"] = np.where(w, q[f"k2_{tag}"], q[f"k1_{tag}"])
    out = bt.merge(q, on="match_id", how="inner")
    # winner/loser -> our p1/p2
    for tag in ("open", "close"):
        out[f"p1_{tag}"] = np.where(out.y_win == 1, out[f"win_{tag}"],
                                    out[f"lose_{tag}"])
        out[f"p2_{tag}"] = np.where(out.y_win == 1, out[f"lose_{tag}"],
                                    out[f"win_{tag}"])
    return out


def summarise(rows: list[dict], label: str) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        print(f"  no {label} rows with enough bets")
        return
    print(df.to_string(index=False))
    beat = df[(df.ci_lo > 0)]
    print(f"  {len(beat)} of {len(df)} cells clear zero with the whole "
          f"interval; at 95% confidence {0.025 * len(df):.1f} would be "
          f"expected by chance alone.")


def main() -> None:
    con = connect()
    con.execute("PRAGMA busy_timeout=60000")
    bt = load_backtest()[["match_id", "tourney_date", "is_challenger",
                          "y_win", "p_win"]]

    books = pd.read_sql("SELECT book, COUNT(*) n FROM odds GROUP BY book "
                        "HAVING n > 10000 ORDER BY n DESC", con).book.tolist()
    frames = {b: book_frame(bt, b, con) for b in books}
    frames = {b: f for b, f in frames.items() if len(f) > 5000}

    # Common sample: only matches every book priced, so price is the only
    # thing that differs between the rows below.
    common = set.intersection(*(set(f.match_id) for f in frames.values()))
    print(f"=== every book, common sample of {len(common):,} matches ===")
    print("(vig differs between books; the model is identical)\n")

    rows = []
    for b, f in frames.items():
        d = f[f.match_id.isin(common)]
        ovr = (1 / d.p1_price + 1 / d.p2_price).median()
        for e in EDGES:
            sel = bets_at(d, e)
            if len(sel) < MIN_BETS:
                continue
            lo, hi = boot_roi(sel)
            rows.append({"book": b, "vig_%": round((1 - 1 / ovr) * 100, 2),
                         "edge": e, "bets": len(sel),
                         "roi_%": round(float(np.where(sel.won, sel.price - 1,
                                                       -1.0).mean() * 100), 2),
                         "ci_lo": lo, "ci_hi": hi,
                         "avg_price": round(float(sel.price.mean()), 2)})
    summarise(rows, "book")

    # ---------------------------------------------------------------- open
    print("\n=== opening price against closing price ===")
    oc = te_open_close(bt, con)
    if oc.empty:
        print("  no tennisexplorer rows carry both an opening and closing price")
        con.close()
        return
    print(f"{len(oc):,} book-match rows over {oc.match_id.nunique():,} matches, "
          f"{oc.book.nunique()} books\n")

    rows = []
    for tag in ("open", "close"):
        d = oc.rename(columns={f"p1_{tag}": "p1_price", f"p2_{tag}": "p2_price"})
        ovr = (1 / d.p1_price + 1 / d.p2_price).median()
        for e in EDGES:
            sel = bets_at(d, e)
            if len(sel) < MIN_BETS:
                continue
            lo, hi = boot_roi(sel)
            rows.append({"price": tag, "vig_%": round((1 - 1 / ovr) * 100, 2),
                         "edge": e, "bets": len(sel),
                         "roi_%": round(float(np.where(sel.won, sel.price - 1,
                                                       -1.0).mean() * 100), 2),
                         "ci_lo": lo, "ci_hi": hi})
    summarise(rows, "open/close")

    # ------------------------------------------------- best price anywhere
    # What a line shopper actually gets: the best number on offer across every
    # book that priced the match. This is the only configuration in this file
    # that comes out positive, so it carries the controls with it.
    print("\n=== best price across books ===")
    rows = []
    for tag in ("open", "close"):
        g = (oc.groupby("match_id")
               .agg(p1_price=(f"p1_{tag}", "max"), p2_price=(f"p2_{tag}", "max"),
                    p_win=("p_win", "first"), y_win=("y_win", "first"))
               .reset_index())
        ovr = (1 / g.p1_price + 1 / g.p2_price).median()
        for e in EDGES:
            sel = bets_at(g, e)
            if len(sel) < MIN_BETS:
                continue
            lo, hi = boot_roi(sel)
            rows.append({"price": f"best {tag}",
                         "vig_%": round((1 - 1 / ovr) * 100, 2), "edge": e,
                         "bets": len(sel),
                         "roi_%": round(float(np.where(sel.won, sel.price - 1,
                                                       -1.0).mean() * 100), 2),
                         "ci_lo": lo, "ci_hi": hi})
    summarise(rows, "best-price")

    # Control: the same prices with the model switched off. If backing anything
    # at the best available number already returns what the model returns, the
    # "edge" is just line shopping and the model is contributing nothing.
    g = (oc.groupby("match_id")
           .agg(p1_price=("p1_open", "max"), p2_price=("p2_open", "max"),
                p1_med=("p1_open", "median"), p2_med=("p2_open", "median"),
                p_win=("p_win", "first"), y_win=("y_win", "first")).reset_index())
    print("\n  control -- best opening price, no model:")
    for name, pick in (("always p1", np.ones(len(g), bool)),
                       ("always p2", np.zeros(len(g), bool)),
                       ("always the favourite", g.p1_price.values < g.p2_price.values)):
        price = np.where(pick, g.p1_price, g.p2_price)
        won = np.where(pick, g.y_win == 1, g.y_win == 0)
        print(f"    {name:22} ROI {np.where(won, price - 1, -1.0).mean() * 100:+6.2f}%")

    # Control: taking the maximum across books systematically selects whichever
    # book is most wrong, and a badly stale or mistyped price is both the most
    # attractive and the least bettable. Strip the extreme outliers and see
    # whether anything survives.
    sel = bets_at(g, 0.05)
    side1 = sel.price.values == sel.p1_price.values
    med = np.where(side1, sel.p1_med, sel.p2_med)
    prem = (sel.price.values - med) / med * 100
    keep = sel[prem <= 15]
    lo, hi = boot_roi(keep)
    print(f"\n  control -- {(prem > 15).mean() * 100:.0f}% of selected bets take a "
          f"price >15% above the median book (likely stale or mistyped, and the "
          f"least likely to be accepted).\n    excluding them: {len(keep):,} bets, "
          f"ROI {np.where(keep.won, keep.price - 1, -1.0).mean() * 100:+.2f}% "
          f"CI [{lo:+.2f}, {hi:+.2f}]")

    # Did the line move toward us? That is CLV, measured on the same rows.
    d = oc.copy()
    ev1 = d.p_win * d.p1_open - 1
    ev2 = (1 - d.p_win) * d.p2_open - 1
    side1 = ev1 >= ev2
    took_open = np.where(side1, d.p1_open, d.p2_open)
    same_close = np.where(side1, d.p1_close, d.p2_close)
    sel = np.maximum(ev1, ev2) > 0.05
    if sel.sum():
        moved = (took_open[sel] - same_close[sel]) / same_close[sel] * 100
        print(f"\n  On the {int(sel.sum()):,} bets clearing a 5% edge at the "
              f"open, the price we took beat the close by "
              f"{moved.mean():+.2f}% on average "
              f"({(moved > 0).mean() * 100:.1f}% of them closed shorter).")
        print("  A line moving toward the bet is the standard sign the model "
              "sees something real; it is not the same as profit, which the "
              "table above measures directly.")
    con.close()


if __name__ == "__main__":
    main()
