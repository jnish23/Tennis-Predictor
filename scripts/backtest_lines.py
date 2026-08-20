"""Betting backtest on the games totals and handicap ladders.

The moneyline offers one price per match. A ladder offers a dozen or more, and
that changes the statistics in a way worth being explicit about:

* **Rungs on the same match are not independent bets.** They settle on one
  scoreline, so a match whose total lands high wins every over on the ladder at
  once. Resampling individual bets would treat those as separate evidence and
  understate the error badly, so every confidence interval here is bootstrapped
  **by match**, not by bet.
* **A real bettor takes one rung, not all of them.** The default strategy backs
  at most the single best-EV rung per match. "Bet every positive-EV rung" is
  reported alongside it because it is what a naive loop would do, and the gap
  between the two is worth seeing.

Model probabilities come from the same empirical residual machinery as
`totals_vs_market.py` -- conditioned on best-of and predicted value, fitted only
on pre-2023 seasons. Market prices are Shin-devigged. The handicap ladder is
re-oriented before use; see that script's docstring for the two traps involved.

Run:  python scripts/backtest_lines.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tennis.db.schema import connect  # noqa: E402
from tennis.models.evaluate import load_backtest  # noqa: E402
from totals_vs_market import (EmpiricalResiduals, FIT_BEFORE,  # noqa: E402
                              load_market)

RNG = np.random.default_rng(0)
EPS = 1e-6


def build(market: str, pred: str, actual: str, con) -> pd.DataFrame:
    """Match-line rows with a model probability, a price, and an outcome."""
    bt = load_backtest()[["match_id", "tourney_date", "is_challenger", "y_win",
                          pred, actual]]
    tm = pd.read_sql("SELECT te_id, match_id FROM te_matches "
                     "WHERE match_id IS NOT NULL", con)
    mb = pd.read_sql("SELECT match_id, best_of, totals_usable FROM matches", con)
    base = bt.merge(tm, on="match_id").merge(mb, on="match_id")
    base = base[base.totals_usable == 1]
    base["season"] = base.tourney_date // 10000

    emp = EmpiricalResiduals(base[base.season < FIT_BEFORE], pred, actual)
    d = base.merge(load_market(market, con), on="te_id").dropna(subset=[pred, actual])
    d = d[d.season >= FIT_BEFORE].copy()

    if market == "totals":
        d["line_ours"] = d.line
        d["p_mkt_ours"] = d.p_mkt
    else:
        ours = d.y_win.to_numpy() == 1
        d["line_ours"] = np.where(ours, d.line, -d.line)
        d["p_mkt_ours"] = np.where(ours, d.p_mkt, 1 - d.p_mkt)

    d["p_model"] = emp.p_over(d.best_of.to_numpy(), d[pred].to_numpy(),
                              d.line_ours.to_numpy())
    d = d[d[actual] != d.line_ours]                     # drop pushes
    d["won_over"] = (d[actual] > d.line_ours).astype(int)
    # Recover a price from the devigged probability plus the observed overround,
    # so both sides carry the book's actual margin rather than a fair price.
    ovr = d.overround.clip(lower=1.0)
    d["price_over"] = 1.0 / (d.p_mkt_ours * ovr).clip(EPS, 0.999)
    d["price_under"] = 1.0 / ((1 - d.p_mkt_ours) * ovr).clip(EPS, 0.999)
    return d


def settle(d: pd.DataFrame, edge: float, *, one_per_match: bool = True) -> dict:
    ev_over = d.p_model * d.price_over - 1
    ev_under = (1 - d.p_model) * d.price_under - 1
    take_over = ev_over >= ev_under
    ev = np.maximum(ev_over, ev_under)
    price = np.where(take_over, d.price_over, d.price_under)
    won = np.where(take_over, d.won_over == 1, d.won_over == 0)

    g = d.assign(ev=ev, price=price, won=won)
    g = g[g.ev > edge]
    if one_per_match:
        # One rung per match: the ladder is a menu, not a dozen separate bets.
        g = g.sort_values("ev", ascending=False).drop_duplicates("match_id")
    if len(g) < 50:
        return {"edge": edge, "bets": len(g)}

    profit = np.where(g.won, g.price - 1, -1.0)
    # Bootstrap over MATCHES so correlated rungs move together.
    ids = g.match_id.to_numpy()
    uniq, inv = np.unique(ids, return_inverse=True)
    order = np.argsort(inv)
    starts = np.searchsorted(inv[order], np.arange(len(uniq)))
    groups = np.split(profit[order], starts[1:])
    idx = RNG.integers(0, len(groups), (2000, len(groups)))
    boot = np.array([np.concatenate([groups[i] for i in row]).mean()
                     for row in idx[:400]]) * 100
    return {
        "edge": edge, "bets": int(len(g)), "matches": int(len(uniq)),
        "roi_pct": round(float(profit.mean() * 100), 2),
        "ci_lo": round(float(np.percentile(boot, 2.5)), 2),
        "ci_hi": round(float(np.percentile(boot, 97.5)), 2),
        "hit": round(float(g.won.mean()), 4),
        "breakeven": round(float(np.mean(1 / g.price)), 4),
        "avg_price": round(float(g.price.mean()), 3),
    }


def main() -> None:
    con = connect()
    for label, market, pred, actual in (
            ("TOTALS (games)", "totals", "pred_total", "y_total"),
            ("SPREAD (games)", "handicap", "pred_spread", "y_spread")):
        d = build(market, pred, actual, con)
        vig = (1 - 1 / d.overround.median()) * 100
        print(f"\n===== {label} =====")
        print(f"{len(d):,} match-line rows over {d.match_id.nunique():,} matches, "
              f"median vig {vig:.1f}%")

        # Floor: what indiscriminate betting returns, so a negative ROI can be read.
        for side, price, win in (("back every over", d.price_over, d.won_over == 1),
                                 ("back every under", d.price_under, d.won_over == 0)):
            r = np.where(win, price - 1, -1.0).mean() * 100
            print(f"  control: {side:17} {r:+6.2f}%")

        rows = [settle(d, e) for e in (0.0, 0.02, 0.05, 0.10)]
        print(pd.DataFrame([r for r in rows if r.get("bets", 0) >= 50])
              .to_string(index=False))

        allr = settle(d, 0.02, one_per_match=False)
        if allr.get("bets", 0) >= 50:
            print(f"  every positive-EV rung at edge 2%: {allr['bets']:,} bets on "
                  f"{allr['matches']:,} matches, ROI {allr['roi_pct']:+.2f}% "
                  f"[{allr['ci_lo']:+.2f}, {allr['ci_hi']:+.2f}]")
    con.close()


if __name__ == "__main__":
    main()
