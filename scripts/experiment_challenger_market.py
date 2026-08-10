"""Are Challenger betting lines softer than main tour, and is that worth anything?

The question our data could not previously answer: tennis-data.co.uk prices ATP
main tour only, so roughly half the match volume had no price at all.
`tennis/ingest/odds_cbo.py` fills that in from checkbestodds.com.

Two caveats govern how hard the results can be read, and neither is small.

**The price is "best odds", not a book's close.** It is the maximum per side
across whichever books covered the match, so it is an upper bound on what any
single account could have taken. ROI computed from it is optimistic by
construction. It is quoted here as a ceiling, not a forecast.

**Best odds is contaminated by transposed books.** The max is taken per side
independently, so one bookmaker with its two sides swapped poisons a column --
Cecchinato-Sergeyev 2013 listed 3.47/26.00 while eleven of twelve books said
~3.20/~1.29. Rows whose implied overround falls outside [1.00, 1.20] are
dropped before anything is computed; that removes 31% of the scrape, almost all
of it standing "arbitrage" that cannot exist.

The comparison that survives both caveats is the *relative* one: the same source
and the same treatment applied to Challengers and to main tour, so whatever bias
best-odds introduces applies to both sides of the comparison.

Run:  python scripts/experiment_challenger_market.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tennis.models.common import log_loss  # noqa: E402
from tennis.models.evaluate import attach_odds, load_backtest  # noqa: E402

EPS = 1e-6


def logit(p):
    return np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS)))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_logistic(X, y, iters=200):
    X = np.column_stack([np.ones(len(X)), X])
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = np.clip(sigmoid(X @ b), 1e-12, 1 - 1e-12)
        g = X.T @ (y - p)
        H = (X * (p * (1 - p))[:, None]).T @ X
        step = np.linalg.solve(H + 1e-9 * np.eye(X.shape[1]), g)
        b += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return b


def se_of(X, b):
    X = np.column_stack([np.ones(len(X)), X])
    p = np.clip(sigmoid(X @ b), 1e-12, 1 - 1e-12)
    return np.sqrt(np.diag(np.linalg.inv((X * (p * (1 - p))[:, None]).T @ X)))


def roi(d: pd.DataFrame, edge: float) -> dict:
    """Flat stakes on whichever side shows an edge against its own price."""
    ev1 = d["p_win"] * d["p1_price"] - 1
    ev2 = (1 - d["p_win"]) * d["p2_price"] - 1
    side = np.where(ev1 >= ev2, 1, 2)
    ev = np.maximum(ev1, ev2)
    keep = ev > edge
    if keep.sum() == 0:
        return {"edge": edge, "bets": 0}
    price = np.where(side == 1, d["p1_price"], d["p2_price"])[keep]
    won = np.where(side == 1, d["y_win"] == 1, d["y_win"] == 0)[keep]
    profit = np.where(won, price - 1, -1.0)
    return {
        "edge": edge, "bets": int(keep.sum()),
        "roi_pct": round(float(profit.sum() / keep.sum() * 100), 2),
        "hit_rate": round(float(won.mean()), 4),
        "breakeven": round(float(np.mean(1 / price)), 4),
        "avg_price": round(float(price.mean()), 3),
    }


def main() -> None:
    bt = load_backtest()
    d = attach_odds(bt, book="CBO").dropna(subset=["mkt_p1", "p_win"])
    d = d[(d.mkt_p1 > EPS) & (d.mkt_p1 < 1 - EPS)].copy()
    d["season"] = d.tourney_date // 10000
    d["tier"] = np.where(d.is_challenger == 1, "challenger", "main tour")
    d["lm"], d["lk"] = logit(d.p_win), logit(d.mkt_p1)
    d["overround"] = 1 / d.p1_price + 1 / d.p2_price

    print(f"{len(d):,} priced matches, {d.season.min()}-{d.season.max()}")
    print(d.tier.value_counts().to_string(), "\n")

    # ---- is the line softer? -------------------------------------------
    print("=== market quality by tier (same source, same treatment) ===")
    rows = []
    for t, g in d.groupby("tier"):
        y = g.y_win.to_numpy(float)
        X = g[["lm", "lk"]].to_numpy()
        b = fit_logistic(X, y)
        s = se_of(X, b)
        rows.append({
            "tier": t, "n": len(g),
            "overround": round(float(g.overround.median()), 4),
            "model_ll": round(log_loss(y, g.p_win.to_numpy()), 5),
            "market_ll": round(log_loss(y, g.mkt_p1.to_numpy()), 5),
            "gap": round(log_loss(y, g.p_win.to_numpy())
                         - log_loss(y, g.mkt_p1.to_numpy()), 5),
            "w_model": round(b[1], 3), "z_model": round(b[1] / s[1], 1),
            "w_market": round(b[2], 3),
        })
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n  gap > 0 means the market is better. w_model is the weight our")
    print("  probability earns beside the price; z > 2 is the bar.")

    # ---- market calibration, which is what 'soft' really means ----------
    print("\n=== is the Challenger line badly calibrated? ===")
    for t, g in d.groupby("tier"):
        g = g.copy()
        g["bucket"] = pd.cut(g.mkt_p1, np.arange(0, 1.01, 0.1))
        tbl = g.groupby("bucket", observed=True).agg(
            n=("y_win", "size"), implied=("mkt_p1", "mean"),
            actual=("y_win", "mean")).dropna()
        tbl["err"] = (tbl.actual - tbl.implied).round(4)
        print(f"\n  {t}  (mean |error| {tbl.err.abs().mean():.4f})")
        print(tbl.round(4).to_string())

    # ---- ROI ------------------------------------------------------------
    print("\n=== ROI, flat stakes  [CEILING: best-odds, not a single book] ===")
    for t, g in d.groupby("tier"):
        print(f"\n  {t}  (n={len(g):,})")
        print(pd.DataFrame([roi(g, e) for e in (0.0, 0.03, 0.05, 0.10)])
              .to_string(index=False))

    # ---- walk-forward, so nothing is fit on its own test set ------------
    print("\n=== Challenger ROI by season (walk-forward by construction) ===")
    ch = d[d.tier == "challenger"]
    rows = []
    for s, g in ch.groupby("season"):
        if len(g) < 200:
            continue
        r = roi(g, 0.03)
        rows.append({"season": int(s), **r})
    print(pd.DataFrame(rows).to_string(index=False))

    # ---- sanity: does this source agree with Pinnacle on main tour? -----
    print("\n=== sanity check: CBO vs Pinnacle on the same main-tour matches ===")
    ps = attach_odds(bt, book="PS").dropna(subset=["mkt_p1"])[["match_id", "mkt_p1"]]
    ps = ps.rename(columns={"mkt_p1": "ps_p1"})
    both = d[d.tier == "main tour"].merge(ps, on="match_id", how="inner")
    if len(both) > 500:
        y = both.y_win.to_numpy(float)
        print(f"  {len(both):,} matches priced by both")
        print(f"    correlation of implied probs : "
              f"{np.corrcoef(both.mkt_p1, both.ps_p1)[0, 1]:.4f}")
        print(f"    CBO best-odds log loss       : {log_loss(y, both.mkt_p1.to_numpy()):.5f}")
        print(f"    Pinnacle close log loss      : {log_loss(y, both.ps_p1.to_numpy()):.5f}")
        print(f"    median overround CBO {both.overround.median():.4f}")


if __name__ == "__main__":
    main()
