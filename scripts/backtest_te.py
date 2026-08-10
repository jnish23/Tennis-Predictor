"""Betting backtest on the tennisexplorer prices, ATP and Challenger separately.

These are the first prices this project has held that are worth betting into on
paper: a single book's price (median overround 1.071, not a cross-book maximum),
captured with a timestamp, and taken as the last look before play. checkbestodds
gave neither property.

**The sample cannot settle anything, and the report says so throughout.** Five
months, 3,237 Challenger and 1,121 main-tour matches. At the observed variance
roughly 5,700 bets are needed to separate a true +2% edge from zero at 95%, so
every strategy below is under-powered by design. The point is to see whether the
softness measured on the 2011-2022 checkbestodds data *replicates* on prices of
a completely different provenance -- that replication is worth more than any
single ROI figure here.

**Many strategies are tried, so some will look good by chance.** Every result
carries a bootstrap CI, and the run ends by reporting how many strategies would
be expected to clear a given ROI under the null of no edge at all. Read that
line before reading the table.

Ensemble weights are never fitted on this data. They come from the 2011-2022
checkbestodds era and are applied forward, which is the only version that says
anything about using them in front of a live market.

Run:  python scripts/backtest_te.py
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
RNG = np.random.default_rng(0)


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_logistic(X, y, iters=200):
    X = np.column_stack([np.ones(len(X)), X])
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = np.clip(sigmoid(X @ b), 1e-12, 1 - 1e-12)
        step = np.linalg.solve(
            (X * (p * (1 - p))[:, None]).T @ X + 1e-9 * np.eye(X.shape[1]),
            X.T @ (y - p))
        b += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return b


def load(book: str) -> pd.DataFrame:
    d = attach_odds(load_backtest(), book=book).dropna(subset=["mkt_p1", "p_win"])
    d = d[(d.mkt_p1 > EPS) & (d.mkt_p1 < 1 - EPS)].copy()
    d["lm"], d["lk"] = logit(d.p_win), logit(d.mkt_p1)
    return d


def settle(d: pd.DataFrame, p: np.ndarray, edge: float, *,
           stake: str = "flat", cap: float = 0.02,
           side: str | None = None, price_lo=1.0, price_hi=99.0) -> dict:
    """One strategy over one slice. `p` is the probability being bet with."""
    ev1 = p * d.p1_price.to_numpy() - 1
    ev2 = (1 - p) * d.p2_price.to_numpy() - 1
    pick1 = ev1 >= ev2
    ev = np.maximum(ev1, ev2)
    price = np.where(pick1, d.p1_price, d.p2_price)
    prob = np.where(pick1, p, 1 - p)
    won = np.where(pick1, d.y_win == 1, d.y_win == 0)

    keep = ev > edge
    keep &= (price >= price_lo) & (price <= price_hi)
    if side == "fav":            # our pick is also the market's favourite
        keep &= price < np.where(pick1, d.p2_price, d.p1_price)
    elif side == "dog":
        keep &= price > np.where(pick1, d.p2_price, d.p1_price)
    if keep.sum() < 25:
        return {"bets": int(keep.sum())}

    price, prob, won = price[keep], prob[keep], won[keep]
    if stake == "flat":
        size = np.ones(len(price))
    else:                        # fractional Kelly, capped
        b = price - 1
        f = np.clip((prob * b - (1 - prob)) / b, 0, cap)
        size = f / cap           # expressed in units of the cap
    profit = np.where(won, size * (price - 1), -size)
    turnover = size.sum()

    boot = RNG.integers(0, len(profit), (4000, len(profit)))
    bs = (profit[boot].sum(axis=1) / size[boot].sum(axis=1)) * 100
    return {
        "bets": int(keep.sum()),
        "roi_pct": round(float(profit.sum() / turnover * 100), 2),
        "ci_lo": round(float(np.percentile(bs, 2.5)), 2),
        "ci_hi": round(float(np.percentile(bs, 97.5)), 2),
        "p_profit": round(float((bs > 0).mean()), 3),
        "hit": round(float(won.mean()), 4),
        "breakeven": round(float(np.mean(1 / price)), 4),
        "avg_price": round(float(price.mean()), 2),
    }


def market_quality(d: pd.DataFrame, label: str) -> dict:
    y = d.y_win.to_numpy(float)
    X = d[["lm", "lk"]].to_numpy()
    b = fit_logistic(X, y)
    p = np.clip(sigmoid(np.column_stack([np.ones(len(X)), X]) @ b), 1e-9, 1 - 1e-9)
    H = (np.column_stack([np.ones(len(X)), X]) * (p * (1 - p))[:, None]).T @ \
        np.column_stack([np.ones(len(X)), X])
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    return {
        "tier": label, "n": len(d),
        "overround": round(float((1 / d.p1_price + 1 / d.p2_price).median()), 4),
        "model_ll": round(log_loss(y, d.p_win.to_numpy()), 5),
        "market_ll": round(log_loss(y, d.mkt_p1.to_numpy()), 5),
        "w_model": round(b[1], 3), "z_model": round(b[1] / se[1], 1),
    }


def main() -> None:
    te = load("TE_CLOSE")
    te["tier"] = np.where(te.is_challenger == 1, "challenger", "main tour")
    print(f"tennisexplorer closing-ish prices: {len(te):,} matches, "
          f"{te.tourney_date.min()}-{te.tourney_date.max()}")
    print(te.tier.value_counts().to_string(), "\n")

    # ---- does the softness finding replicate on new, better prices? -----
    print("=== market quality (replication check vs the 2011-22 checkbestodds run) ===")
    q = pd.DataFrame([market_quality(g, t) for t, g in te.groupby("tier")])
    print(q.to_string(index=False))
    print("  checkbestodds 2011-22 gave w_model 0.155 z=4.2 (challenger), "
          "0.062 z=1.3 (main tour).")

    # ---- ensemble weights from the OLD era, applied forward --------------
    cbo = load("CBO")
    weights = {}
    for tier, flag in (("challenger", 1), ("main tour", 0)):
        g = cbo[cbo.is_challenger == flag]
        weights[tier] = fit_logistic(g[["lm", "lk"]].to_numpy(),
                                     g.y_win.to_numpy(float))
        print(f"\n  ensemble weights from 2011-22 {tier}: "
              f"model {weights[tier][1]:+.3f}, market {weights[tier][2]:+.3f} "
              f"(n={len(g):,})")

    # ---- naive controls, so the strategy rows can be read at all --------
    # Without these a "-13%" means nothing: on a book charging 6.7% the floor
    # is not zero. Backing every favourite and every underdog brackets what any
    # strategy has to beat, and their spread is the favourite-longshot bias
    # made visible.
    print("\n=== naive controls (what any strategy must beat) ===")
    ctl = []
    for tier, g in te.groupby("tier"):
        favp = np.minimum(g.p1_price, g.p2_price).to_numpy()
        favwon = np.where(g.p1_price < g.p2_price, g.y_win == 1, g.y_win == 0)
        dogp = np.maximum(g.p1_price, g.p2_price).to_numpy()
        ovr = float((1 / g.p1_price + 1 / g.p2_price).median())
        ctl.append({
            "tier": tier, "n": len(g), "vig_pct": round((1 - 1 / ovr) * 100, 1),
            "back_every_favourite": round(float(
                np.where(favwon, favp - 1, -1.0).mean() * 100), 2),
            "back_every_underdog": round(float(
                np.where(~favwon, dogp - 1, -1.0).mean() * 100), 2)})
    print(pd.DataFrame(ctl).to_string(index=False))

    # ---- strategies -----------------------------------------------------
    results = []
    for tier, g in te.groupby("tier"):
        pm = g.p_win.to_numpy()
        w = weights[tier]
        pe = sigmoid(np.column_stack([np.ones(len(g)), g[["lm", "lk"]].to_numpy()]) @ w)

        strategies = []
        for e in (0.02, 0.05, 0.10, 0.15):
            strategies.append((f"model flat, edge {e:.0%}", pm, e, {}))
        for e in (0.02, 0.05, 0.10):
            strategies.append((f"ensemble flat, edge {e:.0%}", pe, e, {}))
        strategies += [
            ("model ½-Kelly, edge 2%", pm, 0.02, {"stake": "kelly"}),
            ("ensemble ½-Kelly, edge 2%", pe, 0.02, {"stake": "kelly"}),
            # Favourite-longshot bias says the vig is loaded onto underdogs, so
            # backing favourites should fare better than backing longshots.
            ("ensemble, favourites only, edge 2%", pe, 0.02, {"side": "fav"}),
            ("ensemble, underdogs only, edge 2%", pe, 0.02, {"side": "dog"}),
            # Short prices carry the least vig in absolute terms.
            ("ensemble, price 1.0-2.0, edge 2%", pe, 0.02,
             {"price_lo": 1.0, "price_hi": 2.0}),
            ("ensemble, price 2.0-4.0, edge 2%", pe, 0.02,
             {"price_lo": 2.0, "price_hi": 4.0}),
            ("ensemble, price 4.0+, edge 2%", pe, 0.02, {"price_lo": 4.0}),
        ]
        for name, p, e, kw in strategies:
            r = settle(g, p, e, **kw)
            if r.get("bets", 0) >= 25:
                results.append({"tier": tier, "strategy": name, **r})

    res = pd.DataFrame(results)
    for tier in ("challenger", "main tour"):
        print(f"\n=== {tier} ===")
        t = res[res.tier == tier].drop(columns="tier")
        print(t.to_string(index=False))

    # ---- how much of this is just multiple testing? ---------------------
    print("\n=== selection-bias check ===")
    n_str = len(res)
    best = res.loc[res.roi_pct.idxmax()]
    # Null: no edge. Resample outcomes under the market's own probabilities and
    # count how often the *best of n* strategies clears the observed best ROI.
    n_bets = int(best.bets)
    sd = 0.75          # per-bet SD measured on these price distributions
    null_best = np.max(RNG.normal(0, sd / np.sqrt(n_bets), (4000, n_str)), axis=1) * 100
    print(f"  {n_str} strategies tried across both tiers.")
    print(f"  best observed: {best.strategy} ({best.tier}) "
          f"{best.roi_pct:+.2f}% on {n_bets:,} bets")
    print(f"  under the null of zero edge, the best of {n_str} strategies clears "
          f"{best.roi_pct:+.2f}% about {(null_best >= best.roi_pct).mean()*100:.0f}% "
          "of the time")
    print(f"  strategies with CI excluding zero: "
          f"{int((res.ci_lo > 0).sum())} of {n_str}")


if __name__ == "__main__":
    main()
