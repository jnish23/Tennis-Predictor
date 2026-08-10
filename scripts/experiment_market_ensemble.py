"""Model probability against the closing line: agreement, and whether blending helps.

Two questions, and the second is the one that matters.

**How closely do they agree?** Correlation on the probabilities themselves is
flattering -- both are bounded in [0,1] and pile up near the extremes, so almost
any two sane forecasts correlate highly. Correlation on the *logit* is the
honest version, because that is the scale a blend actually operates on.

**Does our probability earn any weight beside the market's?** Fit
`y ~ logit(model) + logit(market)` and read the coefficients. A model that is
pure noise once you know the price gets a coefficient of zero; one carrying
independent information gets a positive one. The in-sample fit answers "is there
signal", but it sees the answers, so the weight is also refit **walk-forward** --
seasons before S choose the weight, season S scores it -- which is the only
version that says anything about using this in front of a live market.

Run:  python scripts/experiment_market_ensemble.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tennis.models.common import brier, log_loss  # noqa: E402
from tennis.models.evaluate import attach_odds, load_backtest  # noqa: E402

EPS = 1e-6
BOOKS = ["PS", "B365", "Avg", "Max"]


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def prep(book: str) -> pd.DataFrame:
    df = attach_odds(load_backtest(), book=book).dropna(
        subset=["mkt_p1", "p_win", "y_win"])
    # A handful of rows carry a price so short the devigged probability pins to
    # 0 or 1; they would dominate a log-loss comparison on rounding alone.
    df = df[(df["mkt_p1"] > EPS) & (df["mkt_p1"] < 1 - EPS)].copy()
    df["season"] = df["tourney_date"] // 10000
    df["lm"] = logit(df["p_win"])
    df["lk"] = logit(df["mkt_p1"])
    return df


def fit_logistic(X: np.ndarray, y: np.ndarray, iters: int = 200) -> np.ndarray:
    """Newton-Raphson with an intercept. Small, dense, well conditioned."""
    X = np.column_stack([np.ones(len(X)), X])
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = np.clip(sigmoid(X @ b), 1e-12, 1 - 1e-12)
        W = p * (1 - p)
        g = X.T @ (y - p)
        H = (X * W[:, None]).T @ X
        step = np.linalg.solve(H + 1e-9 * np.eye(X.shape[1]), g)
        b += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return b


def se_of(X: np.ndarray, b: np.ndarray) -> np.ndarray:
    X = np.column_stack([np.ones(len(X)), X])
    p = np.clip(sigmoid(X @ b), 1e-12, 1 - 1e-12)
    H = (X * (p * (1 - p))[:, None]).T @ X
    return np.sqrt(np.diag(np.linalg.inv(H)))


def agreement(df: pd.DataFrame) -> dict:
    from scipy.stats import pearsonr, spearmanr

    d = (df["p_win"] - df["mkt_p1"]).abs()
    return {
        "n": len(df),
        "pearson_prob": round(pearsonr(df["p_win"], df["mkt_p1"])[0], 4),
        "spearman_prob": round(spearmanr(df["p_win"], df["mkt_p1"])[0], 4),
        "pearson_logit": round(pearsonr(df["lm"], df["lk"])[0], 4),
        "mean_abs_diff": round(float(d.mean()), 4),
        "median_abs_diff": round(float(d.median()), 4),
        "pct_within_2pt": round(float((d < 0.02).mean() * 100), 1),
        "pct_over_10pt": round(float((d > 0.10).mean() * 100), 1),
        "model_ll": round(log_loss(df["y_win"].to_numpy(), df["p_win"].to_numpy()), 5),
        "market_ll": round(log_loss(df["y_win"].to_numpy(), df["mkt_p1"].to_numpy()), 5),
    }


def main() -> None:
    df = prep("PS")
    y = df["y_win"].to_numpy(float)
    print(f"Pinnacle closing prices, {len(df):,} matches, "
          f"{df.season.min()}-{df.season.max()}\n")

    # ---- agreement ------------------------------------------------------
    print("=== agreement ===")
    a = agreement(df)
    for k, v in a.items():
        print(f"  {k:18} {v}")

    print("\n  by book:")
    rows = []
    for b in BOOKS:
        try:
            rows.append({"book": b, **agreement(prep(b))})
        except Exception as exc:  # a book may not join at all
            print(f"    {b}: {exc}")
    print(pd.DataFrame(rows).to_string(index=False))

    # ---- who is right when they disagree --------------------------------
    print("\n=== when they disagree, who is right? ===")
    d = df["p_win"] - df["mkt_p1"]
    buckets = pd.cut(d, [-1, -0.10, -0.05, -0.02, 0.02, 0.05, 0.10, 1],
                     labels=["model −10pt+", "−10..−5", "−5..−2", "±2",
                             "+2..+5", "+5..+10", "model +10pt+"])
    out = []
    for lab, g in df.groupby(buckets, observed=True):
        yy = g["y_win"].to_numpy(float)
        out.append({"disagreement": lab, "n": len(g),
                    "model_ll": round(log_loss(yy, g["p_win"].to_numpy()), 5),
                    "market_ll": round(log_loss(yy, g["mkt_p1"].to_numpy()), 5),
                    "model_better": log_loss(yy, g["p_win"].to_numpy())
                    < log_loss(yy, g["mkt_p1"].to_numpy())})
    print(pd.DataFrame(out).to_string(index=False))

    # ---- in-sample weights ----------------------------------------------
    print("\n=== ensemble weights (in-sample, sees the answers) ===")
    X = df[["lm", "lk"]].to_numpy()
    b = fit_logistic(X, y)
    se = se_of(X, b)
    names = ["intercept", "logit(model)", "logit(market)"]
    for n_, c, s in zip(names, b, se):
        z = c / s
        print(f"  {n_:14} {c:+.4f}  (se {s:.4f}, z {z:+.1f})")
    # market alone, for a likelihood-ratio comparison
    b_k = fit_logistic(df[["lk"]].to_numpy(), y)
    ll_both = -log_loss(y, sigmoid(np.column_stack([np.ones(len(X)), X]) @ b)) * len(y)
    ll_mkt = -log_loss(y, sigmoid(
        np.column_stack([np.ones(len(df)), df[["lk"]].to_numpy()]) @ b_k)) * len(y)
    lr = 2 * (ll_both - ll_mkt)
    from scipy.stats import chi2
    print(f"  likelihood-ratio vs market-only: chi2(1) = {lr:.1f}, "
          f"p = {chi2.sf(lr, 1):.3g}")

    # ---- fixed-weight blend sweep ---------------------------------------
    print("\n=== fixed logit blend  w*model + (1-w)*market ===")
    sweep = []
    for w in np.round(np.arange(0, 1.01, 0.05), 2):
        p = sigmoid(w * df["lm"].to_numpy() + (1 - w) * df["lk"].to_numpy())
        sweep.append({"w_model": w, "log_loss": round(log_loss(y, p), 5),
                      "brier": round(brier(y, p), 5)})
    sw = pd.DataFrame(sweep)
    print(sw.to_string(index=False))
    best = sw.loc[sw.log_loss.idxmin()]
    print(f"  best fixed weight on the model: {best.w_model:.2f} "
          f"(log loss {best.log_loss:.5f})")

    # ---- walk-forward ensemble ------------------------------------------
    print("\n=== ensemble refit walk-forward (honest) ===")
    rows = []
    for s in sorted(df.season.unique()):
        tr, te = df[df.season < s], df[df.season == s]
        if len(tr) < 3000 or te.empty:
            continue
        bw = fit_logistic(tr[["lm", "lk"]].to_numpy(), tr["y_win"].to_numpy(float))
        Xte = np.column_stack([np.ones(len(te)), te[["lm", "lk"]].to_numpy()])
        p_ens = sigmoid(Xte @ bw)
        yy = te["y_win"].to_numpy(float)
        rows.append({
            "season": int(s), "n": len(te),
            "w_model": round(bw[1], 3), "w_market": round(bw[2], 3),
            "model": round(log_loss(yy, te["p_win"].to_numpy()), 5),
            "market": round(log_loss(yy, te["mkt_p1"].to_numpy()), 5),
            "ensemble": round(log_loss(yy, p_ens), 5),
        })
    wf = pd.DataFrame(rows)
    print(wf.to_string(index=False))

    # Rebuild the walk-forward predictions cleanly rather than inline, so the
    # pooled figure is demonstrably the same thing the per-season table scored.
    parts = []
    for s in wf.season:
        tr, te = df[df.season < s], df[df.season == s]
        bw = fit_logistic(tr[["lm", "lk"]].to_numpy(), tr["y_win"].to_numpy(float))
        Xte = np.column_stack([np.ones(len(te)), te[["lm", "lk"]].to_numpy()])
        parts.append(te.assign(p_ens=sigmoid(Xte @ bw)))
    tot = pd.concat(parts, ignore_index=True)
    yt, pe = tot["y_win"].to_numpy(float), tot["p_ens"].to_numpy()
    print(f"\n  pooled over {len(tot):,} matches:")
    print(f"    model    {log_loss(yt, tot['p_win'].to_numpy()):.5f}")
    print(f"    market   {log_loss(yt, tot['mkt_p1'].to_numpy()):.5f}")
    print(f"    ensemble {log_loss(yt, pe):.5f}")

    cl = lambda p: np.clip(np.asarray(p, float), 1e-15, 1 - 1e-15)  # noqa: E731
    le = -(yt * np.log(cl(pe)) + (1 - yt) * np.log(1 - cl(pe)))
    km = cl(tot["mkt_p1"])
    lk = -(yt * np.log(km) + (1 - yt) * np.log(1 - km))
    diff = le - lk
    rng = np.random.default_rng(0)
    bs = diff[rng.integers(0, len(diff), (3000, len(diff)))].mean(axis=1)
    print(f"    ensemble − market: {diff.mean():+.5f}  "
          f"95% CI [{np.percentile(bs, 2.5):+.5f}, {np.percentile(bs, 97.5):+.5f}]"
          f"  P(ensemble better) = {(bs < 0).mean():.3f}")

    # ---- where a thinner market might leave room -------------------------
    print("\n=== by segment (a softer market is where the model could earn weight) ===")
    print("  Challengers carry no prices at all in tennis-data.co.uk, so the")
    print("  obvious soft-market test is not available. Tournament level is the")
    print("  closest proxy the data supports: a 250 draws less attention than a")
    print("  slam, so if the model has anything to add it should show up there.")
    seg = []
    for label, g in list(df.groupby("level")) + [("ALL", df)]:
        if len(g) < 1500:
            continue
        gy = g["y_win"].to_numpy(float)
        bb = fit_logistic(g[["lm", "lk"]].to_numpy(), gy)
        ss = se_of(g[["lm", "lk"]].to_numpy(), bb)
        seg.append({"level": label, "n": len(g),
                    "w_model": round(bb[1], 3), "z_model": round(bb[1] / ss[1], 1),
                    "w_market": round(bb[2], 3),
                    "model_ll": round(log_loss(gy, g["p_win"].to_numpy()), 5),
                    "market_ll": round(log_loss(gy, g["mkt_p1"].to_numpy()), 5)})
    print(pd.DataFrame(seg).sort_values("n", ascending=False).to_string(index=False))
    print(f"\n  priced challenger matches: {(df.is_challenger == 1).sum()}")

    print(f"    mean walk-forward weight on the model: {wf.w_model.mean():.3f}")


if __name__ == "__main__":
    main()
