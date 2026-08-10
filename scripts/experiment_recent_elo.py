"""Is a short-memory Elo informative, or already encoded in the features?

The all-time book has long memory *by design*: K decays with match count, so an
established player's rating barely moves and carries a decade of evidence. A
"last 12 months" rating asks a different question -- how a player is going now,
weighted by who they beat.

The window used on the dashboard restarts everyone at 1500 on a fixed date,
which is fine for a form table but useless as a feature: it has a hard boundary
and every rating is meaningless each January. The rolling equivalent is a
**constant-K** Elo. Fixing K instead of decaying it makes each rating an
exponentially-weighted average of recent performance, with a memory set by K --
around a year of play at K=32 for a busy player.

That distinction matters for the question being asked. The candidate features:

    elo_fast          constant-K rating (p1, p2, and the difference)
    d_elo_trend       d_elo_fast - d_elo: who is trending, relative to career

`d_elo_trend` is the one with no existing counterpart. The feature set already
has recent *results* (win rate over 10/25/50, EWMA form) and the *strength of
opponents faced* (opp_elo_10/25) as separate signals -- but nothing that
combines them into "recent results, adjusted for who they came against".

Method matches the earlier feature A/B: train on everything before 2023, score
2023+, paired bootstrap on per-match log loss.

Run:  python scripts/experiment_recent_elo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tennis.db.schema import connect  # noqa: E402
from tennis.features.pipeline import FEATURES_PATH  # noqa: E402
from tennis.models.common import (  # noqa: E402
    CATEGORICAL, FEATURE_COLS, log_loss, prepare,
)
from tennis.models.train import WIN_PARAMS, WIN_ROUNDS  # noqa: E402

K_FAST = 32.0
BASE = 1500.0
HOLDOUT_FROM = 2023


def fast_elo_frame(k: float = K_FAST) -> pd.DataFrame:
    """Pre-match constant-K rating for both sides of every match.

    Read-then-update, exactly as the production engine does, so the value on a
    row never contains that row's result.
    """
    con = connect()
    try:
        df = pd.read_sql(
            """SELECT m.match_id, m.winner_id, m.loser_id, m.status
               FROM matches m ORDER BY m.seq""", con)
    finally:
        con.close()

    r: dict[str, float] = {}
    w_pre = np.empty(len(df))
    l_pre = np.empty(len(df))
    W, L, ST = df.winner_id.to_numpy(), df.loser_id.to_numpy(), df.status.to_numpy()
    for i in range(len(df)):
        w, l = W[i], L[i]
        rw, rl = r.get(w, BASE), r.get(l, BASE)
        w_pre[i], l_pre[i] = rw, rl
        if ST[i] == "walkover":
            continue
        ew = 1.0 / (1.0 + 10.0 ** ((rl - rw) / 400.0))
        r[w] = rw + k * (1.0 - ew)
        r[l] = rl - k * (1.0 - ew)
    return pd.DataFrame({"match_id": df.match_id,
                         "win_elo_fast": w_pre, "lose_elo_fast": l_pre})


def main() -> None:
    df = prepare(pd.read_parquet(FEATURES_PATH))
    df["season"] = df["tourney_date"] // 10000

    fast = fast_elo_frame()
    df = df.merge(fast, on="match_id", how="left")
    # features.parquet assigns p1/p2 by a hash of match_id, so orient by y_win
    p1_won = df["y_win"] == 1
    df["p1_elo_fast"] = np.where(p1_won, df.win_elo_fast, df.lose_elo_fast)
    df["p2_elo_fast"] = np.where(p1_won, df.lose_elo_fast, df.win_elo_fast)
    df["d_elo_fast"] = df["p1_elo_fast"] - df["p2_elo_fast"]
    df["d_elo_trend"] = df["d_elo_fast"] - (df["p1_elo"] - df["p2_elo"])

    new = ["p1_elo_fast", "p2_elo_fast", "d_elo_fast", "d_elo_trend"]
    base = [c for c in FEATURE_COLS if c in df.columns]

    tr = df[df.season < HOLDOUT_FROM]
    te = df[df.season >= HOLDOUT_FROM]
    y = te["y_win"].to_numpy()
    print(f"train {len(tr):,}  holdout {len(te):,} ({HOLDOUT_FROM}+)\n")

    # How much of the new signal is already carried? Regress d_elo_trend on the
    # existing recency and opponent-strength columns and look at what is left.
    proxies = [c for c in base if any(k in c for k in
               ("winrate", "ew_result", "opp_elo", "d_elo", "rank"))]
    from sklearn.linear_model import LinearRegression
    m = df.dropna(subset=["d_elo_trend"])
    X = m[proxies].fillna(m[proxies].median())
    lr = LinearRegression().fit(X, m["d_elo_trend"])
    print(f"d_elo_trend explained by {len(proxies)} existing features: "
          f"R² = {lr.score(X, m['d_elo_trend']):.4f}")
    print(f"  corr(d_elo_fast, d_elo) = {m.d_elo_fast.corr(m.p1_elo - m.p2_elo):.4f}\n")

    results = {}
    for name, cols in [("baseline", base), ("+ recent Elo", base + new)]:
        booster = lgb.train(
            WIN_PARAMS,
            lgb.Dataset(tr[cols], label=tr["y_win"],
                        categorical_feature=[c for c in CATEGORICAL if c in cols],
                        free_raw_data=False),
            num_boost_round=WIN_ROUNDS)
        p = booster.predict(te[cols])
        results[name] = p
        print(f"{name:14} log loss {log_loss(y, p):.5f}  "
              f"acc {((p >= .5) == (y == 1)).mean():.5f}")
        if name != "baseline":
            imp = pd.Series(booster.feature_importance("gain"), index=cols)
            share = imp[new].sum() / imp.sum() * 100
            print(f"  new features take {share:.2f}% of total gain")
            for c in new:
                print(f"    {c:16} {imp[c]:>12,.0f}  (rank "
                      f"{int((imp > imp[c]).sum()) + 1} of {len(cols)})")

    a, b = results["baseline"], results["+ recent Elo"]
    da = -(y * np.log(np.clip(a, 1e-15, 1)) + (1 - y) * np.log(np.clip(1 - a, 1e-15, 1)))
    db = -(y * np.log(np.clip(b, 1e-15, 1)) + (1 - y) * np.log(np.clip(1 - b, 1e-15, 1)))
    d = db - da
    rng = np.random.default_rng(0)
    bs = d[rng.integers(0, len(d), (2000, len(d)))].mean(axis=1)
    print(f"\nΔ log loss {d.mean():+.5f}  "
          f"95% CI [{np.percentile(bs, 2.5):+.5f}, {np.percentile(bs, 97.5):+.5f}]")
    print(f"P(recent Elo better) = {(bs < 0).mean():.3f}")


if __name__ == "__main__":
    main()
