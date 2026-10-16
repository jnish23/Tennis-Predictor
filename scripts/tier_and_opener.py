"""Two questions about how the model is graded.

1. **By tier.** The headline accuracy pools Challengers (more parity, noisier)
   with the main tour. Scored separately, per tour, against two baselines: the
   higher-ranked player winning, and the Elo-only model.
2. **Opening vs closing line.** The closing Pinnacle line is the hardest target
   there is. Does the model beat the *opener*, and does a model + opener blend
   beat the opener or even the close? The blend weight is fitted on 2013-2022
   and scored once on 2023-2026.

Run:  python scripts/tier_and_opener.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.argv = sys.argv[:1]                      # slice_analysis parses argv on import

from scripts import slice_analysis as S  # noqa: E402
from tennis.config import ARTIFACTS  # noqa: E402
from tennis.models.common import log_loss  # noqa: E402

pd.set_option("display.width", 200)


def by_tier() -> None:
    bt = pd.read_parquet(ARTIFACTS / "backtest_pooled.parquet")
    for tour in ("atp", "wta"):
        f = pd.read_parquet(ARTIFACTS / f"features_{tour}.parquet",
                            columns=["match_id", "p1_rank", "p2_rank", "best_of"])
        d = bt[bt.tour == tour].merge(f, on="match_id")
        d["tier"] = np.where(d.level.astype(str) == "challenger", "challenger",
                             d.level.astype(str))
        # "higher-ranked wins" baseline, where both ranks exist
        r = d.dropna(subset=["p1_rank", "p2_rank"])
        d["rank_pick"] = np.nan
        d.loc[r.index, "rank_pick"] = (r.p1_rank < r.p2_rank).astype(float)
        rows = []
        groups = {"ALL": d, "main tour (no challenger)": d[d.is_challenger == 0]}
        for lv in ("challenger", "atp250", "atp500", "masters", "grand_slam"):
            groups[lv] = d[d.tier == lv]
        for name, g in groups.items():
            if len(g) < 200:
                continue
            y, p = g.y_win.to_numpy(), g.p_win.to_numpy()
            k = g.dropna(subset=["rank_pick"])
            e = g.dropna(subset=["elo_prob"])
            rows.append({"group": name, "n": len(g),
                         "model_acc": ((p >= .5) == (y == 1)).mean(),
                         "elo_acc": ((e.elo_prob >= .5) == (e.y_win == 1)).mean(),
                         "rank_acc": ((k.rank_pick == 1) == (k.y_win == 1)).mean(),
                         "model_ll": log_loss(y, p),
                         "elo_ll": log_loss(e.y_win.to_numpy(), e.elo_prob.to_numpy())})
        out = pd.DataFrame(rows)
        out["acc_gain_vs_rank"] = out.model_acc - out.rank_acc
        print(f"\n=== {tour.upper()} walk-forward, by tier ===")
        print(out.round(4).to_string(index=False))


def opener() -> None:
    d = S.derive(S.build_frame())
    d = d.dropna(subset=["mkt_o"]).copy()
    d["ll_open"] = S.ll(d.y_win, d.mkt_o)
    d["ll_close"] = S.ll(d.y_win, d.mkt_c)
    d["ll_model"] = S.ll(d.y_win, d.p_win)
    logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))

    def blend(df, w):
        z = (1 - w) * logit(df.mkt_o) + w * logit(df.p_win)
        return 1 / (1 + np.exp(-z))

    def paired(a, b):
        x = (a - b).to_numpy()
        return x.mean(), x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))

    disc, val = d[d.season <= S.DISC_END], d[d.season > S.DISC_END]
    grid = np.linspace(0, 1, 41)
    w = grid[int(np.argmin([S.ll(disc.y_win, blend(disc, x)).mean() for x in grid]))]
    print(f"\n=== Opening vs closing line (ATP, Pinnacle, {len(d)} matches with both) ===")
    print(f"blend weight on the model, fitted on 2013-2022: {w:.2f}")
    rows = []
    for name, g in (("2013-2022 (discovery)", disc), ("2023-2026 (validation)", val),
                    ("validation, main tour only", val[val.is_challenger == 0]),
                    ("validation, challenger only", val[val.is_challenger == 1])):
        bl = S.ll(g.y_win, blend(g, w))
        m1, t1 = paired(g.ll_model, g.ll_open)
        m2, t2 = paired(pd.Series(bl, index=g.index), g.ll_open)
        m3, t3 = paired(pd.Series(bl, index=g.index), g.ll_close)
        rows.append({"sample": name, "n": len(g),
                     "model - open": m1, "t1": t1,
                     "blend - open": m2, "t2": t2,
                     "blend - close": m3, "t3": t3,
                     "open - close": (g.ll_open - g.ll_close).mean()})
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("(negative = first thing better; open - close > 0 means the close is the sharper line)")


if __name__ == "__main__":
    by_tier()
    opener()
