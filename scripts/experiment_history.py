"""Does deeper history help, or does old tennis mislead the model?

The question behind ingesting the 1990s. We cannot test 1990s data we have not
ingested, but we can test the thing that actually decides it: at each walk-
forward fold, is a model trained on *all* prior seasons better than one trained
on only the most recent N? If truncating history helps, tennis has drifted
enough that older matches actively mislead, and going further back would make it
worse. If deeper windows keep winning, the drift is mild and more history pays.

Same folds, same params, same features as the production backtest -- only the
training window changes.

Run:  python scripts/experiment_history.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tennis.features.pipeline import FEATURES_PATH  # noqa: E402
from tennis.models.common import (  # noqa: E402
    CATEGORICAL, FEATURE_COLS, log_loss, prepare,
)
from tennis.models.train import WIN_PARAMS, WIN_ROUNDS  # noqa: E402

# Windows in seasons of training history. None = everything available.
WINDOWS = [4, 6, 8, 10, 14, None]
TEST_SEASONS = list(range(2014, 2026))   # each needs >=14 prior seasons to exist


def main() -> None:
    df = pd.read_parquet(FEATURES_PATH)
    df = prepare(df)
    df["season"] = df["tourney_date"] // 10000
    print(f"{len(df):,} rows, seasons {df.season.min()}-{df.season.max()}")

    rows = []
    for s in TEST_SEASONS:
        test = df[df["season"] == s]
        if test.empty:
            continue
        yte = test["y_win"].to_numpy()
        for w in WINDOWS:
            lo = -np.inf if w is None else s - w
            tr = df[(df["season"] < s) & (df["season"] >= lo)]
            if len(tr) < 5000:
                continue
            t0 = time.time()
            booster = lgb.train(
                WIN_PARAMS,
                lgb.Dataset(tr[FEATURE_COLS], label=tr["y_win"],
                            categorical_feature=CATEGORICAL,
                            free_raw_data=False),
                num_boost_round=WIN_ROUNDS,
            )
            p = booster.predict(test[FEATURE_COLS])
            rows.append({
                "season": s,
                "window": "all" if w is None else w,
                "train_rows": len(tr),
                "train_seasons": int(tr["season"].nunique()),
                "log_loss": log_loss(yte, p),
                "accuracy": float(((p >= 0.5) == (yte == 1)).mean()),
                "secs": round(time.time() - t0, 1),
            })
            print(f"  {s} window={str(w):>4}  n={len(tr):>7,}  "
                  f"ll={rows[-1]['log_loss']:.5f}  ({rows[-1]['secs']}s)")

    out = pd.DataFrame(rows)
    out.to_csv("artifacts/history_window_experiment.csv", index=False)

    print("\n=== mean over seasons ===")
    agg = (out.groupby("window")
              .agg(log_loss=("log_loss", "mean"),
                   accuracy=("accuracy", "mean"),
                   avg_train_rows=("train_rows", "mean"))
              .sort_values("log_loss"))
    print(agg.round(5).to_string())

    piv = out.pivot(index="season", columns="window", values="log_loss")
    print("\n=== per season (log loss, lower better) ===")
    print(piv.round(5).to_string())
    best = piv.idxmin(axis=1)
    print("\nbest window per season:")
    print(best.to_string())
    print("\nwin counts:", best.value_counts().to_dict())


if __name__ == "__main__":
    main()
