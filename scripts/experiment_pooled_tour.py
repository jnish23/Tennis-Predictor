"""Should the WTA model be trained alone, or pooled with the ATP?

The case for pooling is data: 201k ATP rows on top of 72k WTA ones, and most
of what the model reads -- rank gaps, Elo gaps, form, fatigue -- describes a
tennis match rather than a tour. The case against is structure: the ATP holds
serve on 62.3% of points and the WTA on 56.1%, so the mapping from a serve
edge to a match probability genuinely differs, and pooled, the larger tour
sets the shape for the smaller one.

Both arguments are plausible, which is why this measures instead of choosing.

Design, and the parts that matter:

* **Identical test rows.** Both arms are scored on the same walk-forward test
  folds of the target tour. Only the training set differs, so nothing is
  flattered by an easier sample.
* **The same walk-forward discipline for both.** The pooled arm trains on
  every match of *either* tour played before the test season -- never on
  contemporaneous ATP data, which would be a subtler form of leakage than the
  usual kind and just as fatal.
* **`tour` as a feature in the pooled arm**, so it has the chance to learn the
  difference rather than being forced to average over it. Pooling without it
  would be a straw man.
* **Paired bootstrap by match** on the per-row log-loss difference, because
  the two arms predict the same rows and the paired difference is far less
  noisy than two independent intervals.

Run:  python scripts/experiment_pooled_tour.py [--tour=wta]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.isotonic import IsotonicRegression  # noqa: E402

from tennis.features.pipeline import features_path  # noqa: E402
from tennis.models.common import (CATEGORICAL, FEATURE_COLS,  # noqa: E402
                                  log_loss, prepare)
from tennis.models.train import WIN_PARAMS, WIN_ROUNDS, _fit  # noqa: E402

EPS = 1e-6
START_SEASON = 2012


def per_row_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_predict(tr: pd.DataFrame, te: pd.DataFrame, feats: list[str],
                target: str) -> np.ndarray:
    """Fit the winner model with its isotonic calibrator.

    This is the original 85/15 scheme, kept so the two arms stay comparable with
    the figures quoted elsewhere; `train.run_backtest` has since moved to
    out-of-fold calibration. The calibrator is fitted on the target tour's rows
    of the held-out tail only. One curve fitted on the pooled tail (74% ATP)
    pulls the WTA onto the ATP's miscalibration -- its ECE came out 5.4x the
    ATP's -- and that is a property of the calibrator, not of pooling the fit.
    """
    cut = int(len(tr) * 0.85)
    tr_fit, tr_cal = tr.iloc[:cut], tr.iloc[cut:]
    tr_cal = tr_cal[(tr_cal["tour"] == target).to_numpy()]
    m = _fit(WIN_PARAMS, tr_fit, tr_fit["y_win"].to_numpy(), feats,
             rounds=WIN_ROUNDS)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
    iso.fit(m.predict(tr_cal[feats]), tr_cal["y_win"].to_numpy())
    return iso.predict(m.predict(te[feats]))


def main() -> None:
    target = next((a.split("=", 1)[1] for a in sys.argv
                   if a.startswith("--tour=")), "wta")
    other = "atp" if target == "wta" else "wta"

    tgt = prepare(pd.read_parquet(features_path(target)))
    oth = prepare(pd.read_parquet(features_path(other)))
    print(f"target {target}: {len(tgt):,} rows | other {other}: {len(oth):,} rows")

    # `tour` is a feature only in the pooled arm; the separate arm would see a
    # single constant, which LightGBM ignores anyway.
    # Re-run prepare after the concat. Two frames whose categoricals hold
    # different category sets -- and they do, the tours share few tournaments
    # and no players -- concatenate to plain object/str, which LightGBM
    # rejects outright rather than silently mistreating.
    # Sorted by date: the 85/15 split below is positional, and a frame with the
    # target tour first and the other after it put every ATP row in the
    # calibration tail and every WTA row in the fit.
    both = prepare(pd.concat([tgt, oth], ignore_index=True)
                   .sort_values(["tourney_date", "tour", "seq"], kind="stable")
                   .reset_index(drop=True))
    both["tour"] = both["tour"].astype("category")
    feats_sep = list(FEATURE_COLS)
    feats_pool = list(FEATURE_COLS) + ["tour"]

    season = tgt["tourney_date"].to_numpy() // 10000
    both_season = both["tourney_date"].to_numpy() // 10000

    is_target = (both["tour"] == target).to_numpy()

    rows, losses = [], []
    for s in [x for x in np.unique(season) if x >= START_SEASON]:
        te = tgt[season == s]
        tr_sep = tgt[season < s]
        tr_pool = both[both_season < s]
        # The pooled arm must predict from the *pooled* frame: its categoricals
        # carry the union of both tours' categories, and LightGBM refuses to
        # score a frame whose category set differs from the one it trained on.
        # Same matches either way -- asserted below rather than assumed.
        te_pool = both[(both_season == s) & is_target]
        if len(te) < 200 or len(tr_sep) < 2000:
            continue
        assert (te["match_id"].to_numpy() == te_pool["match_id"].to_numpy()).all(), \
            "pooled and separate arms are not scoring the same matches"
        t0 = time.time()
        p_sep = fit_predict(tr_sep, te, feats_sep, target)
        p_pool = fit_predict(tr_pool, te_pool, feats_pool, target)
        y = te["y_win"].to_numpy()
        l_sep, l_pool = per_row_loss(y, p_sep), per_row_loss(y, p_pool)
        losses.append(pd.DataFrame({"season": s, "sep": l_sep, "pool": l_pool}))
        rows.append({"season": int(s), "test": len(te),
                     "train_sep": len(tr_sep), "train_pool": len(tr_pool),
                     "sep_ll": round(float(l_sep.mean()), 5),
                     "pool_ll": round(float(l_pool.mean()), 5),
                     "delta": round(float(l_pool.mean() - l_sep.mean()), 5),
                     "sec": round(time.time() - t0, 1)})
        print(f"  {s}: sep {rows[-1]['sep_ll']:.5f}  pool {rows[-1]['pool_ll']:.5f}  "
              f"delta {rows[-1]['delta']:+.5f}  ({rows[-1]['sec']}s)")

    if not rows:
        print("no usable folds")
        return

    print("\n=== per season (delta < 0 means pooling is better) ===")
    print(pd.DataFrame(rows).to_string(index=False))

    all_l = pd.concat(losses, ignore_index=True)
    diff = (all_l["pool"] - all_l["sep"]).to_numpy()
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(diff), (2000, len(diff)))
    boot = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    print(f"\n=== pooled overall, {len(all_l):,} scored matches ===")
    print(f"  separate log loss : {all_l['sep'].mean():.5f}")
    print(f"  pooled   log loss : {all_l['pool'].mean():.5f}")
    print(f"  delta             : {diff.mean():+.5f}  "
          f"95% CI [{lo:+.5f}, {hi:+.5f}]")
    print(f"  P(pooling better) : {(boot < 0).mean():.3f}")
    seasons_better = sum(1 for r in rows if r["delta"] < 0)
    print(f"  pooling wins {seasons_better} of {len(rows)} seasons")

    verdict = ("POOL" if hi < 0 else "SEPARATE" if lo > 0 else "NO CLEAR WINNER")
    print(f"\n  verdict: {verdict}")
    if verdict == "NO CLEAR WINNER":
        print("  The interval spans zero, so the data does not distinguish them."
              "\n  Separate is then the better default: it is simpler to reason"
              "\n  about, and it cannot let one tour's structure drift into the"
              "\n  other's as their sample sizes diverge.")


if __name__ == "__main__":
    main()
