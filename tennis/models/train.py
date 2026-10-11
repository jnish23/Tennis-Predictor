"""Train the three targets and run the walk-forward backtest.

Targets
-------
winner  : binary, P(p1 beats p2), isotonic-calibrated.
totals  : **total games**, not sets. Games is the target that carries real
          information -- in a best-of-3 match "total sets" takes only the values
          2 and 3, so it is very nearly a restatement of the winner model, while
          total games spans roughly 12-40 and is what totals markets price.
          Total sets is still produced as a secondary output for the dashboard.
spread  : signed game margin, p1-relative (positive = p1 won by that many games).

Every fold trains on strictly earlier matches than it scores.
"""
from __future__ import annotations

import json
import logging
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from tennis.config import ARTIFACTS
from tennis.db.schema import connect
from tennis.features.pipeline import features_path
from tennis.models.common import (
    CATEGORICAL,
    FEATURE_COLS,
    calibration_table,
    classification_metrics,
    prepare,
    regression_metrics,
    walk_forward_folds,
)

log = logging.getLogger(__name__)

WIN_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "bagging_freq": 1,
    "verbose": -1,
    "num_threads": 0,
    "learning_rate": 0.01936073659144995,
    "num_leaves": 199,
    "min_data_in_leaf": 970,
    "feature_fraction": 0.8672505423794767,
    "bagging_fraction": 0.8584450945794978,
    "lambda_l1": 1.0,
    "lambda_l2": 5.44483123690318,
    "max_depth": 7,
}

# Kept out of WIN_PARAMS on purpose: LightGBM silently ignores unknown keys in
# the params dict (it only logs "Unknown parameter"), so a round count left in
# there does nothing and the model quietly trains to N_ROUNDS instead. The
# tuning notebook reports this as `num_boost_round_suggestion`; it belongs here.
WIN_ROUNDS = 805

REG_PARAMS = {
    "objective": "regression",
    "metric": "l2",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 5.0,
    "verbose": -1,
    "num_threads": 0
}

N_ROUNDS = 700          # regression targets (totals/spread), untuned so far

# The totals model cares about the *size* of the mismatch, not its direction:
# a 200-point Elo gap shortens the match whichever player holds it.
ABS_FEATURES = ["d_elo", "d_elo_surf", "d_rank", "d_log_rank", "d_serve_edge",
                "d_spw", "d_rpw", "d_winrate_25", "d_h2h"]


TOURS = ("atp", "wta")


def _playable(df: pd.DataFrame) -> pd.DataFrame:
    """Drop walkovers. Nothing was played, so there is no result to predict or
    to be scored on -- they inflated the training and backtest sets by 1,324
    rows. The engine still emits them; they just never reach a model."""
    if "status" not in df.columns:
        return df
    return df[df["status"] != "walkover"].reset_index(drop=True)


def load_features(tour: str = "pooled") -> pd.DataFrame:
    """Features for one tour, or both pooled with `tour` as a model feature.

    Pooled is the default because it was measured to be better, not assumed.
    A model trained on both tours and given the `tour` column beats a WTA-only
    model by 0.0044 log loss (95% CI [-0.0057, -0.0031], 14 of 15 walk-forward
    seasons) and costs the ATP nothing measurable. The tours are genuinely
    different -- 62.3% of serve points won against 56.1% -- but that argues for
    letting the model *see* the difference, not for hiding half the data from
    it. Ratings feeding these features stay strictly per tour; only the fit is
    shared. See scripts/experiment_pooled_tour.py.

    Rows are returned in global chronological order, so the walk-forward split
    below cuts both tours at the same instant rather than by position.
    """
    if tour != "pooled":
        return prepare(_playable(pd.read_parquet(features_path(tour))))
    frames = [_playable(pd.read_parquet(features_path(t))) for t in TOURS]
    df = pd.concat(frames, ignore_index=True)
    # `seq` is assigned per tour, so it cannot order a pooled frame; the date
    # can. Sorting by seq here would interleave two tours' counters and put
    # 2026 WTA matches before 2005 ATP ones.
    df = df.sort_values(["tourney_date", "tour", "seq"]).reset_index(drop=True)
    return prepare(df)


def add_abs(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    cols = []
    for c in ABS_FEATURES:
        if c in out.columns:
            out[f"abs_{c}"] = out[c].abs()
            cols.append(f"abs_{c}")
    return out, cols


def _fit(params, X, y, feats, rounds=N_ROUNDS):
    ds = lgb.Dataset(X[feats], label=y, categorical_feature=[c for c in CATEGORICAL if c in feats],
                     free_raw_data=False)
    return lgb.train(params, ds, num_boost_round=rounds)


def run_backtest(start_season: int = 2010, run_tag: str | None = None,
                 tour: str = "pooled") -> dict:
    t0 = time.time()
    run_tag = run_tag or f"wf_{tour}"
    df = load_features(tour)
    df, abs_cols = add_abs(df)
    win_feats = list(FEATURE_COLS)
    reg_feats = list(FEATURE_COLS) + abs_cols

    folds = walk_forward_folds(df, start_season=start_season)
    log.info("%d walk-forward folds: %s", len(folds), [f[2] for f in folds])

    preds = []
    for train_idx, test_idx, season in folds:
        tr, te = df.iloc[train_idx], df.iloc[test_idx]

        # ---- winner ------------------------------------------------------
        # Hold out the last 15% of the training window (by date) to fit the
        # isotonic calibrator, so calibration is never fitted on test data.
        cut = int(len(tr) * 0.85)
        tr_fit, tr_cal = tr.iloc[:cut], tr.iloc[cut:]
        m_win = _fit(WIN_PARAMS, tr_fit, tr_fit["y_win"].to_numpy(), win_feats,
                     rounds=WIN_ROUNDS)
        raw_cal = m_win.predict(tr_cal[win_feats])
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
        iso.fit(raw_cal, tr_cal["y_win"].to_numpy())
        p_raw = m_win.predict(te[win_feats])
        p_win = iso.predict(p_raw)

        # ---- totals & spread (completed matches only) ---------------------
        trt = tr[tr["y_total_games"].notna()]
        m_tot = _fit(REG_PARAMS, trt, trt["y_total_games"].to_numpy(), reg_feats)
        m_set = _fit(REG_PARAMS, trt, trt["y_total_sets"].to_numpy(), reg_feats)
        m_spr = _fit(REG_PARAMS, trt, trt["y_spread"].to_numpy(), reg_feats)

        preds.append(pd.DataFrame({
            "run_tag": run_tag,
            "match_id": te["match_id"].to_numpy(),
            "seq": te["seq"].to_numpy(),
            "tourney_date": te["tourney_date"].to_numpy(),
            "season": season,
            "surface": te["surface"].astype(str).to_numpy(),
            "level": te["level"].astype(str).to_numpy(),
            # Carried through so a pooled backtest can still be reported one
            # tour at a time -- the fit is shared, the reporting is not.
            "tour": te["tour"].astype(str).to_numpy(),
            "is_challenger": te["is_challenger"].to_numpy(),
            "p1_id": te["p1_id"].to_numpy(),
            "p2_id": te["p2_id"].to_numpy(),
            "y_win": te["y_win"].to_numpy(),
            "p_win": p_win,
            "p_win_raw": p_raw,
            "elo_prob": te["elo_prob"].to_numpy(),
            "y_total": te["y_total_games"].to_numpy(),
            "pred_total": m_tot.predict(te[reg_feats]),
            "y_total_sets": te["y_total_sets"].to_numpy(),
            "pred_total_sets": m_set.predict(te[reg_feats]),
            "y_spread": te["y_spread"].to_numpy(),
            "pred_spread": m_spr.predict(te[reg_feats]),
        }))
        log.info("  season %s: train=%d test=%d", season, len(tr), len(te))

    out = pd.concat(preds, ignore_index=True)
    out.to_parquet(ARTIFACTS / f"backtest_{tour}.parquet", index=False)

    con = connect()
    con.execute("DELETE FROM backtest WHERE run_tag=?", (run_tag,))
    cols = ["run_tag", "match_id", "seq", "tourney_date", "surface", "level",
            "is_challenger", "p1_id", "p2_id", "y_win", "p_win", "y_total",
            "pred_total", "y_spread", "pred_spread", "tour"]
    keep = out[cols].copy()
    keep["close_p1"] = np.nan
    keep["close_p2"] = np.nan
    keep.to_sql("backtest", con, if_exists="append", index=False)
    con.commit()
    con.close()

    return summarize(out, t0)


def summarize(out: pd.DataFrame, t0: float | None = None) -> dict:
    y, p = out["y_win"].to_numpy(), out["p_win"].to_numpy()
    res = {"winner_overall": classification_metrics(y, p)}

    base = out["elo_prob"].to_numpy()
    ok = ~np.isnan(base)
    res["winner_elo_baseline"] = classification_metrics(y[ok], base[ok])

    t = out.dropna(subset=["y_total"])
    res["totals_games"] = regression_metrics(
        t["y_total"].to_numpy(), t["pred_total"].to_numpy())
    ts = out.dropna(subset=["y_total_sets"])
    res["totals_sets"] = regression_metrics(
        ts["y_total_sets"].to_numpy(), ts["pred_total_sets"].to_numpy())
    s = out.dropna(subset=["y_spread"])
    res["spread"] = regression_metrics(
        s["y_spread"].to_numpy(), s["pred_spread"].to_numpy())

    if t0:
        res["seconds"] = round(time.time() - t0, 1)
    return res


def train_production_models(tour: str = "pooled") -> dict:
    """Fit final models on all available history for live prediction."""
    df = load_features(tour)
    df, abs_cols = add_abs(df)
    win_feats = list(FEATURE_COLS)
    reg_feats = list(FEATURE_COLS) + abs_cols

    # Same 85/15 split the backtest measures, so production is the setup that
    # was evaluated. The calibrator is fitted on the held-out tail against a
    # model that has not seen it; the *served* winner model is then refit on
    # every row. Fitting only the first 90% left the model blind to the last
    # 2+ years (it stopped at 2024-07) while the calibrator was the only thing
    # that saw them.
    cut = int(len(df) * 0.85)
    fit, cal = df.iloc[:cut], df.iloc[cut:]
    m_cal = _fit(WIN_PARAMS, fit, fit["y_win"].to_numpy(), win_feats,
                 rounds=WIN_ROUNDS)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
    iso.fit(m_cal.predict(cal[win_feats]), cal["y_win"].to_numpy())
    m_win = _fit(WIN_PARAMS, df, df["y_win"].to_numpy(), win_feats,
                 rounds=WIN_ROUNDS)

    trt = df[df["y_total_games"].notna()]
    m_tot = _fit(REG_PARAMS, trt, trt["y_total_games"].to_numpy(), reg_feats)
    m_set = _fit(REG_PARAMS, trt, trt["y_total_sets"].to_numpy(), reg_feats)
    m_spr = _fit(REG_PARAMS, trt, trt["y_spread"].to_numpy(), reg_feats)

    import pickle
    with open(ARTIFACTS / f"models_{tour}.pkl", "wb") as fh:
        pickle.dump({
            "winner": m_win, "calibrator": iso, "totals": m_tot,
            "totals_sets": m_set, "spread": m_spr,
            "win_feats": win_feats, "reg_feats": reg_feats,
            "abs_features": ABS_FEATURES,
            "trained_through": int(df["tourney_date"].max()),
            "n_train": len(df), "tour": tour,
        }, fh)

    imp = pd.DataFrame({
        "feature": m_win.feature_name(),
        "gain": m_win.feature_importance("gain"),
    }).sort_values("gain", ascending=False)
    imp.to_csv(ARTIFACTS / f"winner_feature_importance_{tour}.csv", index=False)
    return {"tour": tour, "n_train": len(df),
            "top_features": imp.head(15).to_dict("records")}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import sys
    tours = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--tour=")]
    for t in (tours or ["pooled"]):
        print(f"===== {t} =====")
        res = run_backtest(tour=t)
        print(json.dumps(res, indent=2, default=str))
        prod = train_production_models(tour=t)
        print(f"\nproduction models ({t}) trained on", prod["n_train"], "rows")
        for r in prod["top_features"][:12]:
            print(f"  {r['feature']:<28} {r['gain']:,.0f}")
