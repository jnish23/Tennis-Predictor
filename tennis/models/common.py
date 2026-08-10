"""Shared feature contract, splitting and metrics for all three targets.

All three models read the same feature frame; only the label and objective
differ. FEATURE_COLS is the single allow-list of model inputs -- nothing outside
it reaches a model, which is what makes the leakage test in
tests/test_leakage.py meaningful.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Columns that exist in the feature frame but must never be model inputs.
LABEL_COLS = [
    "y_win", "y_total_games", "y_total_sets", "y_spread", "total_games",
    "total_sets", "game_margin", "totals_usable", "status", "minutes",
]
ID_COLS = ["match_id", "seq", "tourney_date", "p1_id", "p2_id"]

PER_PLAYER = [
    "elo", "elo_n", "elo_surf", "elo_surf_n", "winrate_10", "winrate_25",
    "winrate_50", "surf_winrate", "career_n", "career_winrate", "spw", "rpw",
    "first_in", "first_won", "second_won", "ace_rate", "df_rate", "bp_saved",
    "hold_pct", "break_pct", "serve_edge", "avg_games_per_set",
    "avg_total_games", "avg_margin", "rest_days", "matches_14d", "matches_30d",
    "minutes_14d", "rank", "rank_points", "age", "ht", "seed", "lefty",
    # exponentially weighted form (smooth decay, no 25-match cliff)
    "ew_winrate_fast", "ew_winrate_slow", "ew_spw_fast", "ew_spw_slow",
    "ew_rpw_fast", "ew_rpw_slow", "ew_margin_fast", "ew_margin_slow",
    # context for the raw rates above: who they were compiled against
    "opp_elo_25", "opp_elo_10",
    # clutch and surface continuity
    "decider_winrate", "decider_n", "tb_winrate", "tb_n", "surf_streak",
]
DIFFS = [
    "d_elo", "d_elo_surf", "d_winrate_10", "d_winrate_25", "d_winrate_50",
    "d_surf_winrate", "d_spw", "d_rpw", "d_hold_pct", "d_break_pct",
    "d_serve_edge", "d_first_won", "d_second_won", "d_ace_rate",
    "d_career_winrate", "d_rest_days", "d_matches_14d", "d_avg_total_games",
    "d_avg_margin", "d_avg_games_per_set", "d_rank", "d_log_rank",
    "d_rank_points", "d_age", "d_ht", "d_h2h",
    "d_ew_winrate_fast", "d_ew_winrate_slow", "d_ew_spw_fast", "d_ew_spw_slow",
    "d_ew_rpw_fast", "d_ew_rpw_slow", "d_ew_margin_fast", "d_ew_margin_slow",
    "d_opp_elo_25", "d_opp_elo_10", "d_decider_winrate", "d_tb_winrate",
    "d_surf_streak",
]
CONTEXT = [
    "best_of", "round_idx", "draw_size", "indoor", "is_challenger",
    "h2h_p1", "h2h_p2", "h2h_n", "h2h_surf_p1", "h2h_surf_p2",
    "elo_prob", "elo_surf_prob", "log_p1_rank", "log_p2_rank",
]
CATEGORICAL = ["surface", "level"]

FEATURE_COLS: list[str] = (
    [f"p1_{c}" for c in PER_PLAYER]
    + [f"p2_{c}" for c in PER_PLAYER]
    + DIFFS
    + CONTEXT
    + CATEGORICAL
)

# Features that are inherently orientation-symmetric matter for the totals
# model: the total does not depend on which player we called p1.
SYMMETRIC_DROP = ["d_elo", "d_rank", "d_h2h"]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce dtypes so LightGBM sees numerics and proper categoricals."""
    out = df.copy()
    for c in CATEGORICAL:
        out[c] = out[c].astype("category")
    for c in FEATURE_COLS:
        if c in CATEGORICAL:
            continue
        if c not in out.columns:
            out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def walk_forward_folds(
    df: pd.DataFrame,
    *,
    start_season: int = 2010,
    min_train_seasons: int = 6,
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """Expanding-window folds split strictly by date, one per season.

    Fold k trains on every match played before season S and scores every match
    in season S. No shuffling and no random splits: a training row is always
    strictly earlier than every row it is used to predict.
    """
    seasons = np.sort(df["tourney_date"].to_numpy() // 10000)
    uniq = [s for s in np.unique(seasons) if s >= start_season]
    date = df["tourney_date"].to_numpy() // 10000
    folds = []
    for s in uniq:
        train = np.where(date < s)[0]
        test = np.where(date == s)[0]
        if len(train) == 0 or len(test) == 0:
            continue
        if (s - date.min()) < min_train_seasons:
            continue
        folds.append((train, test, int(s)))
    return folds


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-15) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def classification_metrics(y: np.ndarray, p: np.ndarray, thresh: float = 0.5) -> dict:
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    )

    yhat = (p >= thresh).astype(int)
    return {
        "n": int(len(y)),
        "log_loss": round(log_loss(y, p), 5),
        "brier": round(brier(y, p), 5),
        "accuracy": round(float(accuracy_score(y, yhat)), 5),
        "auc": round(float(roc_auc_score(y, p)), 5) if len(np.unique(y)) > 1 else np.nan,
        "precision": round(float(precision_score(y, yhat, zero_division=0)), 5),
        "recall": round(float(recall_score(y, yhat, zero_division=0)), 5),
        "f1": round(float(f1_score(y, yhat, zero_division=0)), 5),
    }


def regression_metrics(y: np.ndarray, yhat: np.ndarray) -> dict:
    err = yhat - y
    return {
        "n": int(len(y)),
        "mae": round(float(np.mean(np.abs(err))), 4),
        "rmse": round(float(np.sqrt(np.mean(err ** 2))), 4),
        "bias": round(float(np.mean(err)), 4),
        "r2": round(float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)), 5),
    }


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        rows.append({
            "bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}",
            "n": int(m.sum()),
            "pred_mean": round(float(p[m].mean()), 4),
            "actual": round(float(y[m].mean()), 4),
        })
    return pd.DataFrame(rows)
