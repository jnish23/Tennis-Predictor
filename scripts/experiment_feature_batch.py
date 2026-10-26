"""Which of the candidate feature changes survive a walk-forward backtest?

Items covered (see the variants table below):

  V0   the model as it stood: best_of NaN everywhere (it was merged in twice and
       came out as best_of_x / best_of_y) and a "tiebreak" win rate that was
       really a match win rate
  V1a  real tiebreak win rate only (best_of still blanked)
  V1   + best_of reaches the model                      -> the new baseline
  V2   V1 with decider / tiebreak / career win rates shrunk toward 0.5
  V3   V1 + tournament speed index (prior seasons only)
  V4   V1 + symmetric serve features (totals and spread models only)
  V5   V1 + opponent-adjusted serve / return EWMAs
  V6   V1 + surface-specific serve / return EWMAs
  ALL  V1 + every group that earned its place (set below after the runs)

Each variant is fitted exactly as `train.run_backtest` fits: expanding window
by season, the booster on every earlier row, a per-tour isotonic calibrator fitted
on the previous seasons' out-of-fold predictions, both orientations scored and
averaged, the production hyper-parameters.

Selection discipline: a variant is judged on seasons before `HOLDOUT_FROM`
(`delta`, `ci_*`), and the seasons after it are shown apart (`holdout_delta`) and
never used to choose. An earlier round chose its feature subset on the whole
2012+ window, so the holdout seasons were not out of sample for it. Differences are reported *paired*: the same test rows under
both models, with a bootstrap over matches, because two models that predict the
same rows disagree far less than their two independent intervals suggest.

Run:  python scripts/experiment_feature_batch.py V1 V2 ...   (names, or "all")
      python scripts/experiment_feature_batch.py --report
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tennis.config import ARTIFACTS  # noqa: E402
from tennis.models.common import (FEATURE_COLS, prepare,  # noqa: E402
                                  walk_forward_folds)
from tennis.models.evaluate import HOLDOUT_FROM  # noqa: E402
from tennis.models.train import (CAL_SEASONS, REG_PARAMS, WIN_PARAMS,  # noqa: E402
                                 WIN_ROUNDS, _calibrator_from, _fit, _playable,
                                 _symmetric_regression, _winner_raw, add_abs)

START_SEASON = 2012
OLD_DIR = Path(sys.argv[sys.argv.index("--old") + 1]) if "--old" in sys.argv else None
OUT = ROOT / "artifacts" / "feature_experiments"

BASE = list(FEATURE_COLS)


def both(names):
    return [f"p{i}_{n}" for n in names for i in (1, 2)]


def swap(feats, pairs):
    m = {}
    for old, new in pairs:
        for pre in ("p1_", "p2_", "d_"):
            m[pre + old] = pre + new
    return [m.get(f, f) for f in feats]


SPEED = ["tspeed_ace", "tspeed_spw", "tspeed_gps", "tspeed_n"]
SYM = ["spw_sum", "hold_prod", "exp_serve_p1", "exp_serve_p2", "exp_serve_sum"]
ADJ = both(["ew_adj_spw_fast", "ew_adj_spw_slow", "ew_adj_rpw_fast",
            "ew_adj_rpw_slow"]) + ["d_ew_adj_spw_slow", "d_ew_adj_rpw_slow"]
SURF = both(["spw_surf", "rpw_surf"]) + ["d_spw_surf", "d_rpw_surf"]
SHRINK = [("decider_winrate", "decider_wr_s"), ("tb_winrate", "tb_wr_s"),
          ("career_winrate", "career_wr_s")]


ADJSURF = both(["adj_spw_surf", "adj_rpw_surf"]) + ["d_adj_spw_surf", "d_adj_rpw_surf"]
SPEED3 = ["tspeed3_spw", "tspeed3_ace", "x_spd_spw", "x_spd_ace", "x_spd_edge"]


def speed_interactions(df):
    """Court speed (trailing 3 seasons) times the serve gap it should amplify."""
    df = df.copy()
    df["x_spd_spw"] = df["tspeed3_spw"] * df["d_spw"]
    df["x_spd_ace"] = df["tspeed3_ace"] * df["d_ace_rate"]
    df["x_spd_edge"] = df["tspeed3_spw"] * df["d_serve_edge"]
    return df


def blank_best_of(df):
    df = df.copy()
    df["best_of"] = np.nan
    return df


# name -> dict(old=bool, win=[...], reg=[...] | None for "same as win", mod=fn,
#              reuse={"win": "V1"} to borrow another variant's predictions)
VARIANTS = {
    "V0": dict(old=True, win=BASE, reg=BASE),
    "V1a": dict(win=BASE, reg=BASE, mod=blank_best_of),
    "V1": dict(win=BASE, reg=BASE),
    "V2": dict(win=swap(BASE, SHRINK), reg=swap(BASE, SHRINK)),
    "V3": dict(win=BASE + SPEED, reg=BASE + SPEED),
    "V4": dict(win=BASE, reg=BASE + SYM, reuse={"win": "V1"}),
    "V5": dict(win=BASE + ADJ, reg=BASE + ADJ),
    "V6": dict(win=BASE + SURF, reg=BASE + SURF),
}
VARIANTS["V7"] = dict(win=BASE + ADJSURF, reg=BASE + ADJSURF)
VARIANTS["V8"] = dict(win=BASE + SPEED3, reg=BASE + SPEED3, mod=speed_interactions)
# Filled in once the runs are in, from the groups that helped.
VARIANTS["ALL"] = dict(win=BASE, reg=BASE)


def load(old: bool) -> pd.DataFrame:
    d = OLD_DIR if old else ARTIFACTS
    frames = [_playable(pd.read_parquet(d / f"features_{t}.parquet"))
              for t in ("atp", "wta")]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["tourney_date", "tour", "seq"]).reset_index(drop=True)
    return prepare(df)


def run(name: str) -> None:
    spec = VARIANTS[name]
    reuse = spec.get("reuse", {})
    df = load(spec.get("old", False))
    if "mod" in spec:
        df = spec["mod"](df)
    df, abs_cols = add_abs(df)
    win_feats = [c for c in spec["win"]]
    reg_feats = [c for c in spec["reg"]] + abs_cols
    for c in set(win_feats + reg_feats):
        assert c in df.columns, f"{name}: missing column {c}"

    OUT.mkdir(parents=True, exist_ok=True)
    folds = walk_forward_folds(df, start_season=START_SEASON - CAL_SEASONS)
    rows, oof = [], []
    t0 = time.time()
    for train_idx, test_idx, season in folds:
        tr, te = df.iloc[train_idx], df.iloc[test_idx]
        scored = season >= START_SEASON
        rec = pd.DataFrame({
            "match_id": te["match_id"].to_numpy(), "season": season,
            "tour": te["tour"].astype(str).to_numpy(),
            "y_win": te["y_win"].to_numpy(),
            "y_total": te["y_total_games"].to_numpy(),
            "y_spread": te["y_spread"].to_numpy(),
        })
        if "win" not in reuse:
            m = _fit(WIN_PARAMS, tr, tr["y_win"].to_numpy(), win_feats,
                     rounds=WIN_ROUNDS)
            raw_f, raw_r = _winner_raw(m, te, win_feats)
            oof.append(pd.DataFrame({"season": season, "tour": rec["tour"],
                                     "y": rec["y_win"], "raw_fwd": raw_f,
                                     "raw_rev": raw_r}))
            if scored:
                cal = _calibrator_from(pd.concat(
                    [o for o in oof if season - CAL_SEASONS <= o["season"].iat[0] < season],
                    ignore_index=True))
                t = rec["tour"].to_numpy()
                rec["p_win"] = (cal.predict(raw_f, t) + 1.0 - cal.predict(raw_r, t)) / 2.0
        if not scored:
            continue
        trt = tr[tr["y_total_games"].notna()]
        rec["pred_total"] = _symmetric_regression(
            _fit(REG_PARAMS, trt, trt["y_total_games"].to_numpy(), reg_feats),
            te, reg_feats)
        rec["pred_spread"] = _symmetric_regression(
            _fit(REG_PARAMS, trt, trt["y_spread"].to_numpy(), reg_feats),
            te, reg_feats, signed=True)
        rows.append(rec)
        print(f"  {name} season {season} done ({time.time() - t0:.0f}s)", flush=True)
    out = pd.concat(rows, ignore_index=True)
    if "win" in reuse:
        ref = pd.read_parquet(OUT / f"{reuse['win']}.parquet")[["match_id", "p_win"]]
        out = out.merge(ref, on="match_id", how="left")
    out.to_parquet(OUT / f"{name}.parquet", index=False)


# --------------------------------------------------------------------------
def _loss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def per_row(res: pd.DataFrame, target: str) -> pd.Series:
    """Per-row loss indexed by match_id; NaN where the label is missing."""
    r = res.set_index("match_id")
    if target == "win":
        return pd.Series(_loss(r["y_win"].to_numpy(), r["p_win"].to_numpy()), r.index)
    if target == "total":
        return (r["pred_total"] - r["y_total"]).abs()
    return (r["pred_spread"] - r["y_spread"]).abs()


def compare(a: str, b: str, target: str, rng) -> dict:
    """b minus a, paired over the same rows (negative = b better, all targets)."""
    ra = per_row(pd.read_parquet(OUT / f"{a}.parquet"), target)
    rb = per_row(pd.read_parquet(OUT / f"{b}.parquet"), target)
    d = (rb - ra).dropna()
    meta = pd.read_parquet(OUT / f"{b}.parquet").drop_duplicates("match_id").set_index("match_id")
    seasons = meta.loc[d.index, "season"].to_numpy()
    # Choose on the development seasons only; the holdout is reported, not used.
    dev = seasons < HOLDOUT_FROM
    d_dev = d[dev]
    by_season = pd.Series(d_dev.to_numpy()).groupby(seasons[dev]).mean()
    boots = [d_dev.to_numpy()[rng.integers(0, len(d_dev), len(d_dev))].mean()
             for _ in range(300)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    hold = d[~dev]
    return {"vs": a, "variant": b, "target": target, "n": len(d_dev),
            "level": round(float(ra.dropna().loc[ra.dropna().index.isin(d_dev.index)].mean()), 5),
            "delta": round(float(d_dev.mean()), 5), "ci_lo": round(float(lo), 5),
            "ci_hi": round(float(hi), 5),
            "seasons_better": f"{int((by_season < 0).sum())}/{len(by_season)}",
            "holdout_n": len(hold),
            "holdout_delta": round(float(hold.mean()), 5) if len(hold) else np.nan}


def report() -> None:
    rng = np.random.default_rng(0)
    have = {p.stem for p in OUT.glob("*.parquet")}
    plan = [("V0", "V1a"), ("V1a", "V1"), ("V0", "V1"), ("V1", "V2"), ("V1", "V3"),
            ("V1", "V4"), ("V1", "V5"), ("V1", "V6"), ("V1", "V7"), ("V1", "V8"), ("V1", "ALL"), ("V0", "ALL")]
    rows = []
    for a, b in plan:
        if a in have and b in have:
            for t in ("win", "total", "spread"):
                rows.append(compare(a, b, t, rng))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.to_string(index=False))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--") and a != str(OLD_DIR)]
    if "--report" in sys.argv:
        report()
    else:
        for n in (list(VARIANTS) if args == ["all"] else args):
            print("=== variant", n, flush=True)
            run(n)
