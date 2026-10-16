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
by season, a held-out 15% tail for the isotonic calibrator, the production
hyper-parameters. Differences are reported *paired*: the same test rows under
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

from sklearn.isotonic import IsotonicRegression  # noqa: E402

from tennis.config import ARTIFACTS  # noqa: E402
from tennis.models.common import (CATEGORICAL, FEATURE_COLS, prepare,  # noqa: E402
                                  walk_forward_folds)
from tennis.models.train import (REG_PARAMS, WIN_PARAMS, WIN_ROUNDS, _fit,  # noqa: E402
                                 _playable, add_abs)

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
    folds = walk_forward_folds(df, start_season=START_SEASON)
    rows = []
    t0 = time.time()
    for train_idx, test_idx, season in folds:
        tr, te = df.iloc[train_idx], df.iloc[test_idx]
        rec = pd.DataFrame({
            "match_id": te["match_id"].to_numpy(), "season": season,
            "tour": te["tour"].astype(str).to_numpy(),
            "y_win": te["y_win"].to_numpy(),
            "y_total": te["y_total_games"].to_numpy(),
            "y_spread": te["y_spread"].to_numpy(),
        })
        if "win" not in reuse:
            cut = int(len(tr) * 0.85)
            fit, cal = tr.iloc[:cut], tr.iloc[cut:]
            m = _fit(WIN_PARAMS, fit, fit["y_win"].to_numpy(), win_feats,
                     rounds=WIN_ROUNDS)
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
            iso.fit(m.predict(cal[win_feats]), cal["y_win"].to_numpy())
            rec["p_win"] = iso.predict(m.predict(te[win_feats]))
        trt = tr[tr["y_total_games"].notna()]
        rec["pred_total"] = _fit(REG_PARAMS, trt, trt["y_total_games"].to_numpy(),
                                 reg_feats).predict(te[reg_feats])
        rec["pred_spread"] = _fit(REG_PARAMS, trt, trt["y_spread"].to_numpy(),
                                  reg_feats).predict(te[reg_feats])
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
    by_season = pd.Series(d.to_numpy()).groupby(seasons).mean()
    boots = [d.to_numpy()[rng.integers(0, len(d), len(d))].mean() for _ in range(300)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"vs": a, "variant": b, "target": target, "n": len(d),
            "level": round(float(ra.dropna().mean()), 5),
            "delta": round(float(d.mean()), 5), "ci_lo": round(float(lo), 5),
            "ci_hi": round(float(hi), 5),
            "seasons_better": f"{int((by_season < 0).sum())}/{len(by_season)}"}


def report() -> None:
    rng = np.random.default_rng(0)
    have = {p.stem for p in OUT.glob("*.parquet")}
    plan = [("V0", "V1a"), ("V1a", "V1"), ("V0", "V1"), ("V1", "V2"), ("V1", "V3"),
            ("V1", "V4"), ("V1", "V5"), ("V1", "V6"), ("V1", "ALL"), ("V0", "ALL")]
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
