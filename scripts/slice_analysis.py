"""Where, if anywhere, does the model beat the market?

The trap with slicing a backtest is that every slice is another lottery ticket:
test 40 and a few will show a profit by luck alone. So nothing here is judged
on realised ROI. Each slice is scored on

  1. **dLL**: model log loss minus the Shin-devigged Pinnacle *closing* line's,
     on the same matches, with a paired t-test. Does the model hold information
     the market lacks, or did some underdogs happen to win?
  2. **CLV**: when betting the model's side at Pinnacle's *opening* price, how
     far does the devigged market move toward us by the close?
  3. **ROI** at the opening price (edge > 3%), reported last, with its t.

and split chronologically: a slice is *discovered* on 2013-2022 and then tested
once on 2023-2026, untouched. The honest summary is not "which slices passed"
but "how many passed discovery, how many of those replicated, and how many would
have by chance", plus Benjamini-Hochberg q-values over the whole family.

Every slice mask is a function of the model, the market and pre-match features
only -- never the outcome -- so conditioning on a slice cannot bias dLL.

Run:  python scripts/slice_analysis.py [--tour=atp] [--refresh]
      python scripts/slice_analysis.py --market=totals|handicap   (games lines)

The games markets have no per-match opening/closing pair in the backtest frame,
so they are scored on dLL alone against Pinnacle's closing line, one line per
match (the one priced nearest 50/50), with discovery 2018-2022 because the
empirical residual model that turns a point forecast into P(over) is itself
fitted before 2018.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tennis.config import ARTIFACTS  # noqa: E402
from tennis.db.schema import connect  # noqa: E402
from tennis.models.devig import METHODS  # noqa: E402
from tennis.models.evaluate import load_backtest  # noqa: E402

MARKET = next((a.split("=")[1] for a in sys.argv if a.startswith("--market=")), "h2h")
TOUR = next((a.split("=")[1] for a in sys.argv if a.startswith("--tour=")), "atp")
OUT = ROOT / "artifacts" / "slice_analysis"
DISC_END = 2022          # discovery 2013..2022, validation 2023..
MIN_N = 250
EDGE = 0.03
EPS = 1e-6

FEATS = ["round_idx", "draw_size", "best_of", "p1_career_n", "p2_career_n",
         "p1_surf_streak", "p2_surf_streak", "p1_rest_days", "p2_rest_days",
         "p1_matches_14d", "p2_matches_14d", "p1_minutes_14d", "p2_minutes_14d",
         "p1_hold_pct", "p2_hold_pct", "d_spw", "d_rpw", "p1_lefty", "p2_lefty",
         "p1_age", "p2_age", "surface", "level", "is_challenger"]


def ll(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def shin_p1(a, b):
    out = np.full(len(a), np.nan)
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.any():
        out[ok] = METHODS["shin"](np.c_[a[ok], b[ok]])[:, 0]
    return out


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def build_frame() -> pd.DataFrame:
    cache = OUT / f"frame_{TOUR}.parquet"
    if cache.exists() and "--refresh" not in sys.argv:
        return pd.read_parquet(cache)
    OUT.mkdir(parents=True, exist_ok=True)
    bt = load_backtest(TOUR)[["match_id", "tourney_date", "season", "y_win", "p_win"]]
    con = connect()
    q = pd.read_sql("""
        SELECT m.match_id, m.p1_is_winner,
          MAX(CASE WHEN q.side='p1' THEN q.price_open  END) k1o,
          MAX(CASE WHEN q.side='p2' THEN q.price_open  END) k2o,
          MAX(CASE WHEN q.side='p1' THEN q.price_close END) k1c,
          MAX(CASE WHEN q.side='p2' THEN q.price_close END) k2c
        FROM odds_quotes q JOIN te_matches m USING(te_id)
        WHERE q.market='h2h' AND q.book='Pinnacle' AND m.match_id IS NOT NULL
              AND m.p1_is_winner IS NOT NULL
        GROUP BY m.match_id""", con)
    con.close()
    # tennisexplorer's k1 is the match winner (99.8%); move into the backtest's
    # own p1/p2 via who actually won. The prices were set pre-match, so using
    # the result only to find which column is which leaks nothing.
    d = bt.merge(q, on="match_id")
    w = d.p1_is_winner == 1
    for t in ("o", "c"):
        win, lose = np.where(w, d[f"k1{t}"], d[f"k2{t}"]), np.where(w, d[f"k2{t}"], d[f"k1{t}"])
        d[f"p1_{t}"] = np.where(d.y_win == 1, win, lose)
        d[f"p2_{t}"] = np.where(d.y_win == 1, lose, win)
        bad = ~((d[f"p1_{t}"] > 1.001) & (d[f"p2_{t}"] > 1.001))
        d.loc[bad, [f"p1_{t}", f"p2_{t}"]] = np.nan
    d = d.dropna(subset=["p1_c", "p2_c"])
    d["mkt_c"] = shin_p1(d.p1_c.to_numpy(float), d.p2_c.to_numpy(float))
    d["mkt_o"] = shin_p1(d.p1_o.to_numpy(float), d.p2_o.to_numpy(float))
    d["vig"] = 1 / d.p1_c + 1 / d.p2_c - 1
    feats = pd.read_parquet(ARTIFACTS / f"features_{TOUR}.parquet", columns=["match_id"] + FEATS)
    d = d.merge(feats, on="match_id", how="left").drop(columns="p1_is_winner")
    d.to_parquet(cache, index=False)
    return d


def derive(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["ll_model"], d["ll_mkt"] = ll(d.y_win, d.p_win), ll(d.y_win, d.mkt_c)
    d["dll"] = d.ll_model - d.ll_mkt
    # bet: the model's best side at the OPENING price, if the edge clears
    ev1, ev2 = d.p_win * d.p1_o - 1, (1 - d.p_win) * d.p2_o - 1
    s1 = ev1 >= ev2
    d["bet"] = np.maximum(ev1, ev2) > EDGE
    d["side1"] = s1
    d["bet_price"] = np.where(s1, d.p1_o, d.p2_o)
    d["bet_won"] = np.where(s1, d.y_win == 1, d.y_win == 0)
    d["profit"] = np.where(d.bet_won, d.bet_price - 1, -1.0)
    d["clv"] = np.where(s1, d.mkt_c - d.mkt_o, (1 - d.mkt_c) - (1 - d.mkt_o)) * 100
    d.loc[~d.bet, ["profit", "clv"]] = np.nan
    # model-backed side (even when below the edge): who is the model with?
    side_p = np.where(d.p_win >= d.mkt_c, d.p1_c, d.p2_c)
    d["backed_close_price"] = np.where(d.p_win >= d.mkt_c, d.p1_c, d.p2_c)
    d["gap"] = (d.p_win - d.mkt_c).abs()
    d["fav_price"] = np.minimum(d.p1_c, d.p2_c)
    d["dog_price"] = np.maximum(d.p1_c, d.p2_c)
    return d


def slices(d: pd.DataFrame) -> dict[str, tuple[str, pd.Series]]:
    """name -> (family, boolean mask). All outcome-independent."""
    g = {}
    f = lambda fam, name, m: g.__setitem__(name, (fam, m.fillna(False).astype(bool)))
    lvl = d.level.astype(str)
    for t in ("challenger", "atp250", "atp500", "masters", "grand_slam"):
        f("tier", f"tier={t}", lvl == t)
    first = 3 + (7 - np.ceil(np.log2(d.draw_size.clip(lower=2))))
    late = d.round_idx.isin([7, 8, 10])
    f("round", "round=first", (d.round_idx == first) & (d.round_idx != 11))
    f("round", "round=QF+", late)
    f("round", "round=middle", ~late & (d.round_idx != first) & (d.round_idx < 11))
    h2h = "p1_c" in d
    mn, mx = np.minimum(d.p1_career_n, d.p2_career_n), np.maximum(d.p1_career_n, d.p2_career_n)
    f("experience", "exp=a_player<20_matches", mn < 20)
    f("experience", "exp=both>100", mn > 100)
    if h2h:
        f("market", "mkt=pickem(both 1.75-2.15)", (d.p1_c.between(1.75, 2.15)) & d.p2_c.between(1.75, 2.15))
        f("market", "mkt=heavy_fav(<1.40)", d.fav_price < 1.40)
        f("market", "mkt=longshot_match(dog>3.0)", d.dog_price > 3.0)
        bp = d.backed_close_price
        f("backed", "backs=favourite(<1.90)", bp < 1.90)
        f("backed", "backs=coinflip(1.90-2.15)", bp.between(1.90, 2.15))
        f("backed", "backs=moderate_dog(2.15-3.0)", (bp > 2.15) & (bp <= 3.0))
        f("backed", "backs=longshot(>3.0)", bp > 3.0)
    f("divergence", "gap<2pts", d.gap < 0.02)
    f("divergence", "gap2-5pts", d.gap.between(0.02, 0.05, inclusive="left"))
    f("divergence", "gap5-10pts", d.gap.between(0.05, 0.10, inclusive="left"))
    f("divergence", "gap>10pts", d.gap >= 0.10)
    if not h2h:
        f("direction", "model_says_OVER/p1-covers-more", d.p_win > d.mkt_c)
        f("direction", "model_says_UNDER/less", d.p_win <= d.mkt_c)
        if "line_ours" in d:
            lays = ((d.p_win > d.mkt_c) & (d.line_ours < 0)) | ((d.p_win < d.mkt_c) & (d.line_ours > 0))
            f("direction", "spread: model lays games (backs the favourite)", lays & (d.line_ours != 0))
            f("direction", "spread: model takes games (backs the dog)", ~lays & (d.line_ours != 0))
    f("vig", "vig<3%", d.vig < 0.03)
    f("vig", "vig3-5%", d.vig.between(0.03, 0.05, inclusive="left"))
    f("vig", "vig>5%", d.vig >= 0.05)
    s1, s2 = d.p1_surf_streak, d.p2_surf_streak
    f("schedule", "surface_switch(one player first match on surface)",
      ((s1 == 0) & (s2 > 0)) | ((s2 == 0) & (s1 > 0)))
    f("schedule", "both_settled(streak>=3)", (s1 >= 3) & (s2 >= 3))
    f("schedule", "layoff(a player rest>30d)", np.maximum(d.p1_rest_days, d.p2_rest_days) > 30)
    f("schedule", "quick_turnaround(rest<=3 & >=5 matches/14d)",
      ((d.p1_rest_days <= 3) & (d.p1_matches_14d >= 5)) | ((d.p2_rest_days <= 3) & (d.p2_matches_14d >= 5)))
    f("schedule", "workload_gap>120min & late round",
      ((d.p1_minutes_14d - d.p2_minutes_14d).abs() > 120) & late)
    f("style", "both_big_servers(hold>85%)", (d.p1_hold_pct > .85) & (d.p2_hold_pct > .85))
    qs, qr = d.d_spw.abs().quantile(.75), d.d_rpw.abs().quantile(.75)
    f("style", "server_vs_returner(opposite d_spw/d_rpw, top quartile)",
      (d.d_spw.abs() >= qs) & (d.d_rpw.abs() >= qr) & (np.sign(d.d_spw) != np.sign(d.d_rpw)))
    diff_hand = d.p1_lefty != d.p2_lefty
    for s in ("Clay", "Grass", "Hard"):
        f("style", f"lefty_vs_righty on {s}", diff_hand & (d.surface.astype(str) == s))
    f("age", "young<21 vs 30+", ((d.p1_age < 21) & (d.p2_age > 30)) | ((d.p2_age < 21) & (d.p1_age > 30)))
    f("age", "age_gap>8y", (d.p1_age - d.p2_age).abs() > 8)
    f("format", "best_of=3", d.best_of == 3)
    f("format", "best_of=5", d.best_of == 5)
    for s in ("Hard", "Clay", "Grass"):
        f("surface", f"surface={s}", d.surface.astype(str) == s)
    return g


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------
def t_of(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 3 or x.std(ddof=1) == 0:
        return len(x), np.nan, np.nan
    return len(x), float(x.mean()), float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def blend_gain(disc: pd.DataFrame, val: pd.DataFrame) -> tuple[float, float]:
    """Weight on the model fitted on discovery, and the validation log-loss gain
    over the market from using it. A positive weight with a gain says the model
    adds information even where it loses to the market alone."""
    def logit(p):
        p = np.clip(p, EPS, 1 - EPS)
        return np.log(p / (1 - p))
    grid = np.linspace(0, 1, 21)
    def loss(df, w):
        z = (1 - w) * logit(df.mkt_c) + w * logit(df.p_win)
        return ll(df.y_win, 1 / (1 + np.exp(-z))).mean()
    if len(disc) < MIN_N or len(val) < 50:
        return np.nan, np.nan
    w = grid[int(np.argmin([loss(disc, x) for x in grid]))]
    return float(w), float(loss(val, 0.0) - loss(val, w))


def evaluate(d: pd.DataFrame, g: dict) -> pd.DataFrame:
    rows = []
    disc_all, val_all = d.season <= DISC_END, d.season > DISC_END
    items = [("ALL", ("all", pd.Series(True, index=d.index)))] + list(g.items())
    for name, (fam, m) in items:
        r = {"slice": name, "family": fam}
        for tag, per in (("disc", disc_all), ("val", val_all)):
            s = d[m & per]
            n, mean, t = t_of(s.dll)
            nb, roi, tr = t_of(s.profit)
            _, clv, tc = t_of(s.clv)
            r.update({f"{tag}_n": n, f"{tag}_dLL": mean, f"{tag}_t": t,
                      f"{tag}_p": float(stats.t.cdf(t, n - 1)) if t == t else np.nan,
                      f"{tag}_bets": nb, f"{tag}_roi": roi, f"{tag}_roi_t": tr,
                      f"{tag}_clv": clv, f"{tag}_clv_t": tc})
        r["blend_w"], r["val_blend_gain"] = blend_gain(d[m & disc_all], d[m & val_all])
        rows.append(r)
    res = pd.DataFrame(rows)
    # Benjamini-Hochberg over the validation dLL p-values of the sliced family.
    body = res[res.slice != "ALL"].dropna(subset=["val_p"]).sort_values("val_p")
    k = len(body)
    q = (body.val_p.to_numpy() * k / np.arange(1, k + 1))[::-1]
    q = np.minimum.accumulate(q)[::-1]
    res.loc[body.index, "val_q"] = np.minimum(q, 1.0)
    return res


def report(res: pd.DataFrame) -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 52)
    allr = res[res.slice == "ALL"].iloc[0]
    print(f"\n=== {TOUR.upper()} vs Pinnacle close | all matches: "
          f"disc n={int(allr.disc_n)} dLL={allr.disc_dLL:+.4f} | "
          f"val n={int(allr.val_n)} dLL={allr.val_dLL:+.4f}  (positive = model worse)")
    body = res[res.slice != "ALL"].copy()
    ok = body[(body.disc_n >= MIN_N) & (body.val_n >= MIN_N)]
    show = ok[["slice", "disc_n", "disc_dLL", "val_n", "val_dLL", "val_t", "val_q",
               "disc_clv", "val_clv", "val_bets", "val_roi", "val_roi_t",
               "blend_w", "val_blend_gain"]].copy()
    print(show.round(4).sort_values("val_dLL").to_string(index=False))

    # The gate from the proposal: discovery t>1.5 on ROI AND dLL<0, then validate.
    gate = ok[(ok.disc_roi_t > 1.5) & (ok.disc_dLL < 0)]
    print(f"\nSlices tested with n>={MIN_N} in both periods: {len(ok)}")
    print(f"Pass the discovery gate (ROI t>1.5 and dLL<0): {len(gate)}")
    if len(gate):
        print(gate[["slice", "disc_dLL", "disc_roi", "val_dLL", "val_roi", "val_roi_t"]]
              .round(4).to_string(index=False))
    # How often does the model beat the market (dLL<0) at all, versus chance?
    sig_disc = (ok.disc_p < .05).sum()
    both = ((ok.disc_p < .05) & (ok.val_p < .05)).sum()
    print(f"dLL<0 at p<.05 in discovery: {sig_disc} (chance alone: {0.05 * len(ok):.1f});"
          f" of those, also p<.05 in validation: {both}")
    print(f"Slices with validation BH q<0.10: {(ok.val_q < .10).sum()}")
    print(f"Slices where the model is significantly WORSE than the market in "
          f"validation (p>.95): {(ok.val_p > .95).sum()} of {len(ok)}")


def lines_frame() -> pd.DataFrame:
    """Games-market rows reshaped to the moneyline frame's column names."""
    from tennis.models import lines
    lines.FIT_BEFORE = 2018
    pred, actual = (("pred_total", "y_total") if MARKET == "totals"
                    else ("pred_spread", "y_spread"))
    con = connect()
    d = lines._frame(MARKET, pred, actual, con, load_backtest(TOUR), book="Pinnacle")
    con.close()
    # a ladder is a menu, not many independent observations: keep the main line
    d = d.assign(dist=(d.p_mkt_ours - 0.5).abs()).sort_values("dist").drop_duplicates("match_id")
    e = pd.DataFrame({"match_id": d.match_id, "season": d.season, "y_win": d.y,
                      "p_win": d.p_model, "mkt_c": d.p_mkt_ours,
                      "vig": d.overround - 1, "line_ours": d.line_ours})
    if MARKET == "handicap":
        e["line_ours"] = d.line_ours
    feats = pd.read_parquet(ARTIFACTS / f"features_{TOUR}.parquet", columns=["match_id"] + FEATS)
    e = e.merge(feats, on="match_id", how="left")
    e["ll_model"], e["ll_mkt"] = ll(e.y_win, e.p_win), ll(e.y_win, e.mkt_c)
    e["dll"] = e.ll_model - e.ll_mkt
    e["gap"] = (e.p_win - e.mkt_c).abs()
    for c in ("profit", "clv"):
        e[c] = np.nan
    return e


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if MARKET != "h2h":
        DISC_END = 2022
        d = lines_frame()
        print(f"{TOUR} {MARKET}: {len(d)} matches, seasons {d.season.min()}-{d.season.max()}")
        res = evaluate(d, slices(d))
        res.to_csv(OUT / f"{MARKET}_{TOUR}.csv", index=False)
        report(res)
        sys.exit()
    d = derive(build_frame())
    print(f"{TOUR}: {len(d)} matches with Pinnacle closing prices, "
          f"{d.season.min()}-{d.season.max()}; {int(d.bet.sum())} bets at the open")
    res = evaluate(d, slices(d))
    res.to_csv(OUT / f"moneyline_{TOUR}.csv", index=False)
    report(res)
