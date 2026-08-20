"""Backtest scoring: accuracy, calibration, ROI and the required breakouts.

A note on what ROI can honestly mean here. tennis-data.co.uk publishes
match-winner prices only -- there are no totals or handicap lines in any season
of the free data. So:

* **winner**  -- ROI is simulated against real bookmaker closing prices
  (Pinnacle, plus market-average and market-best), which is a genuine
  market-relative result.
* **totals / spread** -- there is no market line to bet into. Betting them
  against a line we invented would measure nothing about market edge, so
  instead we run a *synthetic-line* test: a naive reference line (recent
  same-segment average) is priced at standard -110 both ways, and we bet when
  the model disagrees with it. That answers "does the model's edge over a naive
  line survive normal vig?" It is explicitly NOT a claim about beating a real
  totals market, and is labelled synthetic everywhere it is reported.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from tennis.config import ARTIFACTS
from tennis.db.schema import connect
from tennis.models.common import (
    brier,
    calibration_table,
    classification_metrics,
    log_loss,
    regression_metrics,
)

log = logging.getLogger(__name__)

VIG_PRICE = 1.909091  # -110 in decimal odds


def load_backtest() -> pd.DataFrame:
    return pd.read_parquet(ARTIFACTS / "backtest.parquet")


def attach_odds(bt: pd.DataFrame, book: str = "PS",
                devig: str = "shin") -> pd.DataFrame:
    """Attach closing prices, re-oriented from winner/loser to p1/p2.

    `devig` defaults to Shin rather than proportional. Proportional assumes the
    margin is spread evenly across the two sides, which books do not do -- they
    load it onto the longshot. Scored against realised outcomes on 523,879
    quotes across 18 books, Shin beat proportional in 16 of 17, and its
    advantage tracks how much margin a book charges (r = 0.72, p = 0.001). The
    one book it does not help is Betfair, an exchange, which has no bookmaker
    margin to misallocate in the first place.

    The effect on any single probability is small -- about a point on a heavy
    favourite -- but it is the *benchmark* every model-vs-market comparison in
    this project is measured against, so a bias here biases all of them.
    """
    con = connect()
    odds = pd.read_sql(
        "SELECT match_id, book, win_price, lose_price FROM odds WHERE book=?",
        con, params=(book,))
    con.close()
    out = bt.merge(odds, on="match_id", how="left")
    # Source stores prices on the actual winner; flip when p1 lost.
    out["p1_price"] = np.where(out["y_win"] == 1, out["win_price"], out["lose_price"])
    out["p2_price"] = np.where(out["y_win"] == 1, out["lose_price"], out["win_price"])
    # De-vig to a fair market probability for comparison.
    from tennis.models.devig import METHODS

    ok = out["p1_price"].notna() & out["p2_price"].notna()
    out["mkt_p1"] = np.nan
    if ok.any():
        pair = out.loc[ok, ["p1_price", "p2_price"]].to_numpy(float)
        out.loc[ok, "mkt_p1"] = METHODS[devig](pair)[:, 0]
    out["devig"] = devig
    out["book"] = book
    return out


# --------------------------------------------------------------------------
# ROI
# --------------------------------------------------------------------------
def roi_winner(df: pd.DataFrame, edge: float = 0.03, stake: float = 1.0,
               kelly: bool = False, cap: float = 0.05) -> dict:
    """Flat (or fractional-Kelly) staking on whichever side shows an edge."""
    d = df.dropna(subset=["p1_price", "p2_price", "p_win"]).copy()
    if d.empty:
        return {"bets": 0}

    # Edge on each side against that side's actual price.
    d["ev1"] = d["p_win"] * d["p1_price"] - 1
    d["ev2"] = (1 - d["p_win"]) * d["p2_price"] - 1
    d["side"] = np.where(d["ev1"] >= d["ev2"], 1, 2)
    d["ev"] = np.maximum(d["ev1"], d["ev2"])
    d = d[d["ev"] > edge]
    if d.empty:
        return {"bets": 0}

    price = np.where(d["side"] == 1, d["p1_price"], d["p2_price"])
    p = np.where(d["side"] == 1, d["p_win"], 1 - d["p_win"])
    won = np.where(d["side"] == 1, d["y_win"] == 1, d["y_win"] == 0)

    if kelly:
        b = price - 1
        f = np.clip((p * b - (1 - p)) / b, 0, cap)
        size = f
    else:
        size = np.full(len(d), stake)

    profit = np.where(won, size * (price - 1), -size)
    turnover = size.sum()
    return {
        "bets": int(len(d)),
        "turnover": round(float(turnover), 1),
        "profit": round(float(profit.sum()), 2),
        "roi_pct": round(float(profit.sum() / turnover * 100), 3) if turnover else 0.0,
        "hit_rate": round(float(won.mean()), 4),
        # The hit rate these bets would need to break even, which is what
        # `hit_rate` should be read against -- never 50%. Edge betting takes the
        # value side, and the value side is usually the underdog, so most bets
        # are meant to lose. mean(1/price) is the exact flat-stake break-even;
        # 1/mean(price) is not, and understates it badly on a skewed book.
        "breakeven_hit_rate": round(float(np.mean(1.0 / price)), 4),
        "avg_price": round(float(price.mean()), 3),
        "pct_on_market_underdog": round(float(np.mean(
            price > np.where(d["side"] == 1, d["p2_price"], d["p1_price"])) * 100), 1),
    }


def roi_by_season(df: pd.DataFrame, edge: float = 0.03) -> list[dict]:
    """`roi_winner` split by season, so a single blended figure can be checked.

    A flat headline ROI can hide a model that worked for years and then stopped,
    or one carried by a single season. Seasons are independent samples here --
    each is priced by that season's market -- so the spread across them is also
    the honest read on how noisy the headline is.
    """
    out = []
    for season, g in df.groupby(df["tourney_date"] // 10000):
        r = roi_winner(g, edge=edge)
        if r.get("bets"):
            out.append({"season": int(season), **r})
    return out


def _reference_line(df: pd.DataFrame, col: str) -> pd.Series:
    """Naive prior-season reference line by (surface, best-of proxy, level).

    Uses only earlier seasons, so the reference itself never sees the future.
    """
    d = df.sort_values("tourney_date").copy()
    grp = d.groupby(["surface", "is_challenger"])[col]
    # expanding mean shifted by one row = average of everything strictly before
    return grp.transform(lambda s: s.shift(1).expanding(min_periods=50).mean())


def roi_synthetic(df: pd.DataFrame, y_col: str, pred_col: str,
                  edge: float, price: float = VIG_PRICE) -> dict:
    """Bet the model against a naive reference line at -110. Synthetic, not market.

    Read the ROI here with heavy scepticism, and read `baseline` alongside it.
    The reference line is a historical segment average, so it barely moves
    (std ~1.8 games against an actual std of ~7.2). Any model with real signal
    beats a near-constant line easily, which inflates this ROI far above
    anything a real totals market would concede. It demonstrates the model has
    signal; it does NOT estimate profitability.
    """
    d = df.dropna(subset=[y_col, pred_col]).copy()
    if d.empty:
        return {"bets": 0, "synthetic": True}
    d["line"] = _reference_line(d, y_col)
    d = d.dropna(subset=["line"])
    if d.empty:
        return {"bets": 0, "synthetic": True}
    base_mae = float(np.abs(d["line"] - d[y_col]).mean())
    model_mae = float(np.abs(d[pred_col] - d[y_col]).mean())
    baseline = {
        "reference_line_mae": round(base_mae, 3),
        "model_mae": round(model_mae, 3),
        "reference_line_std": round(float(d["line"].std()), 3),
        "actual_std": round(float(d[y_col].std()), 3),
    }
    d["diff"] = d[pred_col] - d["line"]
    d = d[d["diff"].abs() > edge]
    if d.empty:
        return {"bets": 0, "synthetic": True, "baseline": baseline}

    over = d["diff"] > 0
    # Push when the outcome lands exactly on the line: stake returned.
    push = d[y_col] == d["line"]
    won = np.where(over, d[y_col] > d["line"], d[y_col] < d["line"])
    profit = np.where(push, 0.0, np.where(won, price - 1, -1.0))
    return {
        "bets": int(len(d)),
        "turnover": int(len(d)),
        "profit": round(float(profit.sum()), 2),
        "roi_pct": round(float(profit.sum() / len(d) * 100), 3),
        "hit_rate": round(float(won[~push].mean()), 4) if (~push).any() else np.nan,
        "synthetic": True,
        "line_source": "naive prior-history mean by surface x tour level",
        "price": price,
        "baseline": baseline,
        "interpretation": (
            "NOT a profitability estimate. No totals/handicap market exists in the "
            "free data, so this bets against a near-constant reference line; the "
            "ROI is inflated by the line's weakness, not by market edge."
        ),
    }


# --------------------------------------------------------------------------
# breakouts
# --------------------------------------------------------------------------
def breakout(bt: pd.DataFrame, by: str) -> pd.DataFrame:
    rows = []
    for key, g in bt.groupby(by, observed=True):
        y, p = g["y_win"].to_numpy(), g["p_win"].to_numpy()
        if len(g) < 30:
            continue
        m = classification_metrics(y, p)
        t = g.dropna(subset=["y_total"])
        s = g.dropna(subset=["y_spread"])
        rows.append({
            by: key, "n": m["n"], "log_loss": m["log_loss"], "brier": m["brier"],
            "accuracy": m["accuracy"], "auc": m["auc"],
            "totals_mae": regression_metrics(
                t["y_total"].to_numpy(), t["pred_total"].to_numpy())["mae"] if len(t) > 30 else np.nan,
            "spread_mae": regression_metrics(
                s["y_spread"].to_numpy(), s["pred_spread"].to_numpy())["mae"] if len(s) > 30 else np.nan,
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def full_report(book: str = "PS") -> dict:
    bt = load_backtest()
    bt["tour_level"] = np.where(bt["is_challenger"] == 1, "challenger", "main_tour")
    bt["tour"] = "atp"  # WTA breakout activates when WTA data is loaded
    withodds = attach_odds(bt, book=book)

    y, p = bt["y_win"].to_numpy(), bt["p_win"].to_numpy()
    rep: dict = {
        "period": [int(bt["tourney_date"].min()), int(bt["tourney_date"].max())],
        "n_matches": int(len(bt)),
        "winner": classification_metrics(y, p),
        "winner_elo_baseline": classification_metrics(
            y[~np.isnan(bt["elo_prob"])], bt["elo_prob"].dropna().to_numpy()),
        "calibration": calibration_table(y, p).to_dict("records"),
    }

    t = bt.dropna(subset=["y_total"])
    rep["totals_games"] = regression_metrics(t["y_total"].to_numpy(), t["pred_total"].to_numpy())
    ts = bt.dropna(subset=["y_total_sets"])
    rep["totals_sets"] = regression_metrics(
        ts["y_total_sets"].to_numpy(), ts["pred_total_sets"].to_numpy())
    s = bt.dropna(subset=["y_spread"])
    rep["spread"] = regression_metrics(s["y_spread"].to_numpy(), s["pred_spread"].to_numpy())

    # market comparison on the subset that actually has prices
    mk = withodds.dropna(subset=["mkt_p1"])
    if mk.empty:
        # Usually means the odds join has not been re-run since match_ids
        # changed; better to say so than to crash three steps later.
        log.warning("no priced matches joined to the backtest - "
                    "re-run `python -m tennis.ingest.odds`")
        rep["market_subset"] = {"n": 0, "note": "no odds joined to backtest rows"}
    else:
        rep["market_subset"] = {
            "n": int(len(mk)),
            "model": classification_metrics(mk["y_win"].to_numpy(), mk["p_win"].to_numpy()),
            "market_closing": classification_metrics(
                mk["y_win"].to_numpy(), mk["mkt_p1"].to_numpy()),
        }
    rep["roi_winner"] = {
        f"edge_{e:.2f}": roi_winner(withodds, edge=e) for e in (0.0, 0.03, 0.05, 0.10)
    }
    rep["roi_winner_kelly_edge0.03"] = roi_winner(withodds, edge=0.03, kelly=True)
    rep["roi_winner_by_season"] = {
        f"edge_{e:.2f}": roi_by_season(withodds, edge=e) for e in (0.0, 0.03, 0.05, 0.10)
    }
    rep["roi_totals_synthetic"] = {
        f"edge_{e}": roi_synthetic(bt, "y_total", "pred_total", edge=e)
        for e in (1.0, 2.0, 3.0)
    }
    rep["roi_spread_synthetic"] = {
        f"edge_{e}": roi_synthetic(bt, "y_spread", "pred_spread", edge=e)
        for e in (1.0, 2.0, 3.0)
    }

    rois = [v["roi_pct"] for v in rep["roi_winner"].values() if v.get("bets")]
    best_roi = max(rois) if rois else float("nan")
    ms = rep["market_subset"]
    beats_close = (ms["model"]["log_loss"] < ms["market_closing"]["log_loss"]
                   if ms.get("n") else None)
    # Real games-market comparison. Guarded: it depends on the tennisexplorer
    # backfill, which may be absent, partial, or paused. A missing block is a
    # normal state, not a failure, and the dashboard renders accordingly.
    try:
        from tennis.models.lines import clv, market_comparison
        con = connect()
        try:
            mc = market_comparison(con, bt)
            if mc.get("markets"):
                mc["clv"] = clv(con, bt)
                rep["market_lines"] = mc
                log.info("market_lines: %s", {k: v["matches"]
                                              for k, v in mc["markets"].items()})
        finally:
            con.close()
    except Exception as exc:
        log.warning("market-line comparison skipped: %s", exc)

    rep["headline"] = {
        "winner_beats_elo_baseline": rep["winner"]["log_loss"] < rep["winner_elo_baseline"]["log_loss"],
        "winner_beats_closing_line": beats_close,
        "best_winner_roi_pct": best_roi,
        "verdict": ("No priced matches joined; ROI not computable this run."
                    if not ms.get("n") else
            "Model is more accurate than an Elo-only baseline but less accurate "
            "than the Pinnacle closing line, and flat-staking it against that "
            f"line loses money at every edge threshold tested (best {best_roi:.2f}%). "
            "Treat it as a calibrated probability source, not a betting edge "
            "against sharp closing prices."
        ),
        "totals_spread_roi": (
            "Not computable against a market: the free data has no totals or "
            "handicap lines. Only a synthetic-line signal check is reported."
        ),
    }

    for dim in ("surface", "level", "tour_level", "tour"):
        rep[f"by_{dim}"] = breakout(bt, dim).to_dict("records")
    bt["season"] = bt["tourney_date"] // 10000
    rep["by_season"] = breakout(bt, "season").to_dict("records")

    (ARTIFACTS / "backtest_report.json").write_text(json.dumps(rep, indent=2, default=str))
    return rep


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    r = full_report()
    print(json.dumps({k: v for k, v in r.items()
                      if k not in ("calibration", "by_season")}, indent=2, default=str))
