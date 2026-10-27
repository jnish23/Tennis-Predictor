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
from tennis.ingest.parse import parse_score
from tennis.models.common import (
    calibration_table,
    classification_metrics,
    regression_metrics,
    season_of,
)
from tennis.models.stats import cluster_bootstrap_ratio, tourney_key_of

log = logging.getLogger(__name__)

VIG_PRICE = 1.909091  # -110 in decimal odds

# Hyperparameters were tuned on 2019-2022 folds and reported on 2023+, but the
# Elo constants and the feature subset were chosen by scoring the whole 2010+
# walk-forward window, so no season already in the backtest is untouched. Matches
# after this date are: nothing about the model has been selected on them. The
# report breaks them out so the claim can be checked as they accrue. Moving this
# date forward is the only legitimate way to retune.
HOLDOUT_FROM = 2023
SELECTION_FROZEN_ON = 20261026

# How each book settles a moneyline bet when a match ends in retirement, as the
# fewest completed sets after which the bet stands. Below it the stake is
# returned. Pinnacle voids unless a full set was played; every other book here is
# assumed to void all retirements (settling only on a completed match). These are
# the books' published rules as understood when written -- check them before
# trusting an ROI to the second decimal.
RETIREMENT_MIN_SETS = {"PS": 1}


def load_backtest(tour: str = "atp") -> pd.DataFrame:
    """Walk-forward predictions for one tour.

    The production fit is pooled across both tours, but every evaluation here
    is per tour: a blended ATP+WTA log loss describes no tour anyone can bet
    into, and would move whenever the mix of the two changed.
    """
    p = ARTIFACTS / "backtest_pooled.parquet"
    if not p.exists():
        p = ARTIFACTS / f"backtest_{tour}.parquet"
    df = pd.read_parquet(p)
    if "tour" in df.columns:
        df = df[df["tour"] == tour].reset_index(drop=True)
    return df


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
    out = out.merge(_early_finishes(), on="match_id", how="left")
    out["status"] = out["status"].fillna("completed")
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


def completed_sets(score, best_of=None) -> int:
    """Sets actually finished in a score line, ignoring the one cut short.

    The parser counts a set in progress when play stopped ("6-4 3-2 RET" reads
    as two sets for the winner), which is right for a games total and wrong for
    a bookmaker deciding whether the match got far enough to stand.
    """
    done = 0
    for a, b in parse_score(score, best_of).sets:
        hi, lo = max(a, b), min(a, b)
        if (hi >= 6 and hi - lo >= 2) or (hi == 7 and lo in (5, 6)):
            done += 1
    return done


def _early_finishes() -> pd.DataFrame:
    """Status and completed sets for every match that did not finish normally."""
    con = connect()
    try:
        m = pd.read_sql("SELECT match_id, status, score, best_of FROM matches "
                        "WHERE status IN ('retired', 'default', 'unfinished')", con)
    finally:
        con.close()
    m["sets_done"] = [completed_sets(sc, b) for sc, b in zip(m["score"], m["best_of"])]
    return m[["match_id", "status", "sets_done"]]


def voided(df: pd.DataFrame, book: str | None = None) -> np.ndarray:
    """Rows whose moneyline bet the book would have refunded.

    A retirement is a win on the scoreboard but only a bet for the books that
    count it; settling every one at the closing price handed the model wins it
    could not have been paid on and losses it could not have suffered.
    """
    if "status" not in df.columns:
        return np.zeros(len(df), dtype=bool)
    if book is None:
        book = df["book"].iat[0] if "book" in df.columns and len(df) else None
    status = df["status"].to_numpy()
    done = df["sets_done"].fillna(0).to_numpy() if "sets_done" in df.columns \
        else np.zeros(len(df))
    need = RETIREMENT_MIN_SETS.get(book, np.inf)
    return (status == "default") | (status == "unfinished") | \
           ((status == "retired") & (done < need))


# --------------------------------------------------------------------------
# ROI
# --------------------------------------------------------------------------
def roi_winner(df: pd.DataFrame, edge: float = 0.03, stake: float = 1.0,
               kelly: bool = False, cap: float = 0.05) -> dict:
    """Flat (or fractional-Kelly) staking on whichever side shows an edge.

    Bets the book would void (see `voided`) are not bets: they return the stake
    and count in `voided_matches`. The ROI interval resamples whole tournaments.
    """
    d = df.dropna(subset=["p1_price", "p2_price", "p_win"]).copy()
    if d.empty:
        return {"bets": 0}

    # Edge on each side against that side's actual price.
    d["ev1"] = d["p_win"] * d["p1_price"] - 1
    d["ev2"] = (1 - d["p_win"]) * d["p2_price"] - 1
    d["side"] = np.where(d["ev1"] >= d["ev2"], 1, 2)
    d["ev"] = np.maximum(d["ev1"], d["ev2"])
    d = d[d["ev"] > edge]
    void = voided(d)
    n_void_pool = int(void.sum())
    d = d[~void]
    if d.empty:
        return {"bets": 0, "voided_matches": n_void_pool}

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
    # Without ids there is no way to tell which bets share an event.
    clusters = (tourney_key_of(d["match_id"]).to_numpy() if "match_id" in d
                else np.arange(len(d)))
    lo, hi = cluster_bootstrap_ratio(profit, size, clusters)
    return {
        "bets": int(len(d)),
        "voided_matches": n_void_pool,
        "turnover": round(float(turnover), 1),
        "profit": round(float(profit.sum()), 2),
        "roi_pct": round(float(profit.sum() / turnover * 100), 3) if turnover else 0.0,
        "roi_ci95_tournament_pct": [round(lo * 100, 2), round(hi * 100, 2)],
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
    for season, g in df.groupby(season_of(df["tourney_date"].to_numpy())):
        r = roi_winner(g, edge=edge)
        if r.get("bets"):
            out.append({"season": int(season), **r})
    return out


def _reference_line(df: pd.DataFrame, col: str) -> pd.Series:
    """Naive prior-season reference line by (surface, level).

    Uses only STRICTLY EARLIER seasons. The earlier version shifted one row in a
    date-sorted frame, which -- every match in an event sharing one date -- let
    a row's line include other matches of its own event, in an arbitrary order.
    """
    d = df[["surface", "is_challenger", "tourney_date", col]].copy()
    d["season"] = season_of(d["tourney_date"].to_numpy())
    g = (d.groupby(["surface", "is_challenger", "season"], observed=True)[col]
           .agg(["sum", "count"]).reset_index().sort_values("season"))
    keys = ["surface", "is_challenger"]
    g["cum_sum"] = g.groupby(keys, observed=True)["sum"].cumsum() - g["sum"]
    g["cum_n"] = g.groupby(keys, observed=True)["count"].cumsum() - g["count"]
    g["line"] = (g["cum_sum"] / g["cum_n"]).where(g["cum_n"] >= 50)
    out = d.merge(g[keys + ["season", "line"]], on=keys + ["season"], how="left")
    return pd.Series(out["line"].to_numpy(), index=df.index)


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


def _carry_market_lines(rep: dict, tour: str) -> None:
    """Keep the previous report's games-market block when this run has none.

    The block rests on `odds_quotes`, a research table that exists only on the
    machine that ran the backfill. Where it is absent (a cloud rebuild) the last
    computed block is kept, stamped with how old it is, rather than silently
    dropping the section from the dashboard.
    """
    if "market_lines" in rep:
        return
    prior = ARTIFACTS / f"backtest_report_{tour}.json"
    if not prior.exists():
        return
    try:
        old = json.loads(prior.read_text())
    except ValueError:
        return
    if old.get("market_lines"):
        rep["market_lines"] = {
            **old["market_lines"],
            "carried_forward_from": old["market_lines"].get("carried_forward_from")
            or old.get("generated_at", "an earlier run")}


def _verdict(rep: dict, best_roi: float, beats_close: bool | None) -> str:
    """One-paragraph reading of the numbers in `rep`, written from the numbers.

    This was a fixed string asserting "loses money at every edge" whatever the
    run found, so a profitable or a worse-than-Elo run would still have been
    published with the same sentence.
    """
    ms = rep["market_subset"]
    if not ms.get("n"):
        return "No priced matches joined; ROI not computable this run."
    ll, base = rep["winner"]["log_loss"], rep["winner_elo_baseline"]["log_loss"]
    parts = [("Model is more accurate than an Elo-only baseline"
              if ll < base else "Model is NOT more accurate than an Elo-only baseline")
             + f" (log loss {ll:.4f} vs {base:.4f})"]
    if beats_close is not None:
        m = ms["market_closing"]["log_loss"]
        parts.append(("and beats" if beats_close else "and is less accurate than")
                     + f" the closing line on priced matches ({ms['model']['log_loss']:.4f}"
                       f" vs {m:.4f}).")
    rois = {k: v["roi_pct"] for k, v in rep["roi_winner"].items() if v.get("bets")}
    if rois and np.isfinite(best_roi):
        if best_roi > 0:
            k = max(rois, key=rois.get)
            parts.append(f"Flat-staking against that line returned {best_roi:+.2f}% at "
                         f"{k.replace('_', ' ')} (best of {len(rois)} thresholds); check the "
                         "per-season table before trusting it.")
        else:
            parts.append("Flat-staking against that line loses money at every edge "
                         f"threshold tested (best {best_roi:.2f}%).")
    return " ".join(parts)


def _selection_windows(bt: pd.DataFrame) -> dict:
    """Winner metrics split by how much of the window the model was selected on.

    `dev` is before HOLDOUT_FROM, `holdout` is after it, and `post_freeze` is
    after SELECTION_FROZEN_ON. Only the last was never scored while Elo
    constants or features were being chosen, so only it is a clean out-of-sample
    figure; the others are in-sample to those choices, if mildly.
    """
    season = season_of(bt["tourney_date"].to_numpy())
    windows = {"dev": season < HOLDOUT_FROM, "holdout": season >= HOLDOUT_FROM,
               "post_freeze": bt["tourney_date"].to_numpy() > SELECTION_FROZEN_ON}
    out: dict = {"holdout_from": HOLDOUT_FROM, "frozen_on": SELECTION_FROZEN_ON}
    for k, m in windows.items():
        out[k] = (classification_metrics(bt.loc[m, "y_win"].to_numpy(),
                                         bt.loc[m, "p_win"].to_numpy())
                  if m.sum() >= 100 else {"n": int(m.sum())})
    return out


def full_report(book: str = "PS", tour: str = "atp") -> dict:
    bt = load_backtest(tour)
    bt["tour_level"] = np.where(bt["is_challenger"] == 1, "challenger", "main_tour")
    bt["tour"] = tour
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

    rep["selection_windows"] = _selection_windows(bt)

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
    if not mk.empty:
        # Retirements are real results but noisy ones, and a market that prices
        # them as ordinary matches is scored on them too. Reported apart so the
        # headline cannot be read as either cleaner or worse than it is.
        cm = mk[mk["status"] == "completed"]
        rep["market_subset"]["completed_only"] = {
            "n": int(len(cm)),
            "model": classification_metrics(cm["y_win"].to_numpy(), cm["p_win"].to_numpy()),
            "market_closing": classification_metrics(
                cm["y_win"].to_numpy(), cm["mkt_p1"].to_numpy()),
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

    _carry_market_lines(rep, tour)
    rep["generated_at"] = pd.Timestamp.now("UTC").isoformat(timespec="seconds")

    rep["headline"] = {
        "winner_beats_elo_baseline": rep["winner"]["log_loss"] < rep["winner_elo_baseline"]["log_loss"],
        "winner_beats_closing_line": beats_close,
        "best_winner_roi_pct": best_roi,
        "verdict": _verdict(rep, best_roi, beats_close),
        "totals_spread_roi": (
            "Not computable against a market: the free data has no totals or "
            "handicap lines. Only a synthetic-line signal check is reported."
        ),
    }

    for dim in ("surface", "level", "tour_level", "tour"):
        rep[f"by_{dim}"] = breakout(bt, dim).to_dict("records")
    bt["season"] = season_of(bt["tourney_date"].to_numpy())
    rep["by_season"] = breakout(bt, "season").to_dict("records")

    (ARTIFACTS / f"backtest_report_{tour}.json").write_text(
        json.dumps(rep, indent=2, default=str))
    return rep


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    tours = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--tour=")]
    for t in (tours or ["atp", "wta"]):
        print(f"===== {t} =====")
        r = full_report(tour=t)
        print(json.dumps({k: v for k, v in r.items()
                          if k not in ("calibration", "by_season")},
                         indent=2, default=str))
