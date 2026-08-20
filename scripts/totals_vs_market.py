"""Totals and spread against a real market, for the first time.

Until the tennisexplorer backfill there was no source carrying over/under or
handicap lines on *games*, so two of this project's three models had never been
scored against anything but themselves. This is that comparison.

Turning a point prediction into a price is the whole problem. The model emits
"21.4 games"; a market asks P(total > 21.5). Three things make the obvious
normal approximation wrong, all measured rather than assumed:

* **best-of matters enormously.** Residual sd is 5.90 games in best-of-three
  and 9.12 in best-of-five. Pooling them is the single biggest available error.
* **residuals are heteroscedastic.** Within best-of-three, sd runs 5.57 at the
  low end of predicted totals to 6.78 at the high end.
* **residuals are right-skewed** (skew ~0.5). A normal misprices the tails,
  which is exactly where the outer lines sit -- and outer lines are where a
  totals model would have to find its edge.

So the conversion uses the **empirical** residual distribution, conditioned on
best-of and on predicted total, and fitted only on seasons *before* the ones
scored. The market side is Shin-devigged (see `models/devig.py`).

**Two orientation traps, both resolved.** The handicap market needs the line
and the price attached to the same player, and tennisexplorer makes that awkward
in two independent ways.

*The line is not in k1's frame.* The k1/k2 columns are winner-first, but the
`value` line is quoted against a pre-match ordering, so the two coincide only
about half the time -- measured, 51%. Pooled, that cancels the signal exactly:
within-match Spearman(line, P) came out bimodal, 45.7% of matches near -1 and
48.3% near +1, giving a pooled correlation of +0.09 and a market that looked no
better than a coin flip. The fix needs no re-parse: the ladder's own direction
reveals whose frame the line is in, since P(k1 covers) must fall as the line
rises. That alone lifted the correlation from +0.09 to +0.43.

*k1 is the winner* (99.8% of resolved matches), so evaluating in k1's frame
makes the outcome trivially biased -- actual cover rate 0.82 against a market
mean of 0.50. Moving into the backtest's own p1, which is chosen by hashing the
match id and is therefore independent of the result, removes it: market 0.500,
actual 0.501. Knowing who won is needed only to identify *which column* holds
our p1's price; the price itself was set pre-match, so nothing leaks.

Totals need neither correction -- over/under has no player orientation, which is
why that half worked from the start.

Run:  python scripts/totals_vs_market.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scipy.stats import spearmanr  # noqa: E402

from tennis.db.schema import connect  # noqa: E402
from tennis.models.common import log_loss  # noqa: E402
from tennis.models.devig import shin  # noqa: E402
from tennis.models.evaluate import load_backtest  # noqa: E402

FIT_BEFORE = 2023          # residuals fitted on earlier seasons only
EPS = 1e-6


def load_market(market: str, con) -> pd.DataFrame:
    """One row per (match, line): devigged market probability for the first side.

    Prices are averaged across books at the same line *in probability space
    after devigging*, not by averaging raw odds -- averaging odds and devigging
    once would fold the books' differing margins into the consensus.
    """
    q = pd.read_sql(f"""
        SELECT q.te_id, q.line, q.book,
               MAX(CASE WHEN q.side IN ('over','p1')  THEN q.price_close END) o1,
               MAX(CASE WHEN q.side IN ('under','p2') THEN q.price_close END) o2
        FROM odds_quotes q JOIN te_matches m USING(te_id)
        WHERE q.market = '{market}' AND q.line_unit = 'games'
          AND m.match_id IS NOT NULL AND q.price_close IS NOT NULL
        GROUP BY q.te_id, q.line, q.book""", con)
    q = q.dropna(subset=["o1", "o2"])
    q = q[(q.o1 > 1.001) & (q.o2 > 1.001)]
    if q.empty:
        return q
    q["p_first"] = shin(q[["o1", "o2"]].to_numpy(float))[0][:, 0]
    q["overround"] = 1 / q.o1 + 1 / q.o2
    agg = (q.groupby(["te_id", "line"])
             .agg(p_mkt=("p_first", "mean"), books=("book", "nunique"),
                  overround=("overround", "median"))
             .reset_index())
    if market == "handicap":
        # Put the line in k1's frame. See the module docstring: the stored sign
        # follows a pre-match ordering that agrees with the winner-first k1/k2
        # columns only ~51% of the time, and the ladder's own direction is what
        # tells them apart. Matches with too few rungs to establish a direction
        # are dropped rather than guessed.
        rho = (agg.groupby("te_id").filter(lambda g: len(g) >= 4)
                  .groupby("te_id")
                  .apply(lambda g: spearmanr(g.line, g.p_mkt).statistic,
                         include_groups=False)
                  .rename("rho"))
        agg = agg.merge(rho, on="te_id")
        agg["line"] = np.where(agg.rho < 0, agg.line, -agg.line)
        agg = agg.drop(columns="rho")
    return agg


class EmpiricalResiduals:
    """P(actual > line) from the empirical residual distribution.

    Conditioned on best-of and a predicted-value bucket, because both shift the
    spread of the residuals materially. Falls back to the wider pool whenever a
    cell is too thin to be trusted on its own.
    """

    def __init__(self, df: pd.DataFrame, pred: str, actual: str, nbins: int = 6):
        self.pred, self.nbins = pred, nbins
        d = df.dropna(subset=[pred, actual]).copy()
        d["res"] = d[actual] - d[pred]
        self.edges = {}
        self.cells: dict = {}
        self.pool: dict = {}
        for bo, g in d.groupby("best_of"):
            self.pool[bo] = g.res.to_numpy()
            try:
                _, e = pd.qcut(g[pred], nbins, retbins=True, duplicates="drop")
            except ValueError:
                e = np.array([-np.inf, np.inf])
            e[0], e[-1] = -np.inf, np.inf
            self.edges[bo] = e
            for i in range(len(e) - 1):
                m = (g[pred] > e[i]) & (g[pred] <= e[i + 1])
                if m.sum() >= 500:
                    self.cells[(bo, i)] = np.sort(g.loc[m, "res"].to_numpy())
        self.all = np.sort(d.res.to_numpy())

    def _sample(self, bo, pv):
        e = self.edges.get(bo)
        if e is not None:
            i = int(np.clip(np.searchsorted(e, pv, side="right") - 1, 0, len(e) - 2))
            c = self.cells.get((bo, i))
            if c is not None:
                return c
        p = self.pool.get(bo)
        return np.sort(p) if p is not None and len(p) >= 500 else self.all

    def p_over(self, best_of, pred_val, line) -> np.ndarray:
        """P(actual > line). Vectorised over rows, cell lookup per row."""
        out = np.empty(len(line))
        for k, (bo, pv, ln) in enumerate(zip(best_of, pred_val, line)):
            s = self._sample(bo, pv)
            # P(res > line - pred) = fraction of residuals above that threshold
            out[k] = 1.0 - np.searchsorted(s, ln - pv, side="right") / len(s)
        return np.clip(out, EPS, 1 - EPS)


def score(label, y, p_model, p_mkt, extra=None) -> dict:
    row = {
        "market": label, "n": len(y),
        "model_ll": round(log_loss(y, p_model), 5),
        "market_ll": round(log_loss(y, p_mkt), 5),
        "model_brier": round(float(np.mean((p_model - y) ** 2)), 5),
        "market_brier": round(float(np.mean((p_mkt - y) ** 2)), 5),
    }
    row["gap"] = round(row["model_ll"] - row["market_ll"], 5)
    if extra:
        row.update(extra)
    return row


def main() -> None:
    con = connect()
    bt = load_backtest()[["match_id", "tourney_date", "is_challenger", "y_win",
                          "pred_total", "y_total", "pred_spread", "y_spread"]]
    tm = pd.read_sql("SELECT te_id, match_id, p1_is_winner FROM te_matches "
                     "WHERE match_id IS NOT NULL", con)
    mb = pd.read_sql("SELECT match_id, best_of, totals_usable FROM matches", con)
    base = bt.merge(tm, on="match_id").merge(mb, on="match_id")
    base = base[base.totals_usable == 1]
    base["season"] = base.tourney_date // 10000
    # Whose p1? The backtest picks p1 by hashing match_id; tennisexplorer has
    # its own ordering. A handicap is quoted from *their* p1, while y_spread is
    # margin relative to *ours*, so every handicap row must be re-oriented or
    # the comparison silently prices the wrong player. Both frames know who
    # won, which is enough to align them.
    base["same_side"] = (base.p1_is_winner == 1) == (base.y_win == 1)

    fit = base[base.season < FIT_BEFORE]
    print(f"residual distributions fitted on {len(fit):,} matches before {FIT_BEFORE}")

    results = []
    for label, market, pred, actual, sign in (
            ("totals (games)", "totals", "pred_total", "y_total", +1),
            ("spread (games)", "handicap", "pred_spread", "y_spread", +1)):
        mkt = load_market(market, con)
        if mkt.empty:
            print(f"\n{label}: no market rows")
            continue
        d = base.merge(mkt, on="te_id")
        d = d.dropna(subset=[pred, actual])
        d = d[d.season >= FIT_BEFORE]

        emp = EmpiricalResiduals(fit, pred, actual)
        if market == "totals":
            # Over/under does not depend on player order, so nothing to align.
            thresh = d.line.to_numpy()
            p_mkt = d.p_mkt.to_numpy()
        else:
            # `line` now sits in k1's frame, and k1 is the winner. Move both the
            # line and the price into our p1's frame; our p1 is hash-chosen, so
            # the resulting sample is independent of the result.
            ours = d.y_win.to_numpy() == 1
            line_ours = np.where(ours, d.line.to_numpy(), -d.line.to_numpy())
            p_mkt = np.where(ours, d.p_mkt.to_numpy(), 1.0 - d.p_mkt.to_numpy())
            # p1 covers by exceeding the line. Confirmed against the corrected
            # orientation: this convention gives market 0.500 / actual 0.501,
            # the other gives 0.83 log loss.
            thresh = line_ours
        p_model = emp.p_over(d.best_of.to_numpy(), d[pred].to_numpy(), thresh)
        y = (d[actual].to_numpy() > thresh).astype(float)
        push = d[actual].to_numpy() == thresh
        keep = ~push
        print(f"\n{label}: {len(d):,} match-line rows, {push.sum():,} pushes dropped")

        results.append(score(label, y[keep], p_model[keep],
                             np.clip(p_mkt[keep], EPS, 1 - EPS),
                             {"matches": int(d.te_id.nunique()),
                              "overround": round(float(d.overround.median()), 4)}))
        d = d[keep].assign(p_model=p_model[keep], y=y[keep], p_mkt=p_mkt[keep])
        for tier, g in d.groupby(np.where(d.is_challenger == 1, "challenger", "main tour")):
            results.append(score(f"  {label} / {tier}", g.y.to_numpy(),
                                 g.p_model.to_numpy(),
                                 np.clip(g.p_mkt.to_numpy(), EPS, 1 - EPS)))
    con.close()

    print("\n=== model vs market (log loss; gap > 0 means the market is better) ===")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
