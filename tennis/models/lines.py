"""Totals and handicap lines: market probabilities, and the model against them.

Extracted from `scripts/totals_vs_market.py` so the nightly evaluation can
write these figures into the report and the dashboard can render them without
touching `odds_quotes`, which is 15M rows heading for 99M. The dashboard must
never run this; it reads the precomputed block.

Two orientation traps live here, both peculiar to the handicap ladder and both
invisible until pooled.

**The line is not in k1's frame.** The k1/k2 columns are winner-first, while the
`value` line is quoted against a pre-match ordering, so the two agree only ~51%
of the time. Pooled that cancels the signal exactly -- within-match
Spearman(line, P) is bimodal, 45.7% near -1 and 48.3% near +1, giving a market
that scored worse than a coin flip. The ladder's own direction identifies whose
frame the line is in, since P(k1 covers) must fall as the line rises.

**k1 is the winner** (99.8% of resolved matches), so scoring in k1's frame makes
the outcome trivially biased. Everything is moved into the backtest's own p1,
which is chosen by hashing the match id and so is independent of the result.
Knowing who won only identifies *which column* holds our p1's price; the price
itself was set pre-match, so nothing leaks.

Totals need neither correction: over/under has no player orientation.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from tennis.models.common import log_loss
from tennis.models.devig import shin

log = logging.getLogger(__name__)

FIT_BEFORE = 2023          # residual distributions fitted on earlier seasons only
EPS = 1e-6
MIN_LADDER = 4             # rungs needed before a ladder's direction is trusted


def load_market(market: str, con, book: str | None = None) -> pd.DataFrame:
    """One row per (match, line): devigged market probability for the first side.

    Books are averaged in probability space *after* devigging; averaging raw
    odds first would fold their differing margins into the consensus.
    """
    where = "" if book is None else f" AND q.book = '{book}'"
    q = pd.read_sql(f"""
        SELECT q.te_id, q.line, q.book,
               MAX(CASE WHEN q.side IN ('over','p1')  THEN q.price_close END) o1,
               MAX(CASE WHEN q.side IN ('under','p2') THEN q.price_close END) o2
        FROM odds_quotes q JOIN te_matches m USING(te_id)
        WHERE q.market = '{market}' AND q.line_unit = 'games'
          AND m.match_id IS NOT NULL AND q.price_close IS NOT NULL{where}
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
        agg = _orient_ladder(agg)
    return agg


def _orient_ladder(agg: pd.DataFrame) -> pd.DataFrame:
    """Put each handicap line into k1's frame using the ladder's direction."""
    from scipy.stats import spearmanr

    rho = (agg.groupby("te_id").filter(lambda g: len(g) >= MIN_LADDER)
              .groupby("te_id")
              .apply(lambda g: spearmanr(g.line, g.p_mkt).statistic,
                     include_groups=False)
              .rename("rho"))
    agg = agg.merge(rho, on="te_id")       # matches with a short ladder drop out
    agg["line"] = np.where(agg.rho < 0, agg.line, -agg.line)
    return agg.drop(columns="rho")


class EmpiricalResiduals:
    """P(actual > line) from the empirical residual distribution.

    Conditioned on best-of and a predicted-value bucket. Both matter and were
    measured, not assumed: residual sd is 5.90 games in best-of-three against
    9.12 in best-of-five, and within best-of-three it runs 5.57 to 6.78 across
    the range of predicted totals. Residuals are also right-skewed (~0.5), so a
    normal would misprice the tails -- which is exactly where the outer rungs of
    a ladder sit.
    """

    def __init__(self, df: pd.DataFrame, pred: str, actual: str, nbins: int = 6):
        d = df.dropna(subset=[pred, actual]).copy()
        d["res"] = d[actual] - d[pred]
        self.edges, self.cells, self.pool = {}, {}, {}
        for bo, g in d.groupby("best_of"):
            self.pool[bo] = np.sort(g.res.to_numpy())
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
        return p if p is not None and len(p) >= 500 else self.all

    def p_over(self, best_of, pred_val, line) -> np.ndarray:
        out = np.empty(len(line))
        for k, (bo, pv, ln) in enumerate(zip(best_of, pred_val, line)):
            s = self._sample(bo, pv)
            out[k] = 1.0 - np.searchsorted(s, ln - pv, side="right") / len(s)
        return np.clip(out, EPS, 1 - EPS)


def _frame(market, pred, actual, con, bt, book=None) -> pd.DataFrame:
    tm = pd.read_sql("SELECT te_id, match_id FROM te_matches "
                     "WHERE match_id IS NOT NULL", con)
    mb = pd.read_sql("SELECT match_id, best_of, totals_usable FROM matches", con)
    base = bt.merge(tm, on="match_id").merge(mb, on="match_id")
    base = base[base.totals_usable == 1]
    base["season"] = base.tourney_date // 10000

    mkt = load_market(market, con, book=book)
    if mkt.empty:
        return mkt
    emp = EmpiricalResiduals(base[base.season < FIT_BEFORE], pred, actual)
    d = base.merge(mkt, on="te_id").dropna(subset=[pred, actual])
    d = d[d.season >= FIT_BEFORE].copy()
    if d.empty:
        return d

    if market == "totals":
        d["line_ours"], d["p_mkt_ours"] = d.line, d.p_mkt
    else:
        ours = d.y_win.to_numpy() == 1
        d["line_ours"] = np.where(ours, d.line, -d.line)
        d["p_mkt_ours"] = np.where(ours, d.p_mkt, 1 - d.p_mkt)
    d["p_model"] = emp.p_over(d.best_of.to_numpy(), d[pred].to_numpy(),
                              d.line_ours.to_numpy())
    d = d[d[actual] != d.line_ours]
    d["y"] = (d[actual] > d.line_ours).astype(float)
    return d


def market_comparison(con, bt: pd.DataFrame) -> dict:
    """Model against the real games market, for the report and the dashboard.

    Returns a dict carrying its own coverage metadata. These figures rest on
    whatever the tennisexplorer backfill has reached, which is not the full
    backtest period, so the numbers must never be read as covering it.
    """
    out: dict = {"fitted_before": FIT_BEFORE, "markets": {}}
    for label, market, pred, actual in (
            ("totals", "totals", "pred_total", "y_total"),
            ("spread", "handicap", "pred_spread", "y_spread")):
        d = _frame(market, pred, actual, con, bt)
        if d.empty:
            continue
        y = d.y.to_numpy()
        entry = {
            "rows": int(len(d)), "matches": int(d.match_id.nunique()),
            "seasons": [int(d.season.min()), int(d.season.max())],
            "median_vig_pct": round(float((1 - 1 / d.overround.median()) * 100), 2),
            "model_ll": round(log_loss(y, d.p_model.to_numpy()), 5),
            "market_ll": round(log_loss(y, d.p_mkt_ours.to_numpy()), 5),
            "by_tier": {},
        }
        entry["gap"] = round(entry["model_ll"] - entry["market_ll"], 5)
        for tier, g in d.groupby(np.where(d.is_challenger == 1,
                                          "challenger", "main tour")):
            gy = g.y.to_numpy()
            entry["by_tier"][tier] = {
                "rows": int(len(g)),
                "model_ll": round(log_loss(gy, g.p_model.to_numpy()), 5),
                "market_ll": round(log_loss(gy, g.p_mkt_ours.to_numpy()), 5),
            }
            entry["by_tier"][tier]["gap"] = round(
                entry["by_tier"][tier]["model_ll"]
                - entry["by_tier"][tier]["market_ll"], 5)
        out["markets"][label] = entry
    return out


def clv(con, bt: pd.DataFrame, book: str = "Pinnacle", edge: float = 0.02) -> dict:
    """Closing-line value on bets placed at the opening price.

    The success metric that matters, because it scores every bet rather than
    only the ones that won. Reported beside the vig deliberately: a positive CLV
    of a third of a point against a 3.5 point margin is signal, not profit, and
    the two numbers are meaningless apart.
    """
    out: dict = {"book": book, "edge": edge, "markets": {}}
    for label, market, pred, actual in (
            ("totals", "totals", "pred_total", "y_total"),
            ("spread", "handicap", "pred_spread", "y_spread")):
        s1, s2 = ("over", "under") if market == "totals" else ("p1", "p2")
        q = pd.read_sql(f"""
            SELECT te_id, line,
                   MAX(CASE WHEN side='{s1}' THEN price_open  END) op1,
                   MAX(CASE WHEN side='{s2}' THEN price_open  END) op2,
                   MAX(CASE WHEN side='{s1}' THEN price_close END) cl1,
                   MAX(CASE WHEN side='{s2}' THEN price_close END) cl2
            FROM odds_quotes
            WHERE market='{market}' AND line_unit='games' AND book='{book}'
            GROUP BY te_id, line""", con).dropna()
        q = q[(q[["op1", "op2", "cl1", "cl2"]] > 1.001).all(axis=1)]
        if len(q) < 500:
            continue
        q["p_open"] = shin(q[["op1", "op2"]].to_numpy(float))[0][:, 0]
        q["p_close"] = shin(q[["cl1", "cl2"]].to_numpy(float))[0][:, 0]
        q["p_mkt"] = q.p_close
        q["overround"] = 1 / q.cl1 + 1 / q.cl2
        if market == "handicap":
            q = _orient_ladder(q)

        tm = pd.read_sql("SELECT te_id, match_id FROM te_matches "
                         "WHERE match_id IS NOT NULL", con)
        mb = pd.read_sql("SELECT match_id, best_of, totals_usable FROM matches", con)
        base = bt.merge(tm, on="match_id").merge(mb, on="match_id")
        base = base[base.totals_usable == 1]
        base["season"] = base.tourney_date // 10000
        emp = EmpiricalResiduals(base[base.season < FIT_BEFORE], pred, actual)
        d = base.merge(q, on="te_id").dropna(subset=[pred, actual])
        d = d[d.season >= FIT_BEFORE].copy()
        if d.empty:
            continue
        if market == "handicap":
            ours = d.y_win.to_numpy() == 1
            d["line"] = np.where(ours, d.line, -d.line)
            o1, o2 = d.op1.to_numpy(), d.op2.to_numpy()
            c1, c2 = d.cl1.to_numpy(), d.cl2.to_numpy()
            d["op1"], d["op2"] = np.where(ours, o1, o2), np.where(ours, o2, o1)
            d["cl1"], d["cl2"] = np.where(ours, c1, c2), np.where(ours, c2, c1)
            d["p_open"] = np.where(ours, d.p_open, 1 - d.p_open)
            d["p_close"] = np.where(ours, d.p_close, 1 - d.p_close)
        d["p_model"] = emp.p_over(d.best_of.to_numpy(), d[pred].to_numpy(),
                                  d.line.to_numpy())
        ev1 = d.p_model * d.op1 - 1
        ev2 = (1 - d.p_model) * d.op2 - 1
        g = d.assign(side1=ev1 >= ev2, ev=np.maximum(ev1, ev2))
        # One bet per match: a ladder is a menu, not a dozen independent bets.
        g = g[g.ev > edge].sort_values("ev", ascending=False)
        g = g.drop_duplicates("match_id")
        if len(g) < 200:
            continue
        po = np.where(g.side1, g.p_open, 1 - g.p_open)
        pc = np.where(g.side1, g.p_close, 1 - g.p_close)
        v = (pc - po) * 100
        beat = float((v > 0).mean())
        out["markets"][label] = {
            "bets": int(len(g)),
            "mean_clv_pts": round(float(v.mean()), 3),
            "se_pts": round(float(v.std(ddof=1) / np.sqrt(len(v))), 3),
            "beat_close_pct": round(beat * 100, 1),
            "z": round(float((beat - 0.5) / np.sqrt(0.25 / len(v))), 1),
            "vig_to_overcome_pts": round(
                float((1 - 1 / (1 / g.cl1 + 1 / g.cl2).median()) * 100), 2),
        }
    return out
