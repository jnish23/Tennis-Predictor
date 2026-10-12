"""Stage 1: are OddsPortal set-winner prices wrong, after the vig?

Runs on whatever `op_matches` / `set_quotes` hold (see
tennis/ingest/odds_oddsportal.py); prints a coverage table first because with a
partial scrape the first question is how far the numbers can be trusted.

Everything is against the *de-vigged* set price (Shin, per book, averaged in
probability space), oriented on the match favourite from the closing moneyline
so the orientation is fixed before the set is played.

  a. calibration  -- does the set-1 price match set-1 results, by disjoint
                     price bucket (favourite-longshot bias lives here)?
  b. coherence    -- the same book's moneyline implies a set-1 probability
                     (Beta kappa fitted in the prescreen). Where the set price
                     and that disagree, who is right? The moneyline is the
                     sharper, thicker market, so a gap that predicts results
                     in the moneyline's direction is the cleanest edge there is.
  c. ROI          -- flat stake at the BEST available price, only on bets the
                     moneyline-implied probability flags by a pre-set margin,
                     against Pinnacle and consensus. Margins are fixed in
                     advance (no tuning) and reported on both halves of a
                     time split. No opening prices are scraped, so there is no
                     closing-line value here.

An edge claim needs ROI above zero with a bootstrap interval that excludes it,
in both halves, after the set market's own margin.

Run:  python scripts/backtest_set_markets.py [--period 1] [--split-year 2024]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import set_markets_prescreen as pre  # noqa: E402
from tennis.db.schema import connect  # noqa: E402
from tennis.models.devig import shin  # noqa: E402
from tennis.models.markov import set_from_match_beta  # noqa: E402

RNG = np.random.default_rng(11)
MARGINS = (0.0, 0.02, 0.04, 0.06)       # fixed in advance; do not tune
N_BOOT = 500


def roi(price: np.ndarray, won: np.ndarray) -> float:
    """Flat 1-unit stake return per unit staked."""
    return float(np.mean(np.where(won == 1, price - 1.0, -1.0))) if len(price) else np.nan


def roi_ci(price, won, n=N_BOOT):
    if len(price) < 30:
        return np.nan, np.nan, np.nan
    r = np.where(won == 1, price - 1.0, -1.0)
    bs = [r[RNG.integers(0, len(r), len(r))].mean() for _ in range(n)]
    return r.mean(), np.percentile(bs, 5), np.percentile(bs, 95)


def build(con, period: int) -> pd.DataFrame:
    base = pre.load(con, None)                       # moneyline, y1/y2, favourite
    q = pd.read_sql("""
        SELECT q.op_id, q.book, q.home_price, q.away_price, o.match_id,
               o.home_is_winner, o.season, o.league
        FROM set_quotes q JOIN op_matches o USING(op_id)
        WHERE q.period = ? AND o.match_id IS NOT NULL
          AND COALESCE(o.set1_agree, 1) = 1""", con, params=(period,))
    if q.empty:
        return q
    q["p_home"] = shin(q[["home_price", "away_price"]].to_numpy(float))[0][:, 0]
    q["overround"] = 1 / q.home_price + 1 / q.away_price
    g = base.set_index("match_id")
    q = q[q.match_id.isin(g.index)].copy()
    q["fav_is_winner"] = q.match_id.map(g.fav_is_winner).astype(bool)
    q["home_is_fav"] = q.home_is_winner.astype(bool) == q.fav_is_winner
    # price of the favourite side, per book
    q["p_fav"] = np.where(q.home_is_fav, q.p_home, 1 - q.p_home)
    q["o_fav"] = np.where(q.home_is_fav, q.home_price, q.away_price)
    q["o_dog"] = np.where(q.home_is_fav, q.away_price, q.home_price)
    col = f"y{period}"
    agg = (q.groupby("match_id")
             .agg(p_book=("p_fav", "mean"), books=("book", "nunique"),
                  overround=("overround", "median"),
                  best_fav=("o_fav", "max"), best_dog=("o_dog", "max"),
                  season_m=("season", "first"), league=("league", "first"))
             .reset_index())
    pin = q[q.book.str.lower().str.contains("pinnacle")].groupby("match_id")[
        ["o_fav", "o_dog"]].first().rename(columns={"o_fav": "pin_fav", "o_dog": "pin_dog"})
    agg = agg.merge(pin, on="match_id", how="left")
    agg = agg.merge(base[["match_id", "m_fav", col, "best_of", "seg"]]
                    .rename(columns={col: "y"}), on="match_id")
    return agg.dropna(subset=["y"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", type=int, default=1, choices=(1, 2))
    ap.add_argument("--split-year", type=int, default=None)
    a = ap.parse_args()
    con = connect()
    d = build(con, a.period)
    if d.empty:
        print("no resolved set prices yet -- run odds_oddsportal --event ... then --resolve")
        return
    n_op = con.execute("SELECT COUNT(*) FROM op_matches").fetchone()[0]
    n_res = con.execute("SELECT COUNT(*) FROM op_matches WHERE match_id IS NOT NULL").fetchone()[0]
    print(f"# Set-{a.period} price check: {len(d):,} matches priced and joined "
          f"(of {n_op:,} scraped, {n_res:,} resolved)")
    print(f"books per match: median {d.books.median():.0f}   "
          f"set-market overround: median {d.overround.median():.3f}")
    print(d.groupby("league").size().rename("n").to_string())

    # kappa from the full moneyline dataset (Stage 0), not from these matches.
    full = pre.load(con, None)
    full = full[(full.best_of == 3) & full.y1.notna() & full.y2.notna()]
    _, ll, _ = pre.kappa_curves(full, 3)
    k = pre.argmax_kappa(ll)
    d["s_ml"] = set_from_match_beta(d.m_fav.to_numpy(), k, 3)
    d = d[d.best_of == 3].copy()          # kappa above is a best-of-3 fit
    print(f"\nmoneyline-implied set probability uses kappa={pre.fmt_k(k)} (best of 3 only)")

    # a. calibration
    print("\n## a. Calibration of the de-vigged set price")
    d["bk"] = pd.cut(d.p_book, pre.BUCKETS, right=False)
    print(f"{'price bucket':>13} {'n':>6} {'price':>7} {'actual':>7} {'95% CI':>15} {'gap':>7}")
    for kk, g in d.groupby("bk", observed=True):
        lo, hi = pre.wilson(g.y.sum(), len(g))
        print(f"{str(kk):>13} {len(g):>6,} {g.p_book.mean():>7.3f} {g.y.mean():>7.3f} "
              f"{lo:>7.3f}-{hi:.3f} {g.y.mean()-g.p_book.mean():>+7.3f}")
    ll_b = pre.set1_loglik(d.y.to_numpy(), d.p_book.to_numpy()).mean()
    ll_m = pre.set1_loglik(d.y.to_numpy(), d.s_ml.to_numpy()).mean()
    ll_x = pre.set1_loglik(d.y.to_numpy(), (0.5 * d.p_book + 0.5 * d.s_ml).to_numpy()).mean()
    print(f"log-lik: set price {ll_b:.5f}   moneyline-implied {ll_m:.5f}   50/50 blend {ll_x:.5f}")

    # b. coherence
    print("\n## b. Set price minus moneyline-implied set probability")
    d["gap"] = d.p_book - d.s_ml
    d["gb"] = pd.qcut(d.gap, 5, duplicates="drop")
    print(f"{'gap quintile':>20} {'n':>6} {'mean gap':>9} {'y - price':>10} {'90% CI':>17}")
    for kk, g in d.groupby("gb", observed=True):
        r = (g.y - g.p_book).to_numpy()
        bs = [r[RNG.integers(0, len(r), len(r))].mean() for _ in range(N_BOOT)]
        print(f"{str(kk):>20} {len(g):>6,} {g.gap.mean():>+9.4f} {r.mean():>+10.4f} "
              f"[{np.percentile(bs,5):+.4f}, {np.percentile(bs,95):+.4f}]")
    print("  if the moneyline is right, 'y - price' should be positive when the gap is "
          "negative (price too short) and negative when it is positive.")

    # c. ROI at the best available price on moneyline-flagged bets
    print("\n## c. Flat-stake ROI on moneyline-flagged bets (margins fixed in advance)")
    d["edge_fav"] = d.s_ml - 1 / d.best_fav          # model prob - implied at best price
    d["edge_dog"] = (1 - d.s_ml) - 1 / d.best_dog
    split = a.split_year or int(d.season_m.median())
    parts = {"all": d, f"<{split}": d[d.season_m < split], f">={split}": d[d.season_m >= split]}
    print(f"{'margin':>7} {'sample':>8} {'n bets':>7} {'ROI best':>9} {'90% CI':>17} "
          f"{'ROI pinn':>9}")
    for mg in MARGINS:
        for name, part in parts.items():
            rows = []
            for side in ("fav", "dog"):
                sel = part[part[f"edge_{side}"] > mg]
                won = sel.y if side == "fav" else 1 - sel.y
                rows.append(pd.DataFrame({"best": sel[f"best_{side}"], "pin": sel[f"pin_{side}"],
                                          "won": won}))
            b = pd.concat(rows)
            if len(b) < 30:
                print(f"{mg:>7.2f} {name:>8} {len(b):>7} {'(too few)':>9}")
                continue
            m_, lo, hi = roi_ci(b.best.to_numpy(), b.won.to_numpy())
            bp = b.dropna(subset=["pin"])
            print(f"{mg:>7.2f} {name:>8} {len(b):>7,} {m_:>+9.4f} [{lo:+.3f}, {hi:+.3f}] "
                  f"{roi(bp.pin.to_numpy(), bp.won.to_numpy()):>+9.4f}")
    allfav = roi(d.best_fav.to_numpy(), d.y.to_numpy())
    print(f"\nbaseline: back every favourite at best price: {allfav:+.4f}  "
          f"(this is the vig bar -- a flagged-bet ROI must clear it, not zero)")


if __name__ == "__main__":
    main()
