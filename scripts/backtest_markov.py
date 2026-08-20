"""Hierarchical Markov chain against the gradient booster, on identical matches.

The Markov model is nearly parameter-free: it takes each player's rolling
serve and return rates, combines them into one point-win probability per
server, and propagates that through the scoring structure exactly. Nothing is
fitted to outcomes, so there is no training set here -- but the serve rates
come from the same leak-safe rolling state the booster uses, so the two see the
same information and the comparison is fair.

Three things this is really testing:

* **Does the scoring structure beat learning the curve?** The booster has to
  infer the point-to-match mapping from outcome labels; the chain gets it
  exactly. If the chain wins, structure is worth more than flexibility here.
* **Is it better where it should be?** The chain's advantage should be largest
  in best-of-five, where the amplification is strongest and the booster has the
  least data.
* **Do they disagree usefully?** Two models with genuinely different failure
  modes can be worth more combined than either alone, which is the case for an
  ensemble rather than a replacement.

Everything is scored on the subset where both models produce a number, so no
comparison is flattered by a different match set.

Run:  python scripts/backtest_markov.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tennis.config import ARTIFACTS  # noqa: E402
from tennis.db.schema import connect  # noqa: E402
from tennis.models.common import log_loss  # noqa: E402
from tennis.models.evaluate import attach_odds, load_backtest  # noqa: E402
from tennis.models import serve_ratings as SR  # noqa: E402
from tennis.models.markov import p_match, tour_average_spw  # noqa: E402

EPS = 1e-6


def score(label: str, y: np.ndarray, p: np.ndarray, extra: dict | None = None):
    p = np.clip(p, EPS, 1 - EPS)
    row = {"model": label, "n": len(y),
           "log_loss": round(log_loss(y, p), 5),
           "brier": round(float(np.mean((p - y) ** 2)), 5),
           "acc": round(float(np.mean((p > 0.5) == (y == 1))), 4)}
    if extra:
        row.update(extra)
    return row


def build() -> pd.DataFrame:
    """Backtest rows with a Markov probability from adjusted serve ratings.

    The raw rolling serve rate in `features.parquet` is deliberately *not*
    used: fed to the chain it calibrates at slope 0.34, because it is neither
    opponent-adjusted nor shrunk and the chain is steeply convex. See
    `tennis.models.serve_ratings`.
    """
    con = connect()
    con.execute("PRAGMA busy_timeout=60000")
    avg = tour_average_spw(con)
    m = pd.read_sql("""
        SELECT mm.match_id, mm.seq, mm.winner_id, mm.loser_id, mm.best_of,
               mm.w_svpt, mm.w_1stWon, mm.w_2ndWon,
               mm.l_svpt, mm.l_1stWon, mm.l_2ndWon, t.surface
        FROM matches mm JOIN tournaments t USING(tourney_key)
        ORDER BY mm.seq""", con)
    con.close()
    print(f"tour average serve points won: {avg:.4f}")

    rat = SR.fit(m, avg)
    bt = load_backtest().merge(m[["match_id", "best_of"]], on="match_id",
                               suffixes=("", "_m"))
    d = bt.merge(rat, on="match_id")
    o = SR.to_p1p2(d[["match_id", "w_serve", "w_ret", "l_serve", "l_ret"]],
                   d.y_win.to_numpy())
    d = d.drop(columns=["w_serve", "w_ret", "l_serve", "l_ret"]).merge(
        o, on="match_id")

    d["best_of"] = d["best_of"].fillna(3)
    # Each player's point-win probability serving to this specific opponent.
    d["p1_pt"] = np.clip(avg + d.p1_serve - d.p2_ret, 0.30, 0.90)
    d["p2_pt"] = np.clip(avg + d.p2_serve - d.p1_ret, 0.30, 0.90)
    d["p_markov"] = p_match(d.p1_pt.to_numpy(), d.p2_pt.to_numpy(),
                            d.best_of.to_numpy())
    return d.dropna(subset=["p_markov", "p_win", "y_win"])


def main() -> None:
    d = build()
    y = d.y_win.to_numpy()
    print(f"{len(d):,} matches carry both a Markov and a booster probability "
          f"({d.tourney_date.min() // 10000}-{d.tourney_date.max() // 10000})\n")

    rows = [score("Markov (chain)", y, d.p_markov.to_numpy()),
            score("LightGBM (current)", y, d.p_win.to_numpy())]

    # Ensembles. Averaging in log-odds space rather than probability space
    # because both models are probabilistic and the logit is where their
    # errors are closer to additive; the plain mean is reported alongside so
    # the choice is visible rather than assumed.
    def logit(p):
        p = np.clip(p, EPS, 1 - EPS)
        return np.log(p / (1 - p))

    for w in (0.25, 0.5, 0.75):
        z = w * logit(d.p_markov.to_numpy()) + (1 - w) * logit(d.p_win.to_numpy())
        rows.append(score(f"ensemble logit {w:.2f} markov", y, 1 / (1 + np.exp(-z))))
    rows.append(score("ensemble mean 50/50", y,
                      0.5 * d.p_markov.to_numpy() + 0.5 * d.p_win.to_numpy()))

    print("=== overall (lower log loss is better) ===")
    print(pd.DataFrame(rows).to_string(index=False))

    # Where does the chain earn or lose its keep?
    print("\n=== by format and tier ===")
    cuts = []
    for name, mask in (
            ("best-of-3", d.best_of == 3), ("best-of-5", d.best_of == 5),
            ("main tour", d.is_challenger == 0),
            ("challenger", d.is_challenger == 1)):
        g = d[mask]
        if len(g) < 500:
            continue
        yy = g.y_win.to_numpy()
        cuts.append({"split": name, "n": len(g),
                     "markov_ll": round(log_loss(yy, np.clip(g.p_markov, EPS, 1 - EPS)), 5),
                     "lgbm_ll": round(log_loss(yy, np.clip(g.p_win, EPS, 1 - EPS)), 5)})
    cut = pd.DataFrame(cuts)
    cut["gap"] = (cut.markov_ll - cut.lgbm_ll).round(5)
    cut["better"] = np.where(cut.gap < 0, "Markov", "LightGBM")
    print(cut.to_string(index=False))

    # Do they disagree in a way worth combining?
    corr = np.corrcoef(d.p_markov, d.p_win)[0, 1]
    print(f"\ncorrelation between the two probabilities: {corr:.4f}")
    print("A high correlation with a real ensemble gain means they agree on "
          "direction and differ on confidence, which is where the gain is.")

    # Against the market, on matches the market priced.
    mk = attach_odds(d, book="PS").dropna(subset=["mkt_p1"])
    if len(mk) > 500:
        ym = mk.y_win.to_numpy()
        z = 0.5 * logit(mk.p_markov.to_numpy()) + 0.5 * logit(mk.p_win.to_numpy())
        best = 1 / (1 + np.exp(-z))
        print(f"\n=== against Pinnacle closing, {len(mk):,} priced matches ===")
        print(pd.DataFrame([
            score("Markov", ym, mk.p_markov.to_numpy()),
            score("LightGBM", ym, mk.p_win.to_numpy()),
            score("ensemble 50/50", ym, best),
            score("market (Shin)", ym, mk.mkt_p1.to_numpy()),
        ]).to_string(index=False))


if __name__ == "__main__":
    main()
