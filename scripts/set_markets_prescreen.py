"""Stage 0 prescreen for set-winner markets: is there anything to find?

Uses only data already in the database: set scores from TennisMyLife and
closing match / over-under-sets prices from the tennisexplorer backfill. No
set-level odds yet -- this asks the cheaper question first: **does a set behave
like an independent draw at the probability the match price implies?**

The reasoning, in order:

1. A best-of-3 match price m maps to a per-set probability s through
   m = s^2 (3 - 2s) *if sets are independent*. A -250 match favourite is a
   -183 set favourite on that arithmetic alone, so "-180 set against -250
   match" is fair, not mispriced.
2. Sets are not independent: the per-set probability is itself uncertain
   (surface, form, matchup). Model that as s ~ Beta(s0*kappa, (1-s0)*kappa);
   kappa -> inf is independence, small kappa is strong correlation. Given a
   match price and kappa there is exactly one consistent marginal set-1
   probability s0, and then every set quantity follows:

       P(w1)        = s0
       P(w2 | w1)   = (kappa*s0 + 1) / (kappa + 1)
       P(w2 | l1)   = kappa*s0 / (kappa + 1)

   Losing set 1 lowers P(set 2) with *no* momentum at all, purely because it is
   evidence the favourite is weaker than priced. Only a residual beyond this
   counts as psychology -- and with only (set 1, set 2) pairs the two cannot be
   separated, which is stated rather than hidden.
3. Three independent estimates of kappa should agree if the model is coherent:
   from set-1 results alone, from the set-1/set-2 pairing, and from the
   over/under-2.5-sets market. Where the *market's* kappa differs from the
   *actual* one, the O/U sets and correct-score lines are mispriced against the
   moneyline -- testable on ~130k matches with no new scraping.

Everything is evaluated on disjoint probability buckets with bootstrap CIs
(cumulative ">60%" buckets let heavy favourites dominate), and kappa is also
fitted walk-forward (before FIT_BEFORE) and scored on later seasons.

Orientation is fixed by the *price* (the match favourite), never the result, so
there is no winner-orientation leak: the favourite is a pre-match quantity.

Run:  python scripts/set_markets_prescreen.py [--book Pinnacle] [--out PATH]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tennis.db.schema import connect  # noqa: E402
from tennis.ingest.parse import set_winners  # noqa: E402
from tennis.models.devig import shin  # noqa: E402
from tennis.models.markov import (  # noqa: E402
    match_from_set_beta, set_from_match, set_from_match_beta,
)

FIT_BEFORE = 2023
EPS = 1e-9
KAPPAS = np.array([3, 4, 5, 6, 8, 10, 13, 17, 22, 30, 40, 60, 90, 150, 300, 1e6])
BUCKETS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.0001]
N_BOOT = 300
RNG = np.random.default_rng(7)


# --------------------------------------------------------------------------- data
def _consensus(con, sql: str, where_extra: str = "") -> pd.DataFrame:
    """Shin-devig each book's (side1, side2) pair, then average in prob space."""
    q = pd.read_sql(sql, con)
    q = q.dropna(subset=["o1", "o2"])
    q = q[(q.o1 > 1.001) & (q.o2 > 1.001)]
    q["p1"] = shin(q[["o1", "o2"]].to_numpy(float))[0][:, 0]
    return (q.groupby("te_id").agg(p=("p1", "mean"), books=("book", "nunique"))
             .reset_index())


def load(con, book: str | None) -> pd.DataFrame:
    bk = f" AND q.book = '{book}'" if book else ""
    m = pd.read_sql("""
        SELECT t.te_id, m.match_id, m.score, m.best_of, m.status,
               m.tourney_date / 10000 AS season, tn.level, tn.surface,
               tn.is_challenger, t.p1_is_winner
        FROM te_matches t JOIN matches m USING(match_id)
        JOIN tournaments tn ON tn.tourney_key = m.tourney_key
        WHERE m.tour = 'atp' AND t.match_id IS NOT NULL""", con)
    h2h = _consensus(con, f"""
        SELECT q.te_id, q.book,
               MAX(CASE WHEN q.side='p1' THEN q.price_close END) o1,
               MAX(CASE WHEN q.side='p2' THEN q.price_close END) o2
        FROM odds_quotes q JOIN te_matches t USING(te_id)
        WHERE q.market='h2h' AND t.match_id IS NOT NULL
          AND q.price_close IS NOT NULL{bk}
        GROUP BY q.te_id, q.book""")
    m = m.merge(h2h.rename(columns={"p": "p_p1", "books": "n_books"}), on="te_id")
    # TE's p1 is the match winner in 99.6% of rows; orient on the recorded flag
    # (not the assumption) to get the winner's price, then flip to the favourite.
    p_win = np.where(m.p1_is_winner == 1, m.p_p1, 1 - m.p_p1)
    m["fav_is_winner"] = p_win >= 0.5
    m["m_fav"] = np.where(m.fav_is_winner, p_win, 1 - p_win)

    sw = m.score.map(set_winners)
    for k in (0, 1):
        y = sw.map(lambda s, k=k: s[k] if len(s) > k else np.nan)
        m[f"y{k+1}"] = np.where(m.fav_is_winner, y, 1 - y)   # NaN stays NaN
    m["three_sets"] = np.where(
        (m.best_of == 3) & (m.status == "completed"),
        sw.map(lambda s: float(len(s) == 3)), np.nan)

    # P(over 2.5 sets) = P(a third set), orientation-free (no player frame).
    tot = _consensus(con, f"""
        SELECT q.te_id, q.book,
               MAX(CASE WHEN q.side='over'  THEN q.price_close END) o1,
               MAX(CASE WHEN q.side='under' THEN q.price_close END) o2
        FROM odds_quotes q JOIN te_matches t USING(te_id)
        WHERE q.market='totals' AND q.line_unit='sets' AND q.line=2.5
          AND t.match_id IS NOT NULL AND q.price_close IS NOT NULL{bk}
        GROUP BY q.te_id, q.book""")
    m = m.merge(tot.rename(columns={"p": "p_three"})[["te_id", "p_three"]],
                on="te_id", how="left")
    m["seg"] = np.where(m.is_challenger == 1, "challenger",
                        np.where(m.level == "grand_slam", "slam", "tour"))
    print(f"matches with a closing moneyline: {len(m):,}  "
          f"(winner-oriented p1 flag mean {m.p1_is_winner.mean():.4f})")
    return m


# --------------------------------------------------------------------------- maths
def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def pair_loglik(y1, y2, s0, kappa):
    """log P(y1, y2) under the Beta model; one row per match."""
    s0 = np.clip(s0, EPS, 1 - EPS)
    pw1w2 = s0 * (kappa * s0 + 1) / (kappa + 1)
    pw1l2 = s0 * (1 - s0) * kappa / (kappa + 1)
    pl1w2 = (1 - s0) * kappa * s0 / (kappa + 1)
    pl1l2 = (1 - s0) * (kappa * (1 - s0) + 1) / (kappa + 1)
    p = np.select([(y1 == 1) & (y2 == 1), (y1 == 1) & (y2 == 0),
                   (y1 == 0) & (y2 == 1)], [pw1w2, pw1l2, pl1w2], pl1l2)
    return np.log(np.clip(p, EPS, 1))


def set1_loglik(y1, s0):
    s0 = np.clip(s0, EPS, 1 - EPS)
    return np.where(y1 == 1, np.log(s0), np.log(1 - s0))


def kappa_curves(d: pd.DataFrame, best_of: int):
    """Per-match log-likelihood matrices over the kappa grid (n x K)."""
    mf = d.m_fav.to_numpy()
    y1, y2 = d.y1.to_numpy(), d.y2.to_numpy()
    s0s = np.column_stack([set_from_match_beta(mf, k, best_of) for k in KAPPAS])
    pair = np.column_stack([pair_loglik(y1, y2, s0s[:, j], KAPPAS[j])
                            for j in range(len(KAPPAS))])
    one = np.column_stack([set1_loglik(y1, s0s[:, j]) for j in range(len(KAPPAS))])
    return s0s, pair, one


def argmax_kappa(ll: np.ndarray) -> float:
    return float(KAPPAS[int(np.argmax(ll.sum(axis=0)))])


def boot_kappa(ll: np.ndarray) -> tuple[float, float, float]:
    pt = argmax_kappa(ll)
    idx = RNG.integers(0, len(ll), size=(N_BOOT, len(ll)))
    ks = np.array([argmax_kappa(ll[i]) for i in idx])
    return pt, *np.percentile(ks, [5, 95])


def fmt_k(k):
    return "inf" if k >= 1e5 else f"{k:g}"


# --------------------------------------------------------------------------- report
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", default=None, help="restrict to one book (default consensus)")
    ap.add_argument("--out", default=None, help="also write the report to this path")
    args = ap.parse_args()
    out_lines: list[str] = []

    def say(s=""):
        print(s)
        out_lines.append(s)

    con = connect()
    d = load(con, args.book)
    say(f"# Set-market prescreen  ({'consensus' if not args.book else args.book}, "
        f"ATP, {d.season.min()}-{d.season.max()})")

    for bo in (3, 5):
        b = d[(d.best_of == bo) & d.y1.notna()].copy()
        if len(b) < 500:
            continue
        pairs = b[b.y2.notna()].copy()
        say(f"\n## Best of {bo}   n(set 1 known)={len(b):,}  n(sets 1&2 known)={len(pairs):,}")

        # --- kappa three ways
        s0s_p, ll_pair, ll_one = kappa_curves(pairs, bo)
        s0s_1, _, ll_one_all = kappa_curves(b, bo)
        k_pair = boot_kappa(ll_pair)
        k_one = boot_kappa(ll_one_all)
        say(f"kappa from set-1/set-2 pairing : {fmt_k(k_pair[0])}  (90% boot {fmt_k(k_pair[1])}-{fmt_k(k_pair[2])})")
        say(f"kappa from set-1 marginal only : {fmt_k(k_one[0])}  (90% boot {fmt_k(k_one[1])}-{fmt_k(k_one[2])})")
        k_act = k_pair[0]

        # --- set-1 calibration, disjoint buckets, at independent and fitted kappa
        b["s_ind"] = set_from_match(b.m_fav.to_numpy(), bo)
        b["s_fit"] = set_from_match_beta(b.m_fav.to_numpy(), k_act, bo)
        say(f"\nSet-1 favourite win rate vs implied (disjoint match-price buckets)")
        say(f"{'m_fav bucket':>13} {'n':>7} {'indep s':>8} {'fit s0':>7} {'actual':>7} "
            f"{'95% CI':>15} {'act-ind':>8} {'act-fit':>8}")
        b["bk"] = pd.cut(b.m_fav, BUCKETS, right=False)
        for k, g in b.groupby("bk", observed=True):
            lo, hi = wilson(g.y1.sum(), len(g))
            say(f"{str(k):>13} {len(g):>7,} {g.s_ind.mean():>8.3f} {g.s_fit.mean():>7.3f} "
                f"{g.y1.mean():>7.3f} {lo:>7.3f}-{hi:.3f} {g.y1.mean()-g.s_ind.mean():>+8.3f} "
                f"{g.y1.mean()-g.s_fit.mean():>+8.3f}")
        mono = b.groupby("bk", observed=True).y1.mean().is_monotonic_increasing
        say(f"set-1 rate monotone in match price: {mono}")

        # --- set-2 conditional vs Beta and independence
        p = pairs.copy()
        p["s0"] = set_from_match_beta(p.m_fav.to_numpy(), k_act, bo)
        p["s_ind"] = set_from_match(p.m_fav.to_numpy(), bo)
        p["bk"] = pd.cut(p.m_fav, BUCKETS, right=False)
        say(f"\nSet 2 given set 1 (favourite). 'beta' = no-momentum prediction at kappa={fmt_k(k_act)}")
        say(f"{'m_fav bucket':>13} {'n':>7} | {'P(w2|w1)':>8} {'beta':>6} {'indep':>6} | "
            f"{'P(w2|l1)':>8} {'beta':>6} {'indep':>6}")
        for k, g in p.groupby("bk", observed=True):
            w, l = g[g.y1 == 1], g[g.y1 == 0]
            if len(w) < 30 or len(l) < 30:
                continue
            say(f"{str(k):>13} {len(g):>7,} | {w.y2.mean():>8.3f} "
                f"{((k_act*w.s0+1)/(k_act+1)).mean():>6.3f} {w.s_ind.mean():>6.3f} | "
                f"{l.y2.mean():>8.3f} {(k_act*l.s0/(k_act+1)).mean():>6.3f} {l.s_ind.mean():>6.3f}")

        # Pooled no-momentum residual: (actual - beta prediction) for set 2,
        # split by the set-1 result. A single Beta can fit one tail and miss the
        # other, so this is read as a shape check, and CIs are bootstrap.
        for lo_m in (0.5, 0.75):
            g = p[p.m_fav >= lo_m]
            for lab, sub, pred in (
                    ("after W1", g[g.y1 == 1], lambda x: (k_act * x.s0 + 1) / (k_act + 1)),
                    ("after L1", g[g.y1 == 0], lambda x: k_act * x.s0 / (k_act + 1))):
                if len(sub) < 200:
                    continue
                r = (sub.y2 - pred(sub)).to_numpy()
                bs = [r[RNG.integers(0, len(r), len(r))].mean() for _ in range(N_BOOT)]
                say(f"  m_fav>={lo_m:.2f} set-2 {lab}: n={len(sub):,}  actual-beta "
                    f"{r.mean():+.4f}  90% boot [{np.percentile(bs,5):+.4f}, {np.percentile(bs,95):+.4f}]")

        # --- does kappa look constant? (the Beta model says it should be)
        say("\nkappa (pairing) by segment -- constant if the Beta model is the right shape")
        for seg, g in pairs.groupby("seg"):
            if len(g) < 1500:
                continue
            _, llp, _ = kappa_curves(g, bo)
            kp = boot_kappa(llp)
            say(f"  {seg:<11} n={len(g):>6,}  kappa {fmt_k(kp[0]):>5}  "
                f"(90% {fmt_k(kp[1])}-{fmt_k(kp[2])})")
        if bo == 3:
            for surf, g in pairs.groupby("surface"):
                if len(g) < 1500:
                    continue
                _, llp, _ = kappa_curves(g, bo)
                kp = boot_kappa(llp)
                say(f"  {surf:<11} n={len(g):>6,}  kappa {fmt_k(kp[0]):>5}  "
                    f"(90% {fmt_k(kp[1])}-{fmt_k(kp[2])})")

        # --- walk-forward: kappa fitted before FIT_BEFORE, scored after
        tr, te = pairs[pairs.season < FIT_BEFORE], pairs[pairs.season >= FIT_BEFORE]
        if len(tr) > 2000 and len(te) > 1000:
            _, llt, _ = kappa_curves(tr, bo)
            k_tr = argmax_kappa(llt)
            mf, y1, y2 = te.m_fav.to_numpy(), te.y1.to_numpy(), te.y2.to_numpy()
            s_i = set_from_match(mf, bo)
            s_b = set_from_match_beta(mf, k_tr, bo)
            ll_i = pair_loglik(y1, y2, s_i, 1e6).mean()
            ll_b = pair_loglik(y1, y2, s_b, k_tr).mean()
            l1_i = set1_loglik(y1, s_i).mean()
            l1_b = set1_loglik(y1, s_b).mean()
            say(f"\nWalk-forward (kappa={fmt_k(k_tr)} from <{FIT_BEFORE}; n={len(te):,} later matches)")
            say(f"  set-1 log-lik   independent {l1_i:.5f}   beta {l1_b:.5f}   gain {l1_b-l1_i:+.5f}")
            say(f"  (set1,set2) ll  independent {ll_i:.5f}   beta {ll_b:.5f}   gain {ll_b-ll_i:+.5f}")

    # --- the market's own set structure: O/U 2.5 sets vs moneyline (bo3)
    t = d[(d.best_of == 3) & d.three_sets.notna() & d.p_three.notna()].copy()
    if len(t) > 1000:
        say(f"\n## O/U 2.5 sets vs moneyline (best of 3, completed; n={len(t):,})")
        mf = t.m_fav.to_numpy()

        def p3(k):
            s0 = set_from_match_beta(mf, k, 3)
            e2 = ((k * s0 + 1) * s0) / (k + 1)                 # E[s^2]
            e2c = ((k * (1 - s0) + 1) * (1 - s0)) / (k + 1)    # E[(1-s)^2]
            return 1.0 - (e2 + e2c)

        grid = KAPPAS
        sse_mkt = [np.mean((p3(k) - t.p_three.to_numpy()) ** 2) for k in grid]
        sse_act = [np.mean((p3(k) - t.three_sets.to_numpy()) ** 2) for k in grid]
        k_mkt, k_real = grid[int(np.argmin(sse_mkt))], grid[int(np.argmin(sse_act))]
        say(f"kappa implied by the O/U price : {fmt_k(k_mkt)}   "
            f"kappa implied by realised 3-set rate : {fmt_k(k_real)}")
        say(f"mean P(3 sets): market {t.p_three.mean():.4f}   actual {t.three_sets.mean():.4f}   "
            f"indep model {p3(1e6).mean():.4f}")
        say(f"\n{'m_fav bucket':>13} {'n':>7} {'market':>7} {'actual':>7} {'95% CI':>15} {'mkt-act':>8}")
        t["bk"] = pd.cut(t.m_fav, BUCKETS, right=False)
        for k, g in t.groupby("bk", observed=True):
            lo, hi = wilson(g.three_sets.sum(), len(g))
            say(f"{str(k):>13} {len(g):>7,} {g.p_three.mean():>7.3f} {g.three_sets.mean():>7.3f} "
                f"{lo:>7.3f}-{hi:.3f} {g.p_three.mean()-g.three_sets.mean():>+8.3f}")
        # Edge, if any: price vs outcome on the O/U. Flat-stake at the devigged
        # fair price is by construction zero, so report calibration gap with CI.
        diff = t.three_sets.to_numpy() - t.p_three.to_numpy()
        bs = [diff[RNG.integers(0, len(diff), len(diff))].mean() for _ in range(N_BOOT)]
        say(f"\nactual - market P(3 sets): {diff.mean():+.4f}  "
            f"90% boot [{np.percentile(bs,5):+.4f}, {np.percentile(bs,95):+.4f}]")
        br_m = np.mean((t.p_three - t.three_sets) ** 2)
        br_i = np.mean((p3(1e6) - t.three_sets) ** 2)
        br_f = np.mean((p3(k_real) - t.three_sets) ** 2)
        say(f"Brier on 3-set outcome: market {br_m:.5f}   moneyline+independence {br_i:.5f}   "
            f"moneyline+fitted-kappa {br_f:.5f}")

    say("\nReading this: an edge needs the set price to beat the *de-vigged* "
        "probability by more than the set-market margin (typically 5-8%), so "
        "calibration gaps of a point or two are noise against that bar.")
    if args.out:
        Path(args.out).write_text("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()
