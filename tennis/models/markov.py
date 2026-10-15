"""Hierarchical Markov chain: point -> game -> set -> match.

The classical tennis model (Barnett & Clarke 2005, O'Malley 2008). Instead of
learning P(win) directly, it takes one number per player -- the probability of
winning a point on serve -- and propagates it up through the scoring structure
analytically. Everything here is exact given that input; no simulation, no
fitting.

**The assumption that makes it work, and its cost.** Points are treated as
independent and identically distributed: the server's chance is the same at
0-0, at 30-40, and at 5-6 in the fifth. That is known to be false -- servers
win slightly more of the points that matter -- but the error is small and
partially self-cancelling, and no free data source carries the point-by-point
sequences needed to model the alternative. That limitation is the honest
ceiling on this approach here, not a detail to fix later with the same inputs.

**Why it can beat a classifier at the same information.** The scoring system
is a huge amplifier: a 5-point edge on serve (0.62 vs 0.67) is worth about 12
points of match probability, and the mapping is steeply non-linear near the
top. A gradient-booster has to learn that curve from outcome labels; this gets
it exactly for free, and spends its data budget only on estimating the serve
numbers. It is also naturally consistent across best-of-3 and best-of-5, which
a single classifier trained on both is not.

Everything is vectorised over numpy arrays: the state spaces are tiny and
fixed, so the dynamic programmes run as array operations across all matches at
once rather than per-match Python loops.
"""
from __future__ import annotations

import numpy as np

# ATP-wide mean fraction of serve points won. Used to combine a server's rate
# with a returner's into one matchup number; measured from our own data rather
# than taken from the literature (see `tour_average_spw`).
DEFAULT_SPW = 0.6386


def p_game(p: np.ndarray | float) -> np.ndarray:
    """P(server holds) given point-win probability `p` on serve.

    Exact. Win to love, 15, 30 by direct count; from deuce by the standard
    two-point-lead geometric sum, which collapses to p^2 / (p^2 + q^2).
    """
    p = np.asarray(p, dtype=float)
    q = 1.0 - p
    # 4-0, 4-1, 4-2: the loser's points can fall in C(3,0), C(4,1), C(5,2) ways
    straight = p**4 * (1.0 + 4.0 * q + 10.0 * q**2)
    # 3-3 reached in C(6,3)=20 ways, then a two-point-lead race from deuce
    denom = p**2 + q**2
    deuce = np.divide(p**2, denom, out=np.full_like(p, 0.5), where=denom > 0)
    return straight + 20.0 * p**3 * q**3 * deuce


def _two_point_race(pa: np.ndarray, pb: np.ndarray) -> np.ndarray:
    """P(A wins a 2-point-lead race) where A serves first of each pair.

    From 6-6 in a tiebreak the serve order repeats A,B,A,B..., so the race is a
    sequence of independent pairs: A wins both with pa(1-pb), loses both with
    (1-pa)pb, and anything else returns to the same state.
    """
    a = pa * (1.0 - pb)
    b = (1.0 - pa) * pb
    tot = a + b
    return np.divide(a, tot, out=np.full_like(a, 0.5), where=tot > 0)


def p_tiebreak(pa: np.ndarray | float, pb: np.ndarray | float,
               target: int = 7) -> np.ndarray:
    """P(A wins a tiebreak) with A serving the first point.

    `pa` is A's point-win probability on A's own serve, `pb` is B's on B's.
    Serve order is 1 then alternating pairs, which is what makes a tiebreak
    resist the closed forms that work for games -- so this is a dynamic
    programme over (A points, B points), carried as arrays so every match in
    the batch is solved simultaneously.

    `target` is 7 for a standard tiebreak; pass 10 for a match tiebreak.
    """
    pa = np.asarray(pa, dtype=float)
    pb = np.asarray(pb, dtype=float)
    shape = np.broadcast(pa, pb).shape
    states = {(0, 0): np.ones(shape)}
    win = np.zeros(shape)

    # Sweep by total points played so every predecessor is resolved first.
    for n in range(0, 2 * target):
        nxt: dict[tuple[int, int], np.ndarray] = {}
        for (i, j), prob in states.items():
            if i + j != n:
                nxt[(i, j)] = prob
                continue
            # Points 1; 2,3; 4,5; ... -- A serves when ((n+1)//2) is even.
            a_serves = ((n + 1) // 2) % 2 == 0
            p_a_wins_pt = pa if a_serves else 1.0 - pb
            for won, nx in ((p_a_wins_pt, (i + 1, j)),
                            (1.0 - p_a_wins_pt, (i, j + 1))):
                ni, nj = nx
                # Decided only at >= target with a two-point margin.
                if ni >= target and ni - nj >= 2:
                    win = win + prob * won
                    continue
                if nj >= target and nj - ni >= 2:
                    continue
                if ni >= target - 1 and nj >= target - 1:
                    # 6-6 (or 9-9): a two-point race with A serving first.
                    win = win + prob * won * _two_point_race(pa, pb)
                    continue
                nxt[nx] = nxt.get(nx, 0.0) + prob * won
        states = nxt
    return win


def p_set(pa: np.ndarray | float, pb: np.ndarray | float, *,
          a_serves_first: bool = True, tiebreak: bool = True) -> np.ndarray:
    """P(A wins a set), given each player's point-win probability on serve.

    Games are won by whoever is serving with probability `p_game`, so a set is
    a dynamic programme over (A games, B games) with the server alternating.
    At 6-6 the set goes to a tiebreak unless `tiebreak` is False, in which case
    it becomes an advantage set decided by a two-game race.
    """
    pa = np.asarray(pa, dtype=float)
    pb = np.asarray(pb, dtype=float)
    hold_a, hold_b = p_game(pa), p_game(pb)
    shape = np.broadcast(pa, pb).shape
    states = {(0, 0): np.ones(shape)}
    win = np.zeros(shape)

    for n in range(0, 13):
        nxt: dict[tuple[int, int], np.ndarray] = {}
        for (i, j), prob in states.items():
            if i + j != n:
                nxt[(i, j)] = prob
                continue
            a_on_serve = (n % 2 == 0) == a_serves_first
            p_a_wins_game = hold_a if a_on_serve else 1.0 - hold_b
            for won, nx in ((p_a_wins_game, (i + 1, j)),
                            (1.0 - p_a_wins_game, (i, j + 1))):
                ni, nj = nx
                if ni >= 6 and ni - nj >= 2:
                    win = win + prob * won
                    continue
                if nj >= 6 and nj - ni >= 2:
                    continue
                if ni == 6 and nj == 6:
                    if tiebreak:
                        # Whoever returned the last game serves the tiebreak.
                        tb_a_first = ((12 % 2 == 0) == a_serves_first)
                        tb = (p_tiebreak(pa, pb) if tb_a_first
                              else 1.0 - p_tiebreak(pb, pa))
                        win = win + prob * won * tb
                    else:
                        win = win + prob * won * _two_point_race(hold_a, hold_b)
                    continue
                nxt[nx] = nxt.get(nx, 0.0) + prob * won
        states = nxt
    return win


def p_match(pa: np.ndarray | float, pb: np.ndarray | float,
            best_of: np.ndarray | int = 3, **kw) -> np.ndarray:
    """P(A wins the match) from each player's point-win probability on serve.

    Sets are treated as independent draws with the same per-set probability,
    which is the standard simplification: it ignores that the serve order
    carries across a set boundary, worth well under a tenth of a point of match
    probability and far smaller than the error in the serve estimates.
    """
    s = np.asarray(p_set(pa, pb, **kw), dtype=float)
    best_of = np.asarray(best_of)
    # Best of 3: win 2-0 or 2-1. Best of 5: 3-0, 3-1 or 3-2.
    bo3 = s**2 * (3.0 - 2.0 * s)
    bo5 = s**3 * (10.0 - 15.0 * s + 6.0 * s**2)
    # Clip to [0, 1]. The polynomials are exact in real arithmetic but not in
    # floating point: at s just under 1 the best-of-five form rounds to
    # 1 + 2.2e-16, and a probability one ulp above 1 turns log(1 - p) into NaN
    # in any scoring code downstream.
    return np.clip(np.where(best_of >= 5, bo5, bo3), 0.0, 1.0)


def match_from_set(s: np.ndarray | float, best_of: np.ndarray | int = 3) -> np.ndarray:
    """P(match) from a per-set win probability, sets independent."""
    s = np.asarray(s, dtype=float)
    bo3 = s**2 * (3.0 - 2.0 * s)
    bo5 = s**3 * (10.0 - 15.0 * s + 6.0 * s**2)
    return np.clip(np.where(np.asarray(best_of) >= 5, bo5, bo3), 0.0, 1.0)


def _bisect_increasing(f, target: np.ndarray, iters: int = 60) -> np.ndarray:
    lo, hi = np.zeros_like(target), np.ones_like(target)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        up = f(mid) < target
        lo, hi = np.where(up, mid, lo), np.where(up, hi, mid)
    return 0.5 * (lo + hi)


def set_from_match(m: np.ndarray | float, best_of: np.ndarray | int = 3) -> np.ndarray:
    """Invert `match_from_set`: the per-set probability a match price implies.

    Under independent sets a 71.4% best-of-three favourite is a ~64.5% set
    favourite -- which is why a "-180 set" line against a "-250 match" line is
    fair, not underpriced. Both polynomials are monotone on [0, 1], so bisection
    is exact and vectorises over a whole frame.
    """
    m = np.atleast_1d(np.asarray(m, dtype=float))
    bo = np.broadcast_to(np.asarray(best_of), m.shape)
    return _bisect_increasing(lambda s: match_from_set(s, bo), m)


def _beta_moment(n: int, s0: np.ndarray, kappa: float) -> np.ndarray:
    """E[s^n] for s ~ Beta(kappa*s0, kappa*(1-s0)); kappa=inf is a point mass."""
    if not np.isfinite(kappa):
        return s0**n
    a = kappa * s0
    out = np.ones_like(s0)
    for i in range(n):
        out = out * (a + i) / (kappa + i)
    return out


def match_from_set_beta(s0: np.ndarray, kappa: float,
                        best_of: np.ndarray | int = 3) -> np.ndarray:
    """P(match) when the per-set probability is itself uncertain.

    A match is not five independent coin flips at a known bias; the bias is
    uncertain (surface, form, matchup), which makes sets positively correlated.
    `kappa` is the Beta concentration: large means sets are independent, small
    means the first set is highly informative about the rest.
    """
    s0 = np.asarray(s0, dtype=float)
    m3 = 3 * _beta_moment(2, s0, kappa) - 2 * _beta_moment(3, s0, kappa)
    m5 = (10 * _beta_moment(3, s0, kappa) - 15 * _beta_moment(4, s0, kappa)
          + 6 * _beta_moment(5, s0, kappa))
    return np.where(np.asarray(best_of) >= 5, m5, m3)


def set_from_match_beta(m: np.ndarray, kappa: float,
                        best_of: np.ndarray | int = 3) -> np.ndarray:
    """Marginal set-1 win probability consistent with match price `m` and `kappa`."""
    m = np.atleast_1d(np.asarray(m, dtype=float))
    bo = np.broadcast_to(np.asarray(best_of), m.shape)
    return _bisect_increasing(lambda s: match_from_set_beta(s, kappa, bo), m)


def matchup_spw(serve_a: np.ndarray, return_b: np.ndarray,
                avg: float = DEFAULT_SPW) -> np.ndarray:
    """Combine A's serve rate and B's return rate into A's point-win rate.

    The standard additive decomposition: a player's serve rating is their
    surplus over tour average, a returner's is theirs, and a matchup is the
    average plus the server's surplus minus the returner's. Written out,

        p = serve_a - return_b + (1 - avg)

    which reduces to `avg` when both are exactly average. Clipped away from the
    extremes because the chain is very steep near the ends: a serve rate of 0.9
    from a short noisy window would otherwise imply a near-certain match.
    """
    p = np.asarray(serve_a, float) - np.asarray(return_b, float) + (1.0 - avg)
    return np.clip(p, 0.30, 0.90)


def tour_average_spw(con, tour: str = "atp") -> float:
    """Mean serve points won on one tour, for the decomposition above.

    Scoped by tour, and it has to be: the ATP holds serve on 62.3% of points
    and the WTA on 56.5%. A pooled average is wrong for both, and wrong in the
    worst place -- it is the origin the whole chain is anchored to, so every
    rating is measured as a deviation from it and every probability is
    amplified away from it by the scoring structure.
    """
    row = con.execute(
        "SELECT SUM(w_1stWon + w_2ndWon + l_1stWon + l_2ndWon) * 1.0 "
        "/ SUM(w_svpt + l_svpt) FROM matches "
        "WHERE w_svpt > 0 AND l_svpt > 0 AND tour = ?", (tour,)).fetchone()
    return float(row[0]) if row and row[0] else DEFAULT_SPW
