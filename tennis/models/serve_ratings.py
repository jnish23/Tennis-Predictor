"""Opponent-adjusted, shrunk serve and return ratings.

A raw rolling serve percentage is the wrong input to a Markov chain, and the
backtest shows exactly how wrong: fed straight in, the chain is calibrated at
slope 0.34 -- it says 8% where reality is 29%, and 92% where reality is 72%.
Two separate defects cause that, and both are fixed here rather than papered
over with a calibration layer.

**Opponent adjustment.** A raw rate confounds how well a player serves with who
they served against. Serving at 66% against a field of poor returners is not
the same evidence as 66% against good ones, and in a Challenger-heavy database
the strength of the field varies enormously. Ratings here are therefore
relative: each player carries a serve rating and a *separate* return rating,
both as deviations from tour average, and a match updates the server's serve
rating and the returner's return rating against what the pair predicted.

**Shrinkage.** The chain is steeply convex, so plugging in a point estimate of
a noisy quantity does not give the average outcome -- it gives an extreme one.
A player with four matches of history should be pulled hard toward average, and
that is what the sample-size-weighted update does: ratings start at zero and
move by an amount that shrinks as evidence accumulates, so a small sample never
produces a large deviation.

Updates are online and strictly after the fact, single pass in match order, so
a match is always rated using only what was known before it -- the same
discipline as the Elo state and for the same reason.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Learning rate on the serve/return ratings. Chosen by the sweep in
# scripts/backtest_markov.py rather than by taste; the objective is flat
# enough between 0.02 and 0.06 that the exact value is not load-bearing.
K = 0.04

# Points of evidence before a rating is trusted at full weight. Below this the
# update is damped, which is the shrinkage that keeps thin histories near
# average instead of at the extremes the chain would amplify.
PRIOR_POINTS = 400.0


@dataclass
class _P:
    serve: float = 0.0          # deviation from tour-average serve points won
    ret: float = 0.0            # deviation from tour-average return points won
    n_serve: float = 0.0        # serve points seen, for shrinkage
    n_ret: float = 0.0
    surf: dict = field(default_factory=dict)   # surface -> (serve, ret, n)


def _shrunk(rating: float, n: float) -> float:
    """Pull a rating toward zero until enough points support it."""
    return rating * n / (n + PRIOR_POINTS)


def fit(matches: pd.DataFrame, avg: float, *, k: float = K,
        surface_weight: float = 0.35) -> pd.DataFrame:
    """Walk forward through matches, emitting each one's *pre-match* ratings.

    `matches` must be sorted in play order and carry winner_id, loser_id,
    surface and the serve counters. Returns one row per match with the four
    ratings as they stood before it was played.
    """
    st: dict[str, _P] = {}
    out = []

    def get(pid: str) -> _P:
        p = st.get(pid)
        if p is None:
            p = st[pid] = _P()
        return p

    def rating(p: _P, surface: str) -> tuple[float, float]:
        """Blend the overall rating with the surface-specific one."""
        s, r = _shrunk(p.serve, p.n_serve), _shrunk(p.ret, p.n_ret)
        sv = p.surf.get(surface)
        if sv:
            ss, sr, sn = sv
            w = surface_weight * sn / (sn + PRIOR_POINTS)
            s = (1 - w) * s + w * _shrunk(ss, sn)
            r = (1 - w) * r + w * _shrunk(sr, sn)
        return s, r

    for row in matches.itertuples():
        w, l = row.winner_id, row.loser_id
        surface = row.surface or "Hard"
        pw, pl = get(w), get(l)
        w_s, w_r = rating(pw, surface)
        l_s, l_r = rating(pl, surface)
        out.append({"match_id": row.match_id,
                    "w_serve": w_s, "w_ret": w_r,
                    "l_serve": l_s, "l_ret": l_r})

        # Update only from matches that carry the counters. Guard with an
        # explicit NaN test, never `x or 0`: NaN is truthy, so `NaN or 0`
        # returns NaN, and a single missing counter then poisons that player's
        # rating and spreads to every opponent they subsequently meet. That is
        # not hypothetical -- it silently NaN'd the entire rating pool.
        vals = (row.w_svpt, row.l_svpt, row.w_1stWon, row.w_2ndWon,
                row.l_1stWon, row.l_2ndWon)
        if any(v is None or not np.isfinite(v) for v in vals):
            continue
        w_pts, l_pts = float(row.w_svpt), float(row.l_svpt)
        if w_pts <= 0 or l_pts <= 0:
            continue
        w_won = float(row.w_1stWon) + float(row.w_2ndWon)
        l_won = float(row.l_1stWon) + float(row.l_2ndWon)

        for srv, ret, pts, won, s_rat, r_rat in (
                (pw, pl, w_pts, w_won, w_s, l_r),
                (pl, pw, l_pts, l_won, l_s, w_r)):
            expected = avg + s_rat - r_rat
            actual = won / pts
            err = actual - expected
            # Weight by points played: a three-set match is more evidence than
            # a retirement after four games.
            wt = min(pts / 100.0, 1.5)
            srv.serve += k * err * wt
            ret.ret -= k * err * wt
            srv.n_serve += pts
            ret.n_ret += pts
            for who, idx in ((srv, 0), (ret, 1)):
                ss, sr, sn = who.surf.get(surface, (0.0, 0.0, 0.0))
                if idx == 0:
                    ss += k * err * wt
                else:
                    sr -= k * err * wt
                who.surf[surface] = (ss, sr, sn + pts)

    return pd.DataFrame(out)


def to_p1p2(ratings: pd.DataFrame, y_win: np.ndarray) -> pd.DataFrame:
    """Re-orient winner/loser ratings onto the backtest's p1/p2 frame."""
    w = y_win == 1
    return pd.DataFrame({
        "match_id": ratings.match_id.to_numpy(),
        "p1_serve": np.where(w, ratings.w_serve, ratings.l_serve),
        "p1_ret": np.where(w, ratings.w_ret, ratings.l_ret),
        "p2_serve": np.where(w, ratings.l_serve, ratings.w_serve),
        "p2_ret": np.where(w, ratings.l_ret, ratings.w_ret),
    })
