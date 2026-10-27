"""Uncertainty that respects how matches cluster.

Matches in one tournament share a court, its pace, the weather and a draw in
which the same few players recur round after round, and a ladder of bets on one
match settles on a single scoreline. Treating bets or matches as independent
understates the error; resampling whole tournaments does not.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def tourney_key_of(match_id) -> pd.Series:
    """The event a match belongs to, recovered from its id.

    ids are `<tourney_key>-<round>-<winner>-<loser>` and neither player id
    contains a hyphen, so the key is everything before the last three hyphens.
    """
    return pd.Series(match_id).astype(str).str.rsplit("-", n=3).str[0]


def cluster_bootstrap_ratio(num, den, clusters, *, n: int = 2000, seed: int = 0,
                            level: float = 0.95) -> tuple[float, float]:
    """Percentile interval for sum(num) / sum(den), resampling whole clusters.

    ROI is profit over turnover, so `num` is profit and `den` is stake; a mean
    is the case `den = 1`.
    """
    num, den = np.asarray(num, float), np.asarray(den, float)
    codes, uniq = pd.factorize(np.asarray(clusters))
    k = len(uniq)
    if k < 2:
        return float("nan"), float("nan")
    sn = np.bincount(codes, weights=num, minlength=k)
    sd = np.bincount(codes, weights=den, minlength=k)
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    # In chunks: n x k indices at once is 32 MB per thousand clusters per 1k draws.
    step = max(1, int(2e7 // k))
    for a in range(0, n, step):
        b = min(n, a + step)
        idx = rng.integers(0, k, (b - a, k))
        out[a:b] = sn[idx].sum(axis=1) / sd[idx].sum(axis=1)
    lo, hi = np.percentile(out, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def cluster_bootstrap_mean(x, clusters, **kw) -> tuple[float, float]:
    x = np.asarray(x, float)
    return cluster_bootstrap_ratio(x, np.ones(len(x)), clusters, **kw)
