"""Turning bookmaker prices into probabilities.

Every market comparison this project makes rests on this step, and the obvious
method is the one that is wrong.

**Proportional** (what we used everywhere until now) just rescales the implied
probabilities so they sum to one. That assumes the margin is spread evenly
across outcomes. It is not: books load the margin onto longshots, so
proportional over-states an underdog's chance and under-states a favourite's.
Measured directly on our own data -- backing every favourite returned -5.4%
against a 6.7% vig while backing every underdog returned -11.5% -- the bias is
around three points a side and it survives proportional devigging untouched.
That flows straight into the ensemble regressions and the model-vs-market
comparisons, biasing all of them.

**Shin** models the book as protecting itself against a fraction `z` of
insider traders, which produces exactly the favourite-longshot shape as a
consequence rather than as a fudge. **Power** is the atheoretical alternative:
raise each implied probability to a common exponent. Both are one-dimensional
root finds and both are cheap.

None of these is assumed better here. `compare_methods` scores them against
realised outcomes so the choice is made on log loss, not on literature.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def _as_pairs(prices: np.ndarray) -> np.ndarray:
    a = np.asarray(prices, dtype=float)
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError("expected an (n, 2) array of decimal prices")
    return a


def proportional(prices: np.ndarray) -> np.ndarray:
    """Scale implied probabilities to sum to one. Fast, and biased."""
    inv = 1.0 / _as_pairs(prices)
    return inv / inv.sum(axis=1, keepdims=True)


def shin(prices: np.ndarray, *, tol: float = 1e-10,
         max_iter: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Shin (1993) devig. Returns (probabilities, z).

    Solves for the insider fraction `z` that makes the implied probabilities
    sum to one, using

        p_i(z) = [sqrt(z^2 + 4(1-z) * pi_i^2 / PI) - z] / (2(1-z))

    with pi_i = 1/odds_i and PI their sum. At z = 0 this returns pi_i / sqrt(PI),
    which sums to sqrt(PI) > 1 for any real book, so the root always lies at
    z > 0; sum(p) falls monotonically in z, which is what makes plain bisection
    safe here and avoids a scipy dependency in a hot path.
    """
    a = _as_pairs(prices)
    pi = 1.0 / a
    PI = pi.sum(axis=1, keepdims=True)

    def probs(z):
        z = np.clip(z, 0.0, 1.0 - 1e-9)
        return (np.sqrt(z ** 2 + 4.0 * (1.0 - z) * pi ** 2 / PI) - z) / (2.0 * (1.0 - z))

    lo = np.zeros((len(a), 1))
    hi = np.full((len(a), 1), 0.5)
    # Expand the bracket for the rare book whose margin needs z > 0.5.
    for _ in range(20):
        need = probs(hi).sum(axis=1, keepdims=True) > 1.0
        if not need.any():
            break
        hi = np.where(need, np.minimum(hi * 2.0, 1.0 - 1e-9), hi)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        s = probs(mid).sum(axis=1, keepdims=True)
        too_high = s > 1.0
        lo = np.where(too_high, mid, lo)
        hi = np.where(too_high, hi, mid)
        if np.all(hi - lo < tol):
            break
    z = 0.5 * (lo + hi)
    p = probs(z)
    return p / p.sum(axis=1, keepdims=True), z.ravel()


def power(prices: np.ndarray, *, tol: float = 1e-10,
          max_iter: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Power devig: p_i proportional to (1/odds_i) ** k, solving for k.

    No behavioural story behind it -- it is a shape that happens to bend the
    same way the bias does, which makes it a useful foil for Shin rather than a
    rival explanation.
    """
    a = _as_pairs(prices)
    pi = 1.0 / a
    lo = np.full((len(a), 1), 0.2)
    hi = np.full((len(a), 1), 3.0)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        s = (pi ** mid).sum(axis=1, keepdims=True)
        # sum decreases as k grows (each pi < 1), so overshoot means raise k
        too_high = s > 1.0
        lo = np.where(too_high, mid, lo)
        hi = np.where(too_high, hi, mid)
        if np.all(hi - lo < tol):
            break
    k = 0.5 * (lo + hi)
    p = pi ** k
    return p / p.sum(axis=1, keepdims=True), k.ravel()


METHODS = {"proportional": proportional,
           "shin": lambda x: shin(x)[0],
           "power": lambda x: power(x)[0]}


def compare_methods(prices: np.ndarray, won_first: np.ndarray) -> "object":
    """Score every method against realised outcomes.

    `won_first` is 1 when the first column's side won. Log loss decides, not
    the literature: a devig that models the bias correctly should predict
    outcomes better, and if it does not on this data then it is not an
    improvement here whatever its provenance.
    """
    import pandas as pd

    y = np.asarray(won_first, dtype=float)
    rows = []
    for name, fn in METHODS.items():
        p = np.clip(fn(prices)[:, 0], EPS, 1 - EPS)
        rows.append({
            "method": name,
            "log_loss": round(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 6),
            "brier": round(float(np.mean((p - y) ** 2)), 6),
            "mean_fav_prob": round(float(np.mean(np.maximum(p, 1 - p))), 5),
        })
    return pd.DataFrame(rows).sort_values("log_loss")
