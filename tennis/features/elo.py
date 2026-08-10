"""Elo ratings, overall and per surface.

Ratings are pure running state: a match is *rated* using the values held before
it, and only then does it update them. That ordering is what keeps Elo out of
the leakage category, so callers must feed matches in chronological order.

K decays with a player's match count so established players move slowly while
newcomers converge quickly (the shape used by FiveThirtyEight's tennis Elo).

Three tuned behaviours, each measured standalone against 124,899 matches from
2010 on (`scripts/experiment_elo2.py`):

**Surface prior.** A surface rating is seeded from the player's *overall*
rating rather than 1500, then blended toward the surface-specific value as
surface matches accumulate. Without this the surface book knows nothing about a
player until it has watched them for years -- a top seed's first clay match
rated them 1500, the same as a qualifier -- and the surface probability came out
*worse* than ignoring surface entirely (0.6366 against 0.6297). With it, 0.6254.

**Layoff decay.** A rating regresses toward base only after a genuine absence,
past a grace period. Blanket regression at the season boundary, or plain time
decay, both hurt at every rate tested: they compress the whole rating scale and
make predictions underconfident, which costs more than the staleness they fix.

The grace period is keyed to the player's **overall** last match, including
when decaying surface ratings. Keyed per surface, the eight-month gap between
clay seasons reads as an injury, and that alone gave back the entire surface
gain. A player active on hard courts has not lost their clay ability.
"""
from __future__ import annotations

from dataclasses import dataclass, field

BASE_RATING = 1500.0
K_NUM = 200.0
K_SHIFT = 5.0
K_EXP = 0.4
SURFACES = ("Hard", "Clay", "Grass", "Carpet")

# Surface matches at which a surface rating stands on its own; below it the
# rating is a blend of the surface and overall books.
SURF_BLEND_N = 60.0
# Fraction of the gap to BASE_RATING surrendered per 365 days of absence...
IDLE_DECAY = 0.25
# ...counting only days beyond this, so an ordinary two-week schedule is never
# taxed and only a real layoff decays.
IDLE_GRACE_DAYS = 90.0


def expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


def k_factor(n: int) -> float:
    return K_NUM / ((n + K_SHIFT) ** K_EXP)


@dataclass
class EloBook:
    """Overall + per-surface ratings for every player."""

    ratings: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    # Day-number of each player's most recent match, on the overall clock.
    last_day: dict[str, int] = field(default_factory=dict)

    @staticmethod
    def _key(pid: str, scope: str) -> str:
        return f"{pid}\x00{scope}"

    def _decayed(self, key: str, pid: str, day: int | None, base: float) -> float:
        """Rating after any layoff decay owed as of `day`."""
        r = self.ratings.get(key, base)
        if day is None:
            return r
        seen = self.last_day.get(pid)
        if seen is None:
            return r
        idle = (day - seen - IDLE_GRACE_DAYS) / 365.0
        if idle <= 0:
            return r
        return BASE_RATING + (r - BASE_RATING) * ((1.0 - IDLE_DECAY) ** idle)

    def get(self, pid: str, scope: str = "overall", day: int | None = None) -> float:
        """Rating as it stands for a match played on `day`.

        For a surface scope this is the blended figure -- what the model should
        actually see -- not the raw surface number.
        """
        overall = self._decayed(self._key(pid, "overall"), pid, day, BASE_RATING)
        if scope == "overall":
            return overall
        key = self._key(pid, scope)
        surf = self._decayed(key, pid, day, overall)
        n = self.counts.get(key, 0)
        w = n / (n + SURF_BLEND_N)
        return w * surf + (1.0 - w) * overall

    def n(self, pid: str, scope: str = "overall") -> int:
        return self.counts.get(self._key(pid, scope), 0)

    def update(self, winner: str, loser: str, scope: str = "overall",
               day: int | None = None) -> tuple[float, float]:
        """Apply one result; returns the pre-match ratings that were used.

        Note the asymmetry with `get`: the *stored* surface rating updates on
        its own raw value (seeded from overall the first time it is touched),
        while `get` returns the blend. Updating the blend would fold the overall
        rating into the surface book permanently and the two would converge.
        """
        wk, lk = self._key(winner, scope), self._key(loser, scope)
        w_overall = self._decayed(self._key(winner, "overall"), winner, day, BASE_RATING)
        l_overall = self._decayed(self._key(loser, "overall"), loser, day, BASE_RATING)
        rw = self._decayed(wk, winner, day, w_overall)
        rl = self._decayed(lk, loser, day, l_overall)
        ew = expected(rw, rl)
        kw = k_factor(self.counts.get(wk, 0))
        kl = k_factor(self.counts.get(lk, 0))
        self.ratings[wk] = rw + kw * (1.0 - ew)
        self.ratings[lk] = rl - kl * (1.0 - ew)
        self.counts[wk] = self.counts.get(wk, 0) + 1
        self.counts[lk] = self.counts.get(lk, 0) + 1
        return rw, rl

    def touch(self, pid: str, day: int | None) -> None:
        """Mark a player as having played on `day`, stopping the layoff clock.

        Called once per match after every scope has updated -- the clock is
        shared, so moving it while surface updates are still pending would
        cancel the decay those updates are owed.
        """
        if day is not None:
            self.last_day[pid] = int(day)

    # -- persistence ------------------------------------------------------
    def to_rows(self, last_seq: int) -> list[tuple]:
        out = []
        for key, rating in self.ratings.items():
            pid, scope = key.split("\x00")
            out.append((pid, scope, rating, self.counts.get(key, 0), last_seq))
        return out

    @classmethod
    def from_rows(cls, rows) -> "EloBook":
        book = cls()
        for pid, scope, rating, n, _ in rows:
            k = cls._key(pid, scope)
            book.ratings[k] = float(rating)
            book.counts[k] = int(n)
        return book
