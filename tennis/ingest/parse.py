"""Score-string parsing and match-status classification.

The `score` column carries both the result and how the match ended. Retirements,
walkovers and defaults are *not* dropped: they are labelled so downstream code
can decide per target. A retirement is a valid win for the winner model but the
games played are truncated, so it must be excluded from the totals/spread
targets rather than silently treated as a completed match.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Status vocabulary
COMPLETED = "completed"
RETIRED = "retired"        # RET - one player stopped mid-match
WALKOVER = "walkover"      # W/O - never started
DEFAULT = "default"        # DEF - disqualified
UNFINISHED = "unfinished"  # abandoned / unknown / unparseable

_SET_RE = re.compile(r"^(\d{1,2})-(\d{1,2})(?:\((\d+)\))?$")
_TOKEN_CLEAN = re.compile(r"[\[\]]")


@dataclass
class ParsedScore:
    status: str
    sets: list[tuple[int, int]] = field(default_factory=list)  # (winner, loser) games
    w_sets: int = 0
    l_sets: int = 0
    w_games: int = 0
    l_games: int = 0
    tiebreaks: int = 0
    # True only when the games/sets totals describe a full, naturally-ended
    # match and are therefore safe to use as a totals/spread label.
    usable_for_totals: bool = False

    @property
    def total_games(self) -> int:
        return self.w_games + self.l_games

    @property
    def total_sets(self) -> int:
        return self.w_sets + self.l_sets

    @property
    def game_margin(self) -> int:
        """Winner games minus loser games (the spread target)."""
        return self.w_games - self.l_games


def parse_score(score: str | float | None, best_of: int | None = None) -> ParsedScore:
    if score is None or (isinstance(score, float) and score != score):
        return ParsedScore(UNFINISHED)

    s = str(score).strip()
    if not s:
        return ParsedScore(UNFINISHED)

    upper = s.upper()
    status = COMPLETED
    if "W/O" in upper or "WO" == upper.replace(" ", ""):
        return ParsedScore(WALKOVER)
    if "DEF" in upper:
        status = DEFAULT
    elif "RET" in upper:
        status = RETIRED
    elif "ABN" in upper or "ABD" in upper or "UNFINISHED" in upper:
        status = UNFINISHED

    sets: list[tuple[int, int]] = []
    tiebreaks = 0
    malformed = False
    for tok in _TOKEN_CLEAN.sub("", s).split():
        t = tok.strip()
        if not t:
            continue
        if t.upper() in {"RET", "DEF", "W/O", "ABN", "ABD", "(RET)", "-"}:
            continue
        m = _SET_RE.match(t)
        if not m:
            malformed = True
            continue
        wg, lg = int(m.group(1)), int(m.group(2))
        # Guard against corrupt tokens like "60-0".
        if wg > 30 or lg > 30:
            malformed = True
            continue
        if m.group(3) is not None:
            tiebreaks += 1
        sets.append((wg, lg))

    if not sets:
        return ParsedScore(status if status != COMPLETED else UNFINISHED)

    w_sets = sum(1 for a, b in sets if a > b)
    l_sets = sum(1 for a, b in sets if b > a)
    w_games = sum(a for a, _ in sets)
    l_games = sum(b for _, b in sets)

    ps = ParsedScore(
        status=status,
        sets=sets,
        w_sets=w_sets,
        l_sets=l_sets,
        w_games=w_games,
        l_games=l_games,
        tiebreaks=tiebreaks,
    )

    # A completed match must show the winner taking the last set and reaching a
    # plausible set target. We check against {2,3} rather than the declared
    # best_of because a few hundred rows carry a mislabelled best_of; the score
    # itself is the more reliable witness.
    ps.usable_for_totals = (
        status == COMPLETED
        and not malformed
        and w_sets > l_sets
        and w_sets in (2, 3)
        and l_sets <= 2
        and len(sets) <= 5
        and sets[-1][0] > sets[-1][1]  # winner took the final set
    )
    return ps


def round_is_qualifying(rnd: str | None) -> bool:
    return bool(rnd) and str(rnd).upper().startswith("Q")


def score_sets(score: str | float | None) -> list[tuple[int, int, int | None]]:
    """Per-set games for display, as (winner_games, loser_games, tiebreak_pts).

    Always winner-first, matching how the sources record a scoreline. The
    tiebreak figure is the points the *set loser* took, which is the number
    conventionally shown as a superscript beside their 6.

    `parse_score` counts tiebreaks but throws the points away, so this is a
    separate pass rather than a change to the label-producing parser -- nothing
    a model consumes should shift because a display format was added.
    """
    if score is None or (isinstance(score, float) and score != score):
        return []
    out: list[tuple[int, int, int | None]] = []
    for tok in _TOKEN_CLEAN.sub("", str(score)).split():
        t = tok.strip()
        if not t or t.upper() in {"RET", "DEF", "W/O", "ABN", "ABD", "(RET)", "-"}:
            continue
        m = _SET_RE.match(t)
        if not m:
            continue
        wg, lg = int(m.group(1)), int(m.group(2))
        if wg > 30 or lg > 30:
            continue
        tb = int(m.group(3)) if m.group(3) is not None else None
        out.append((wg, lg, tb))
    return out
