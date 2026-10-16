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
    for raw_tok in s.split():
        # A bracketed score, "[10-7]", is a match tiebreak played in place of a
        # deciding set. It is a set won for the sets count but a single game
        # for the games count -- scoring it 10-7 made a 17-game "set" and fed
        # that to the totals model as a label.
        is_mtb = raw_tok.startswith("[")
        t = _TOKEN_CLEAN.sub("", raw_tok).strip()
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
        if is_mtb:
            tiebreaks += 1
            wg, lg = (1, 0) if wg > lg else (0, 1)
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
    # A best-of-5 record showing only two sets won is an unfinished match, not a
    # short one, and is no totals label. The reverse is not symmetric: three
    # sets won cannot happen in a best-of-3, so that is a mislabelled best_of
    # (the score is the more reliable witness) and the match is complete.
    target = {3: (2, 3), 5: (3,)}.get(int(best_of) if best_of == best_of and best_of else 0,
                                    (2, 3))
    ps.usable_for_totals = (
        status == COMPLETED
        and not malformed
        and w_sets > l_sets
        and w_sets in target
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


def _set_complete(wg: int, lg: int) -> bool:
    """A finished set, as opposed to the in-progress score a retirement leaves."""
    hi, lo = max(wg, lg), min(wg, lg)
    return hi >= 6 and (hi - lo >= 2 or (hi == 7 and lo == 6))


def set_winners(score: str | float | None) -> list[int]:
    """Per-set outcome as 1 if the *match winner* took the set, else 0.

    Stops at the first set that is not demonstrably finished: "6-3 2-1 RET"
    yields one set, not two, so a set-2 label is never invented from a game
    score that was interrupted. Set markets settle on the set itself, so a
    retirement later in the match does not void an earlier set -- which is why
    this reads the score rather than the match `status`.
    """
    out: list[int] = []
    for wg, lg, _ in score_sets(score):
        if not _set_complete(wg, lg):
            break
        out.append(1 if wg > lg else 0)
    return out


def tiebreaks_won(score: str | float | None) -> tuple[int, int]:
    """Tiebreaks taken by (match winner, match loser), from the score string.

    A 7-6 set was decided by a tiebreak and its winner is simply the side with
    the 7; the score is winner-first, so "6-7(4)" is a tiebreak the *loser*
    took. A bracketed "[10-7]" match tiebreak, played in place of a deciding
    set, counts too. Sets that are not demonstrably finished (the in-progress
    "6-6" a retirement can leave) are not counted: no tiebreak was completed.
    """
    if score is None or (isinstance(score, float) and score != score):
        return 0, 0
    w = l = 0
    for tok in str(score).split():
        bracketed = tok.startswith("[")
        t = _TOKEN_CLEAN.sub("", tok).strip()
        m = _SET_RE.match(t)
        if not m:
            continue
        a, b = int(m.group(1)), int(m.group(2))
        if bracketed or (max(a, b) >= 10 and abs(a - b) >= 2):
            tb = a != b            # a match tiebreak, however it is written
        else:
            tb = (a, b) in ((7, 6), (6, 7))
        if tb:
            if a > b:
                w += 1
            else:
                l += 1
    return w, l
