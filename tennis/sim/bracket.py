"""Monte Carlo tournament simulation.

Per CLAUDE.md this is explicitly *not* a deterministic round-by-round chain --
advancing the higher-probability player each round collapses the uncertainty
into one fictional bracket by round 3 or 4. Instead every simulated playthrough
resolves each match by a random draw weighted by the model's probability, and
we aggregate over thousands of playthroughs.

Standing default, also from CLAUDE.md: features are frozen at pre-tournament
values for the whole run. The win-probability matrix is therefore computed once
up front and reused for every simulated meeting, rather than being recomputed
as a simulated player accumulates simulated matches.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tennis.models.predict import MatchContext, Predictor

BYE = "__BYE__"


@dataclass
class Draw:
    """A first-round bracket. `slots` is seeded order, length a power of two.

    Non-power-of-two fields are padded with BYE entries, which is how real
    32-in-a-48-draw events work: the top seeds advance without playing.
    """

    name: str
    slots: list[str]
    ctx: MatchContext
    player_names: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        n = len(self.slots)
        size = 1 << max(1, math.ceil(math.log2(max(n, 2))))
        if size != n:
            self.slots = self.slots + [BYE] * (size - n)

    @property
    def size(self) -> int:
        return len(self.slots)

    @property
    def n_rounds(self) -> int:
        return int(math.log2(self.size))


def round_names(n_rounds: int) -> list[str]:
    """Label rounds by what winning that round means (reached next stage)."""
    base = {1: ["Champion"], 2: ["Final", "Champion"],
            3: ["SF", "Final", "Champion"],
            4: ["QF", "SF", "Final", "Champion"],
            5: ["R16", "QF", "SF", "Final", "Champion"],
            6: ["R32", "R16", "QF", "SF", "Final", "Champion"],
            7: ["R64", "R32", "R16", "QF", "SF", "Final", "Champion"]}
    if n_rounds in base:
        return base[n_rounds]
    extra = [f"R{2**(n_rounds-i)}" for i in range(n_rounds - 7)]
    return extra + base[7]


def playing_round_names(n_rounds: int) -> list[str]:
    """Label rounds by the round *being played*, the way the tour names them.

    Distinct from `round_names`, and the two must not be swapped. That one
    names the stage a player has *reached* by winning round r, which is what
    `simulate` counts and what `_actual_progression` joins against column for
    column. This one names the round r itself, which is what a bracket shows.
    They differ by exactly one step, so using the reached-stage names on a
    bracket labels the opening round of a 128-draw "R64" and shifts every
    other column with it -- the round that is really the R16 ends up captioned
    "QF". Cincinnati is a 96-draw padded to 128 slots and showed precisely
    that.

    Naming follows the field size, so it matches the `round` vocabulary in the
    matches table (R128, R64, ..., QF, SF, F) for any draw size.
    """
    special = {8: "QF", 4: "SF", 2: "F"}
    return [special.get(2 ** (n_rounds - r), f"R{2 ** (n_rounds - r)}")
            for r in range(n_rounds)]


def simulate(draw: Draw, predictor: Predictor, n_sims: int = 10_000,
             seed: int | None = 42,
             resolved: dict[int, dict[int, str]] | None = None) -> pd.DataFrame:
    """Run `n_sims` playthroughs; return per-player per-round reach probability.

    `resolved` pins matches that have actually been played:
    ``{round_index: {pair_index: winner_player_id}}``. Those are not simulated —
    the real winner advances in every playthrough. Everything after them is still
    drawn at random, so the output is the distribution *conditional on the
    tournament so far*, which is what you want once a round is in the books.
    Without this, a player already knocked out would keep showing a chance of
    reaching later rounds.
    """
    rng = np.random.default_rng(seed)
    real = [p for p in draw.slots if p != BYE]
    uniq = list(dict.fromkeys(real))
    if len(uniq) < 2:
        raise ValueError("draw needs at least two distinct players")

    prob = predictor.win_prob_matrix(uniq, draw.ctx)
    idx = {p: i for i, p in enumerate(uniq)}

    size = draw.size
    n_rounds = draw.n_rounds
    # field[s, k] = index into `uniq` of the player in slot k of simulation s;
    # -1 marks a bye.
    field_arr = np.array([idx.get(p, -1) for p in draw.slots], dtype=np.int32)
    field_arr = np.tile(field_arr, (n_sims, 1))

    labels = round_names(n_rounds)
    reached = np.zeros((len(uniq), n_rounds), dtype=np.int64)

    resolved = resolved or {}
    cur = field_arr
    for r in range(n_rounds):
        a, b = cur[:, 0::2], cur[:, 1::2]
        # Probability a beats b, vectorised over every simulation and match.
        pa = np.where((a >= 0) & (b >= 0), prob[np.clip(a, 0, None),
                                               np.clip(b, 0, None)], 1.0)
        draws = rng.random(pa.shape)
        a_wins = np.where(b < 0, True, np.where(a < 0, False, draws < pa))
        nxt = np.where(a_wins, a, b)

        # Overwrite the drawn outcome wherever the match was actually played.
        for pair_i, winner in resolved.get(r, {}).items():
            if pair_i >= nxt.shape[1]:
                continue
            wi = idx.get(winner, -1)
            if wi < 0:
                continue
            nxt[:, pair_i] = wi
        # Count everyone who advanced out of this round.
        flat = nxt[nxt >= 0]
        if flat.size:
            counts = np.bincount(flat, minlength=len(uniq))
            reached[:, r] += counts
        cur = nxt

    rows = []
    for p, i in idx.items():
        for r, label in enumerate(labels):
            rows.append({
                "player_id": p,
                "player_name": draw.player_names.get(p, p),
                "round": label,
                "prob": reached[i, r] / n_sims,
            })
    out = pd.DataFrame(rows)
    wide = out.pivot(index=["player_id", "player_name"], columns="round",
                     values="prob").reset_index()
    # drop the pivot's residual column-index name, which otherwise shows up as a
    # stray "round" header when the frame is rendered
    return wide.rename_axis(None, axis=1)


def simulate_to_long(draw: Draw, predictor: Predictor, n_sims: int = 10_000,
                     seed: int | None = 42) -> pd.DataFrame:
    wide = simulate(draw, predictor, n_sims, seed)
    labels = [c for c in wide.columns if c not in ("player_id", "player_name")]
    return wide.melt(id_vars=["player_id", "player_name"], value_vars=labels,
                     var_name="round", value_name="prob")


def expected_match_lines(draw: Draw, predictor: Predictor) -> pd.DataFrame:
    """First-round totals/spread lines.

    Deliberately limited to round 1. CLAUDE.md notes that extending totals and
    spread round-by-round compounds error fast, because each later round is
    conditioned on an opponent who is themselves uncertain. Round 1 is the only
    round where both players are known, so it is the only one where these lines
    mean what they appear to mean.
    """
    pairs = [(draw.slots[i], draw.slots[i + 1])
             for i in range(0, draw.size, 2)
             if draw.slots[i] != BYE and draw.slots[i + 1] != BYE]
    if not pairs:
        return pd.DataFrame()
    out = predictor.predict_many(pairs, draw.ctx)
    out["round"] = "R1"
    return out


# --------------------------------------------------------------------------
# progressive bracket: real matchups only, filled in as results arrive
# --------------------------------------------------------------------------
def bracket_state(draw: Draw, predictor, resolved: dict | None = None,
                  scores: dict | None = None) -> list[dict]:
    """Round-by-round view of the bracket as it currently stands.

    Each round is a list of ties. A tie is one of:

    * **played**   -- both players known and the match is in `resolved`. Still
                      carries the model's pre-match probability, plus
                      `winner_prob` (what it gave the actual winner) and
                      `correct`.
    * **live**     -- both players known, not yet played; carries the model's
                      win probability, and totals/spread for the pairing
    * **pending**  -- one or both sides still to be decided ("TBD")

    Deliberately does *not* invent a projected opponent for undecided ties.
    Naming a single likely occupant and pricing that matchup is the
    deterministic-chain error CLAUDE.md rules out; those slots stay TBD until a
    real result fills them.
    """
    resolved = resolved or {}
    # {(winner_id, loser_id): "7-6(3) 6-4"} -- supplied by whoever knows the
    # results, since the simulator itself has no notion of a scoreline.
    scores = scores or {}
    # The round being played, not the stage its winners reach -- see
    # `playing_round_names`. A bracket column is the round itself.
    labels = playing_round_names(draw.n_rounds)
    field = list(draw.slots)

    # First pass: work out the shape and gather the ties that need pricing.
    rounds: list[dict] = []
    to_price: list[tuple[str, str]] = []
    for r in range(draw.n_rounds):
        ties, nxt = [], []
        for i in range(0, len(field), 2):
            a, b = field[i], field[i + 1]
            winner = resolved.get(r, {}).get(i // 2)
            if a == BYE and b == BYE:
                w = BYE
            elif b == BYE and a not in (None, BYE):
                w = a
            elif a == BYE and b not in (None, BYE):
                w = b
            else:
                w = winner
            known = a not in (None, BYE) and b not in (None, BYE)
            # Price played ties too, not just upcoming ones: seeing what the
            # model said about a match that has already happened is how you
            # judge it. A pre-match probability is still a pre-match
            # probability once the result is in.
            if known:
                to_price.append((a, b))
            tie = {"pair": i // 2, "p1": a, "p2": b, "winner": w,
                   "state": "played" if winner else
                            ("live" if known else "pending")}
            if winner is not None:
                loser = b if winner == a else a
                tie["score"] = scores.get((winner, loser))
            ties.append(tie)
            nxt.append(w)
        rounds.append({"round": labels[r], "index": r, "ties": ties})
        field = nxt

    # Second pass: one batched prediction call for every live tie.
    priced = {}
    if to_price and predictor is not None:
        preds = predictor.predict_many(to_price, draw.ctx)
        for row in preds.itertuples():
            priced[(row.p1_id, row.p2_id)] = {
                "p1_win_prob": float(row.p1_win_prob),
                "total_games": float(row.pred_total_games),
                "spread": float(row.pred_spread),
            }
    for rd in rounds:
        for tie in rd["ties"]:
            if tie["state"] in ("live", "played"):
                tie.update(priced.get((tie["p1"], tie["p2"]), {}))
            if tie["state"] == "played" and tie.get("p1_win_prob") is not None:
                # Probability the model gave the player who actually won. One
                # number says everything: above 0.5 the model was right, and
                # how far below tells you whether it was a coin flip or a
                # genuine upset.
                p1 = tie["p1_win_prob"]
                tie["winner_prob"] = p1 if tie["winner"] == tie["p1"] else 1 - p1
                tie["correct"] = tie["winner_prob"] >= 0.5
    return rounds


def bracket_frame(draw: Draw, rounds: list[dict]) -> "pd.DataFrame":
    """Flatten `bracket_state` into a table, names resolved, for display."""
    name = lambda p: ("BYE" if p == BYE else                       # noqa: E731
                      "TBD" if p is None else draw.player_names.get(p, p))
    out = []
    for rd in rounds:
        for tie in rd["ties"]:
            if tie["p1"] == BYE and tie["p2"] == BYE:
                continue
            prob = tie.get("p1_win_prob")
            out.append({
                "Round": rd["round"],
                "Tie": tie["pair"] + 1,
                "Player 1": name(tie["p1"]),
                "Player 2": name(tie["p2"]),
                "State": tie["state"],
                "P1 win %": round(prob * 100, 1) if prob is not None else None,
                "Total games": (round(tie["total_games"], 1)
                                if tie.get("total_games") is not None else None),
                "Winner": name(tie["winner"]) if tie["winner"] else "",
                "Score": tie.get("score") or "",
                "Model gave winner %": (round(tie["winner_prob"] * 100, 1)
                                        if tie.get("winner_prob") is not None else None),
                "Called it": ("" if tie.get("correct") is None
                              else ("yes" if tie["correct"] else "no")),
            })
    return pd.DataFrame(out)
