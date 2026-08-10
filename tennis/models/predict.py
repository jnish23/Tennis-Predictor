"""Score hypothetical / upcoming matches with the production models.

Feature assembly is delegated to the same FeatureEngine used in training, so a
live prediction and a training row for the same fixture are built by identical
code. Anything else invites the two paths to drift apart silently.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tennis.config import ARTIFACTS
from tennis.db.schema import connect
from tennis.features.build import FeatureEngine
from tennis.models.common import CATEGORICAL, prepare


@dataclass
class MatchContext:
    surface: str = "Hard"
    level: str = "atp250"
    best_of: int = 3
    indoor: float = 0.0
    round: str = "R32"
    round_idx: float = 5.0
    draw_size: float = 32.0
    tourney_date: int = 20260803
    is_challenger: int = 0


class Predictor:
    def __init__(self, models=None, engine: FeatureEngine | None = None,
                 players: pd.DataFrame | None = None, as_of: int | None = None):
        """`as_of` (YYYYMMDD) caps rankings and ages to what was known then.

        Replaying a past tournament needs it: today's rankings already reflect
        that tournament's results, and feeding them back in would leak the
        outcome into the prediction.
        """
        if models is None:
            with open(ARTIFACTS / "models.pkl", "rb") as fh:
                models = pickle.load(fh)
        self.m = models
        self.engine = engine or FeatureEngine.load()
        self.as_of = as_of
        if players is None:
            con = connect()
            players = pd.read_sql(
                "SELECT player_id, name, hand, height_cm FROM players", con)
            cap = "" if as_of is None else f" AND as_of < {int(as_of)}"
            self.latest_rank = pd.read_sql(
                f"SELECT player_id, rank, rank_points FROM rankings r "
                f"WHERE as_of = (SELECT MAX(as_of) FROM rankings r2 "
                f"               WHERE r2.player_id=r.player_id{cap}){cap}",
                con,
            ).drop_duplicates("player_id").set_index("player_id")
            mcap = "" if as_of is None else f" WHERE tourney_date < {int(as_of)}"
            self.last_age = pd.read_sql(
                f"SELECT winner_id AS pid, winner_age AS age, tourney_date FROM matches{mcap} "
                f"UNION ALL SELECT loser_id, loser_age, tourney_date FROM matches{mcap}", con
            ).dropna().sort_values("tourney_date").groupby("pid").last()
            con.close()
        self.players = players.set_index("player_id")
        self._matrix_cache: dict = {}

    # -- helpers ----------------------------------------------------------
    def _attr(self, pid: str, col: str, default=np.nan):
        try:
            return self.players.at[pid, col]
        except KeyError:
            return default

    def _rank(self, pid: str) -> tuple[float, float]:
        if pid in self.latest_rank.index:
            r = self.latest_rank.loc[pid]
            return float(r["rank"]), float(r["rank_points"])
        return np.nan, np.nan

    def _age(self, pid: str, date: int) -> float:
        if pid not in self.last_age.index:
            return np.nan
        row = self.last_age.loc[pid]
        d0 = int(row["tourney_date"])
        years = (pd.Timestamp(str(date)) - pd.Timestamp(str(d0))).days / 365.25
        return float(row["age"]) + years

    def build_row(self, p1: str, p2: str, ctx: MatchContext) -> dict:
        r1, rp1 = self._rank(p1)
        r2, rp2 = self._rank(p2)
        synth = {
            "match_id": f"pred-{p1}-{p2}-{ctx.tourney_date}",
            "seq": -1,
            "tourney_date": ctx.tourney_date,
            "winner_id": p1, "loser_id": p2,     # orientation forced below
            "surface": ctx.surface, "level": ctx.level,
            "is_challenger": ctx.is_challenger, "best_of": ctx.best_of,
            "round_idx": ctx.round_idx, "draw_size": ctx.draw_size,
            "indoor": ctx.indoor,
            "winner_rank": r1, "loser_rank": r2,
            "winner_rank_points": rp1, "loser_rank_points": rp2,
            "winner_age": self._age(p1, ctx.tourney_date),
            "loser_age": self._age(p2, ctx.tourney_date),
            "winner_ht": self._attr(p1, "height_cm"),
            "loser_ht": self._attr(p2, "height_cm"),
            "winner_seed": np.nan, "loser_seed": np.nan,
            "winner_hand": self._attr(p1, "hand"), "loser_hand": self._attr(p2, "hand"),
        }
        return self.engine.features_for(synth, force_winner_first=True)

    # -- scoring ----------------------------------------------------------
    def _score(self, pairs: list[tuple[str, str]], ctx: MatchContext) -> dict:
        """Raw one-orientation scoring for a list of (p1, p2) pairs."""
        rows = [self.build_row(a, b, ctx) for a, b in pairs]
        df = prepare(pd.DataFrame(rows))
        extra = {}
        for c in self.m["abs_features"]:
            if c in df.columns:
                extra[f"abs_{c}"] = df[c].abs()
        for c in self.m["reg_feats"]:
            if c not in df.columns and c not in extra:
                extra[c] = np.nan
        if extra:
            # build in one shot; inserting ~10 columns individually fragments
            # the frame and pandas warns about it
            df = pd.concat([df, pd.DataFrame(extra, index=df.index)], axis=1)
        for c in CATEGORICAL:
            df[c] = df[c].astype("category")

        raw = self.m["winner"].predict(df[self.m["win_feats"]])
        return {
            "win": np.asarray(self.m["calibrator"].predict(raw), dtype=float),
            "total": np.asarray(self.m["totals"].predict(df[self.m["reg_feats"]]), dtype=float),
            "sets": np.asarray(self.m["totals_sets"].predict(df[self.m["reg_feats"]]), dtype=float),
            "spread": np.asarray(self.m["spread"].predict(df[self.m["reg_feats"]]), dtype=float),
        }

    def predict_many(self, pairs: list[tuple[str, str]], ctx: MatchContext,
                     *, symmetric: bool = True) -> pd.DataFrame:
        """Score fixtures, averaging both orientations by default.

        The trained model is not exactly orientation-invariant -- swapping the
        two sides gives probabilities that sum to ~0.95 rather than 1.0,
        because the tree ensemble uses the p1_* and p2_* feature blocks
        asymmetrically. Left alone that would make a dashboard answer depend on
        which player the user typed first. Scoring both ways and averaging
        removes the artefact (and slightly reduces variance):

            P(A beats B) = [ p(A,B) + (1 - p(B,A)) ] / 2

        Totals are order-free so they average directly; the spread is
        sign-flipped before averaging.
        """
        if not pairs:
            return pd.DataFrame()
        fwd = self._score(pairs, ctx)
        if symmetric:
            rev = self._score([(b, a) for a, b in pairs], ctx)
            win = (fwd["win"] + (1.0 - rev["win"])) / 2.0
            total = (fwd["total"] + rev["total"]) / 2.0
            sets = (fwd["sets"] + rev["sets"]) / 2.0
            spread = (fwd["spread"] - rev["spread"]) / 2.0
        else:
            win, total = fwd["win"], fwd["total"]
            sets, spread = fwd["sets"], fwd["spread"]

        out = pd.DataFrame({
            "p1_id": [a for a, _ in pairs],
            "p2_id": [b for _, b in pairs],
            "p1_win_prob": win,
            "pred_total_games": total,
            "pred_total_sets": sets,
            "pred_spread": spread,
        })
        out["p1_name"] = [self._attr(a, "name") for a, _ in pairs]
        out["p2_name"] = [self._attr(b, "name") for _, b in pairs]
        return out

    def win_prob(self, p1: str, p2: str, ctx: MatchContext) -> float:
        return float(self.predict_many([(p1, p2)], ctx)["p1_win_prob"].iloc[0])

    def win_prob_matrix(self, player_ids: list[str], ctx: MatchContext) -> np.ndarray:
        """P[i][j] = probability player i beats player j, for every pairing.

        The simulator needs every possible meeting in the bracket, so we score
        them all once up front rather than per simulated match.
        """
        n = len(player_ids)
        # Scoring 128 players is ~8k pairs and takes several seconds, and callers
        # legitimately ask for the same matrix repeatedly -- simulating the same
        # draw from successive rounds, or re-running with a different sim count.
        # The result depends only on the player set, the context and this
        # Predictor's frozen state, so it is safe to memoise.
        key = (tuple(player_ids), ctx.surface, ctx.level, ctx.best_of,
               ctx.indoor, ctx.draw_size, ctx.tourney_date, ctx.is_challenger)
        cached = self._matrix_cache.get(key)
        if cached is not None:
            return cached.copy()

        # Only unordered pairs: predict_many already averages both orientations,
        # so P(j beats i) is just the complement and needs no second call.
        pairs = [(player_ids[i], player_ids[j])
                 for i in range(n) for j in range(i + 1, n)]
        mat = np.full((n, n), 0.5)
        if pairs:
            preds = self.predict_many(pairs, ctx)
            idx = {p: i for i, p in enumerate(player_ids)}
            for (a, b), p in zip(pairs, preds["p1_win_prob"]):
                mat[idx[a], idx[b]] = p
                mat[idx[b], idx[a]] = 1.0 - p
        np.fill_diagonal(mat, 0.5)
        self._matrix_cache[key] = mat
        return mat.copy()
