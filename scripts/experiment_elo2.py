"""Round two: the winners from experiment_elo.py, refined and combined.

Adds the two ideas the first sweep did not separate:

* **inactivity decay** rather than blanket yearly regression. Regressing every
  player at the year boundary also penalises the ones who played 60 matches,
  which is not what the idea is for. Decaying by *time since that player last
  played* targets the actual problem -- a rating going stale while its owner is
  injured -- and leaves active players alone.
* **margin of victory**, scaling K by how one-sided the result was, so 6-0 6-0
  moves a rating further than 7-6 7-6.

Run:  python scripts/experiment_elo2.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tennis.db.schema import connect  # noqa: E402

BASE = 1500.0
SCORE_FROM = 2010


def load() -> pd.DataFrame:
    con = connect()
    try:
        df = pd.read_sql(
            """SELECT m.seq, m.tourney_date, m.winner_id, m.loser_id,
                      t.surface, m.status, m.total_games, m.game_margin
               FROM matches m JOIN tournaments t USING(tourney_key)
               ORDER BY m.seq""", con)
    finally:
        con.close()
    df = df[df["status"] != "walkover"].copy()
    df["season"] = df["tourney_date"] // 10000
    df["surface"] = df["surface"].fillna("").replace("", "Unknown")
    df["date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")
    df["days"] = (df["date"] - df["date"].min()).dt.days.fillna(0).astype(int)
    return df


@dataclass
class Cfg:
    name: str
    k_num: float = 250.0
    k_shift: float = 5.0
    k_exp: float = 0.4
    surf_prior: bool = False
    surf_blend_n: float = 0.0
    # Rating decays toward BASE by this fraction per 365 idle days.
    idle_decay: float = 0.0
    # Days of inactivity that cost nothing. Below this the rating is untouched,
    # so a player on a normal 2-week schedule is never taxed and only a genuine
    # layoff decays. Zero means every day of the gap counts, which is really
    # time-decay rather than idleness.
    idle_grace: float = 0.0
    mov: bool = False          # scale K by margin of victory
    # Drive decay off the player's OVERALL last-played date, including for
    # surface ratings. A player active on hard courts has not lost their clay
    # ability through inactivity -- they have simply not played clay. Keying
    # surface decay to surface gaps taxes an eight-month clay off-season as if
    # it were an injury.
    decay_by_overall: bool = False


def run(df: pd.DataFrame, c: Cfg) -> dict:
    r: dict[str, float] = {}
    n: dict[str, int] = {}
    last: dict[str, int] = {}

    W = df["winner_id"].to_numpy()
    L = df["loser_id"].to_numpy()
    SF = df["surface"].to_numpy()
    S = df["season"].to_numpy()
    D = df["days"].to_numpy()
    TG = df["total_games"].to_numpy(dtype=float)
    GM = df["game_margin"].to_numpy(dtype=float)

    def fetch(key, day, fallback=BASE, clock=None):
        v = r.get(key, fallback)
        ck = clock if (c.decay_by_overall and clock) else key
        if c.idle_decay and key in last and ck in last:
            idle = (day - last[ck] - c.idle_grace) / 365.0
            if idle > 0:
                v = BASE + (v - BASE) * ((1.0 - c.idle_decay) ** idle)
        return v

    ll_o, ll_s, acc = [], [], []
    for i in range(len(df)):
        w, l, sf, s, day = W[i], L[i], SF[i], S[i], D[i]
        ko, lo = f"{w}\x00overall", f"{l}\x00overall"
        rw, rl = fetch(ko, day), fetch(lo, day)
        ew = 1.0 / (1.0 + 10.0 ** ((rl - rw) / 400.0))

        if sf != "Unknown":
            kws, lws = f"{w}\x00{sf}", f"{l}\x00{sf}"
            nw, nl = n.get(kws, 0), n.get(lws, 0)
            if c.surf_prior:
                aw = nw / (nw + c.surf_blend_n) if c.surf_blend_n else 1.0
                al = nl / (nl + c.surf_blend_n) if c.surf_blend_n else 1.0
                sw = aw * fetch(kws, day, rw, ko) + (1 - aw) * rw
                sl = al * fetch(lws, day, rl, lo) + (1 - al) * rl
            else:
                sw, sl = fetch(kws, day, clock=ko), fetch(lws, day, clock=lo)
            es = 1.0 / (1.0 + 10.0 ** ((sl - sw) / 400.0))
        else:
            kws = lws = None
            es = ew

        if s >= SCORE_FROM:
            ll_o.append(-np.log(max(ew, 1e-15)))
            ll_s.append(-np.log(max(es, 1e-15)))
            acc.append(1.0 if ew >= 0.5 else 0.0)

        mult = 1.0
        if c.mov and TG[i] and TG[i] > 0 and not np.isnan(GM[i]):
            # log of the game margin, normalised so a typical match sits at 1.0
            mult = np.log1p(max(GM[i], 0.0)) / np.log1p(5.0)
            mult = float(np.clip(mult, 0.5, 1.8))

        kw = c.k_num / ((n.get(ko, 0) + c.k_shift) ** c.k_exp) * mult
        kl = c.k_num / ((n.get(lo, 0) + c.k_shift) ** c.k_exp) * mult
        r[ko] = rw + kw * (1 - ew)
        r[lo] = rl - kl * (1 - ew)
        n[ko] = n.get(ko, 0) + 1
        n[lo] = n.get(lo, 0) + 1
        last[ko] = last[lo] = day

        if kws is not None:
            rws = fetch(kws, day, rw if c.surf_prior and n.get(kws, 0) == 0 else BASE, ko)
            rls = fetch(lws, day, rl if c.surf_prior and n.get(lws, 0) == 0 else BASE, lo)
            esr = 1.0 / (1.0 + 10.0 ** ((rls - rws) / 400.0))
            kw2 = c.k_num / ((n.get(kws, 0) + c.k_shift) ** c.k_exp) * mult
            kl2 = c.k_num / ((n.get(lws, 0) + c.k_shift) ** c.k_exp) * mult
            r[kws] = rws + kw2 * (1 - esr)
            r[lws] = rls - kl2 * (1 - esr)
            n[kws] = n.get(kws, 0) + 1
            n[lws] = n.get(lws, 0) + 1
            last[kws] = last[lws] = day

    return {
        "variant": c.name,
        "ll_overall": round(float(np.mean(ll_o)), 5),
        "ll_surface": round(float(np.mean(ll_s)), 5),
        "accuracy": round(float(np.mean(acc)), 5),
    }


def main() -> None:
    df = load()
    print(f"{len(df):,} rated matches, scoring {SCORE_FROM}+\n")

    cfgs = [
        Cfg("baseline (current)"),
        Cfg("K=200", k_num=200),
        Cfg("surf prior n=40", surf_prior=True, surf_blend_n=40),
        Cfg("surf prior n=60", surf_prior=True, surf_blend_n=60),
        Cfg("surf prior n=80", surf_prior=True, surf_blend_n=80),
        Cfg("surf prior n=120", surf_prior=True, surf_blend_n=120),
    ]
    for d in (0.10, 0.20, 0.35, 0.50):
        cfgs.append(Cfg(f"time decay {d:.0%}/yr", idle_decay=d))
    for g in (60, 120, 180):
        for d in (0.20, 0.40):
            cfgs.append(Cfg(f"layoff decay {d:.0%}/yr past {g}d",
                            idle_decay=d, idle_grace=g))
    for k in (170, 200, 230):
        cfgs.append(Cfg(f"K={k} + surf n=60", k_num=k, surf_prior=True,
                        surf_blend_n=60))
    cfgs.append(Cfg("K=200 + surf n=60 + layoff 40%/120d", k_num=200,
                    surf_prior=True, surf_blend_n=60,
                    idle_decay=0.40, idle_grace=120))
    for g in (90, 120, 180):
        for d in (0.25, 0.40, 0.60):
            cfgs.append(Cfg(f"K200+surf60, layoff {d:.0%}/{g}d (overall clock)",
                            k_num=200, surf_prior=True, surf_blend_n=60,
                            idle_decay=d, idle_grace=g, decay_by_overall=True))
    cfgs += [
        Cfg("MoV weighting", mov=True),
        Cfg("K=200 + surf prior n=60", k_num=200, surf_prior=True, surf_blend_n=60),
        Cfg("K=200 + surf n=60 + idle 20%", k_num=200, surf_prior=True,
            surf_blend_n=60, idle_decay=0.20),
        Cfg("K=200 + surf n=60 + idle 20% + MoV", k_num=200, surf_prior=True,
            surf_blend_n=60, idle_decay=0.20, mov=True),
    ]

    rows = [run(df, c) for c in cfgs]
    out = pd.DataFrame(rows)
    base = out[out.variant == "baseline (current)"].iloc[0]
    out["Δ overall"] = (base.ll_overall - out.ll_overall).round(5)
    out["Δ surface"] = (base.ll_surface - out.ll_surface).round(5)
    print(out.sort_values("ll_surface").to_string(index=False))
    print("\n(Δ positive = better than current)")


if __name__ == "__main__":
    main()
