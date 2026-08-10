"""Elo variant sweep: does a better Elo exist inside the data we already hold?

Elo is scored *standalone* here -- log loss of the Elo probability itself, on
matches from 2010 on, with everything earlier acting as burn-in. That isolates
the rating system from LightGBM, which would otherwise paper over a weak Elo by
leaning on the other 160-odd features. A variant has to earn its place here
before it is worth a full retrain.

Run:  python scripts/experiment_elo.py
"""
from __future__ import annotations

import itertools
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
                      t.surface, m.status, m.best_of
               FROM matches m JOIN tournaments t USING(tourney_key)
               ORDER BY m.seq""", con)
    finally:
        con.close()
    # Walkovers never happened on court and must not move a rating.
    df = df[df["status"] != "walkover"].copy()
    df["season"] = df["tourney_date"] // 10000
    df["surface"] = df["surface"].fillna("").replace("", "Unknown")
    return df


@dataclass
class Cfg:
    name: str
    k_num: float = 250.0
    k_shift: float = 5.0
    k_exp: float = 0.4
    # Fraction of the way back to 1500 applied at each season boundary.
    # 0 = current behaviour (ratings persist untouched forever).
    regress: float = 0.0
    # Surface rating seeded from the player's overall rating rather than 1500,
    # and blended with it until enough surface matches accumulate.
    surf_prior: bool = False
    surf_blend_n: float = 0.0   # matches at which surface rating stands alone


def run(df: pd.DataFrame, c: Cfg) -> dict:
    r: dict[str, float] = {}
    n: dict[str, int] = {}

    def rating(key):
        return r.get(key, BASE)

    ll_o, ll_s, acc_o, m = [], [], [], 0
    season = None
    seq_w = df["winner_id"].to_numpy()
    seq_l = df["loser_id"].to_numpy()
    surf = df["surface"].to_numpy()
    seas = df["season"].to_numpy()

    for i in range(len(df)):
        w, l, sf, s = seq_w[i], seq_l[i], surf[i], seas[i]

        if c.regress and season is not None and s != season:
            # Regress every rating toward the mean at the year boundary. A year
            # off court should cost a player rating; holding 2200 through an
            # absence is the clearest way Elo goes stale.
            for k in r:
                r[k] = BASE + (r[k] - BASE) * (1.0 - c.regress)
        season = s

        ko, lo = f"{w}\x00overall", f"{l}\x00overall"
        rw, rl = rating(ko), rating(lo)
        ew = 1.0 / (1.0 + 10.0 ** ((rl - rw) / 400.0))

        # surface-aware probability
        if sf != "Unknown":
            kws, lws = f"{w}\x00{sf}", f"{l}\x00{sf}"
            if c.surf_prior:
                # Unseen surface starts at the player's overall rating, then
                # shrinks toward the surface-specific value as evidence builds.
                nw, nl = n.get(kws, 0), n.get(lws, 0)
                aw = nw / (nw + c.surf_blend_n) if c.surf_blend_n else 1.0
                al = nl / (nl + c.surf_blend_n) if c.surf_blend_n else 1.0
                sw = aw * r.get(kws, rw) + (1 - aw) * rw
                sl = al * r.get(lws, rl) + (1 - al) * rl
            else:
                sw, sl = rating(kws), rating(lws)
            es = 1.0 / (1.0 + 10.0 ** ((sl - sw) / 400.0))
        else:
            kws = lws = None
            es = ew

        if s >= SCORE_FROM:
            ll_o.append(-np.log(max(ew, 1e-15)))
            ll_s.append(-np.log(max(es, 1e-15)))
            acc_o.append(1.0 if ew >= 0.5 else 0.0)
            m += 1

        kw = c.k_num / ((n.get(ko, 0) + c.k_shift) ** c.k_exp)
        kl = c.k_num / ((n.get(lo, 0) + c.k_shift) ** c.k_exp)
        r[ko] = rw + kw * (1 - ew)
        r[lo] = rl - kl * (1 - ew)
        n[ko] = n.get(ko, 0) + 1
        n[lo] = n.get(lo, 0) + 1

        if kws is not None:
            rws, rls = rating(kws), rating(lws)
            if c.surf_prior and n.get(kws, 0) == 0:
                rws = rw          # seed a new surface rating from overall
            if c.surf_prior and n.get(lws, 0) == 0:
                rls = rl
            esr = 1.0 / (1.0 + 10.0 ** ((rls - rws) / 400.0))
            kws_ = c.k_num / ((n.get(kws, 0) + c.k_shift) ** c.k_exp)
            kls_ = c.k_num / ((n.get(lws, 0) + c.k_shift) ** c.k_exp)
            r[kws] = rws + kws_ * (1 - esr)
            r[lws] = rls - kls_ * (1 - esr)
            n[kws] = n.get(kws, 0) + 1
            n[lws] = n.get(lws, 0) + 1

    return {
        "variant": c.name,
        "n": m,
        "logloss_overall": round(float(np.mean(ll_o)), 5),
        "logloss_surface": round(float(np.mean(ll_s)), 5),
        "accuracy": round(float(np.mean(acc_o)), 5),
    }


def main() -> None:
    df = load()
    print(f"{len(df):,} rated matches, scoring {SCORE_FROM}+\n")

    cfgs = [Cfg("baseline (current)")]
    for kn, ke in itertools.product((150, 200, 250, 300), (0.3, 0.4, 0.5)):
        if (kn, ke) == (250, 0.4):
            continue
        cfgs.append(Cfg(f"K num={kn} exp={ke}", k_num=kn, k_exp=ke))
    for g in (0.02, 0.05, 0.10, 0.15, 0.25):
        cfgs.append(Cfg(f"regress {g:.0%}/season", regress=g))
    for b in (0, 10, 20, 40):
        cfgs.append(Cfg(f"surface prior, blend n={b}", surf_prior=True,
                        surf_blend_n=b))

    rows = [run(df, c) for c in cfgs]
    out = pd.DataFrame(rows).sort_values("logloss_overall")
    print(out.to_string(index=False))

    base = next(r for r in rows if r["variant"] == "baseline (current)")
    print(f"\nbaseline overall log loss {base['logloss_overall']}, "
          f"surface {base['logloss_surface']}")
    best = out.iloc[0]
    print(f"best overall: {best['variant']} "
          f"({base['logloss_overall'] - best['logloss_overall']:+.5f})")
    bs = out.sort_values("logloss_surface").iloc[0]
    print(f"best surface: {bs['variant']} "
          f"({base['logloss_surface'] - bs['logloss_surface']:+.5f})")


if __name__ == "__main__":
    main()
