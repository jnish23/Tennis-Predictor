"""Controlled test: frozen vs in-tournament-updated features.

Conditioning is held IDENTICAL in both arms (same played matches pinned), so the
only difference is whether player features were rebuilt to include the rounds
already played. Scored only on rounds still to come, against what happened.

This is deliberately NOT a pre-tournament-vs-conditional comparison. Simulating
from a later round always scores better because less is unknown; that measures
how much information you supplied, not model quality. Holding the pinned rounds
fixed and varying only feature recency isolates the question worth asking.

    python scripts/experiment_feature_recency.py [--events N] [--sims N]

Runtime is roughly 20 minutes for the full 39 events now that the win-probability
matrix is memoised (it was ~55 before, because the frozen arm recomputed an
identical 8k-pair matrix for every round).

Results land in artifacts/feature_recency_results.csv.
"""
import argparse
import sys, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
from tennis.db.schema import connect
from tennis.sim.draws import replay_from_round, engine_as_of, actual_progression
from tennis.sim.bracket import simulate, round_names, BYE
from tennis.models.predict import Predictor
from tennis.models.common import log_loss, brier

_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--events", type=int, default=0,
                 help="limit to the first N events (0 = all)")
_ap.add_argument("--sims", type=int, default=4000, help="playthroughs per run")
_args = _ap.parse_args()
N_SIMS = _args.sims

con = connect()
ev = pd.read_sql("""SELECT tourney_key, name, season FROM tournaments
  WHERE level IN ('grand_slam','masters') AND season BETWEEN 2023 AND 2025
  ORDER BY tourney_date""", con)
con.close()
if _args.events:
    ev = ev.head(_args.events)
print(f"{len(ev)} candidate events, {N_SIMS} sims per run")

rows = []
t0 = time.time()
engine_cache = {}

def engine_for(seq):
    if seq not in engine_cache:
        engine_cache[seq] = engine_as_of(seq)
    return engine_cache[seq]

for n, e in enumerate(ev.itertuples()):
    con = connect()
    base = replay_from_round(con, e.tourney_key, 0)
    con.close()
    if base is None:
        continue
    draw, g = base["draw"], base["matches"]
    pre_seq = int(g["seq"].min())
    date = draw.ctx.tourney_date
    ap = actual_progression(g, draw.slots).set_index("player_id")
    labels = round_names(draw.n_rounds)

    for fr in range(2, draw.n_rounds):          # from round 3 onward (0-indexed 2)
        con = connect()
        r = replay_from_round(con, e.tourney_key, fr)
        con.close()
        if r is None or r["state_seq"] is None or r["completed_rounds"] < fr:
            continue
        alive = {p for p in draw.slots if p != BYE
                 and ap.loc[p, labels[fr - 1]] == 1.0} if fr > 0 else set()
        if len(alive) < 4:
            continue
        future = labels[fr:]

        for arm, seq in (("frozen", pre_seq), ("updated", r["state_seq"])):
            P = Predictor(engine=engine_for(seq), as_of=date)
            sim = simulate(draw, P, n_sims=N_SIMS, resolved=r["resolved"]).set_index("player_id")
            ys, ps = [], []
            for p in alive:
                for lab in future:
                    ys.append(float(ap.loc[p, lab])); ps.append(float(sim.loc[p, lab]))
            ys, ps = np.array(ys), np.clip(np.array(ps), 1e-6, 1 - 1e-6)
            rows.append({"event": f"{e.season} {e.name}", "from_round": fr, "arm": arm,
                         "n": len(ys), "log_loss": log_loss(ys, ps), "brier": brier(ys, ps)})
    print(f"[{n+1}/{len(ev)}] {e.season} {e.name}  ({time.time()-t0:.0f}s)")

r = pd.DataFrame(rows)
from tennis.config import ARTIFACTS
r.to_csv(ARTIFACTS / "feature_recency_results.csv", index=False)
print("\n" + "="*66)
piv = r.pivot_table(index=["event","from_round"], columns="arm",
                    values=["log_loss","brier"]).reset_index()
ll = r.pivot_table(index=["event","from_round"], columns="arm", values="log_loss")
bs = r.pivot_table(index=["event","from_round"], columns="arm", values="brier")
ll["delta"] = ll["updated"] - ll["frozen"]
bs["delta"] = bs["updated"] - bs["frozen"]
print(f"comparisons: {len(ll)}")
print(f"\nLOG LOSS   frozen {ll['frozen'].mean():.5f}   updated {ll['updated'].mean():.5f}   "
      f"delta {ll['delta'].mean():+.5f}")
print(f"BRIER      frozen {bs['frozen'].mean():.5f}   updated {bs['updated'].mean():.5f}   "
      f"delta {bs['delta'].mean():+.5f}")
print(f"\nupdated better in {(ll['delta']<0).sum()}/{len(ll)} comparisons")
from scipy.stats import wilcoxon
try:
    st, pv = wilcoxon(ll["frozen"], ll["updated"])
    print(f"Wilcoxon signed-rank on log loss: p={pv:.4f}")
except Exception as ex:
    print("wilcoxon failed:", ex)
print("\nby from_round:")
print(r.pivot_table(index="from_round", columns="arm", values="log_loss").assign(
    delta=lambda d: d["updated"]-d["frozen"]).round(5).to_string())
