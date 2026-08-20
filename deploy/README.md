# Deployment

CLAUDE.md said to deploy alongside an existing NBA dashboard on a shared cloud
VM. **There is no such VM** — that premise turned out not to hold, so there is
no host to co-locate with and the sizing question is open. Nothing here has been
run against any server.

## Do you actually need a VM?

Measured on this codebase, not estimated:

| workload | peak RSS | wall time | when |
|---|---|---|---|
| Dashboard serving (predictor + backtest + a 20k simulation) | **563 MB** | — | continuous |
| Daily incremental ingest | **1.35 GB** | 30 s | nightly |
| Full feature rebuild | **3.1 GB** | 55 s | initial setup, or if state is lost |
| Model refit + walk-forward backtest | ~3 GB | **20 min** (350% CPU) | weekly, optional |

Disk: ~350 MB of data and artifacts, ~700 MB including the venv.

The binding constraint is the full rebuild and the weekly retrain, not serving.
Serving alone is small enough to fit almost anywhere.

### Option A — no VM, run locally (set up here)

Everything already runs on a laptop. `deploy/launchd/` holds two ready agents:

| agent | when | what | runtime |
|---|---|---|---|
| `com.tennis.odds` | every 3 h | capture live prices from tennisexplorer into `odds_snapshots` | ~5 s |
| `com.tennis.daily` | 07:15 daily | pull TennisMyLife ongoing files + current-season odds, reload matches, extend features, resolve captured prices | ~30 s |
| `com.tennis.weekly` | Mon 08:00 | above, plus re-check every historical file, refit models, rerun the backtest | ~25 min |

Install them:

```bash
cp deploy/launchd/*.plist ~/Library/LaunchAgents/ && launchctl load ~/Library/LaunchAgents/com.tennis.odds.plist && launchctl load ~/Library/LaunchAgents/com.tennis.daily.plist && launchctl load ~/Library/LaunchAgents/com.tennis.weekly.plist
```

Check they are registered, and force a run without waiting for the schedule:

```bash
launchctl list | grep com.tennis
```

```bash
launchctl start com.tennis.daily
```

Output goes to `data/launchd-daily.log` and `data/launchd-weekly.log`; the
dashboard's Status page shows the last run's summary.

`scripts/daily.sh` deliberately does **not** redirect its own output — launchd's
`StandardOutPath` routes each agent to its own file, and a redirect inside the
script would collapse both into one and leave the per-agent logs empty. Running
the script by hand therefore prints to your terminal. cron has no equivalent
mechanism, so `deploy/crontab.example` redirects per entry instead.

**The one caveat: a sleeping laptop skips its window.** `launchd` does not queue
missed `StartCalendarInterval` runs while the machine is asleep — it fires once
on wake if the time has passed, which for a machine that sleeps overnight
usually works out, but is not guaranteed. The job is idempotent and reloads the
full match table each run, so a missed day self-heals on the next one; nothing
accumulates a gap. If you need a hard guarantee, that is the argument for
Option B.

To remove them: `launchctl unload ~/Library/LaunchAgents/com.tennis.{odds,daily,weekly}.plist`.

## Reaching the dashboard from elsewhere

```bash
./scripts/serve.sh
```

Binds to all interfaces and prints the tailnet address. Install Tailscale, sign
in on both machines, and the dashboard is reachable from anywhere without being
exposed to the internet.

This is deliberately the *only* thing that leaves the machine. The collector is
stateful -- a database heading for ~10 GB once the odds backfill lands, a
gzipped scrape cache, jobs that run for days -- so it stays put and only the
view travels. Moving it to a VPS later is a lift-and-shift of the systemd unit
in this directory, not a redesign.

## What does *not* run in CI, and why

`.github/workflows/tests.yml` runs the logic tests on every push and nothing
else. The scheduled jobs cannot run there: Actions caps a job at 6 hours and
gives ephemeral disks, while the odds backfill runs for days against persistent
state, and the free tier's 2,000 minutes a month is dwarfed by the ~34,000 that
backfill needs. Committing the database to make it portable is worse still --
a 256 MB binary rewritten daily, with unresolvable merge conflicts.

Data-dependent tests mark themselves `needs_db` and skip on a clean checkout;
47 of 62 run without any data, and the workflow fails if that number drops.

**Why the odds capture is its own agent.** The daily job is heavy and only needs
to run once, after play. The capture is two HTTP requests, and its value depends
entirely on running often: only the price nearest a match's start can honestly
be called a closing line, and CLV is measured against one. On 4,365 resolved
fixtures the last capture before play beat the first by **0.00410 log loss** —
larger than any modelling change in this project — over a median 10.7-hour gap.
Most lines barely move (median 0.0027), but 30% move more than a point and 5%
more than five. A stale capture is not merely less useful; labelled "closing" it
flatters whatever it is compared against.

The dashboard stays at `localhost:8502` and is not reachable from elsewhere;
that, and the sleep caveat above, are the only things Option B buys.

### Option B — one small VM (recommended if you want it always-on)

**4 GB RAM, 2 vCPU, 20 GB disk.** Roughly €4–7/month (Hetzner CX22) or
$24/month (DigitalOcean). 4 GB rather than 2 GB so a full rebuild and the weekly
retrain fit without swapping; 2 GB would serve fine and handle the nightly
incremental, but would thrash on a rebuild.

If you never retrain on the box — refit locally and copy `artifacts/*.pkl` up —
2 GB is genuinely enough and halves the cost.

### Option C — managed Streamlit hosting

Streamlit Community Cloud is free and the 563 MB serving footprint fits its
1 GB limit. But it needs a public GitHub repo, the 148 MB of artifacts exceed
comfortable git limits without LFS, and the nightly job still needs a home
somewhere else. More moving parts than either option above for this workload.

## Setup, once a host exists

```bash
git clone <repo> /opt/tennis-predictor && cd /opt/tennis-predictor
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m tennis.ingest.download      # ~60 MB, a few minutes
.venv/bin/python -m tennis.ingest.load
.venv/bin/python -m tennis.ingest.odds
.venv/bin/python -m tennis.features.pipeline
.venv/bin/python -m tennis.models.train         # walk-forward backtest + prod models
.venv/bin/python -m tennis.models.evaluate
```

## Services

`tennis-dashboard.service` runs Streamlit on port 8502. There is nothing else on
this host, so 8501 is free — change `Environment=PORT=` if you prefer it.

```bash
sudo cp deploy/tennis-dashboard.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now tennis-dashboard
```

Put it behind a reverse proxy with TLS and auth before exposing it publicly —
Streamlit has no authentication of its own.

## Scheduled ingestion

```bash
crontab -l | { cat; cat deploy/crontab.example; } | crontab -
```

Daily at 06:15 UTC pulls the ongoing-tournament files and the current-season
odds workbook, reloads matches, and extends features incrementally. Weekly on
Monday it also refits the models and reruns the backtest — a single day of
results cannot meaningfully move a model fit on 200k matches, so daily
retraining would just burn CPU.

On a 2 GB host, drop the weekly `--retrain` line and refit elsewhere.
