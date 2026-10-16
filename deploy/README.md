# Deployment

## Public site (recommended): GitHub Actions + Releases + Streamlit Community Cloud

```
GitHub Actions                         GitHub Release "data-latest"            Streamlit Community Cloud
  daily.yml   06:30 UTC  ─┐             state.tar.zst   raw CSVs, models  ─┐     dashboard/app.py
  weekly.yml  Mon 07:30  ─┼─ publish ─▶ serve.tar.zst   slim db + artifacts ─┼─▶  downloads serve.tar.zst on start
  odds.yml    every 3 h  ─┘             odds_snapshots.parquet               │     and when its copy is >6 h old
```

The laptop database is 6.5 GB, but 6.1 GB of it is `odds_quotes`, research data
from the tennisexplorer backfill that no page reads. The site needs ~150 MB of
core tables plus ~120 MB of artifacts, which the jobs rebuild from ~72 MB of raw
files and ship as one download. Nothing is stored in git.

* **daily** restores `state.tar.zst`, pulls TennisMyLife and the current
  tennis-data workbook, reloads, rebuilds features for both tours in full
  (no saved engine state, so no incremental drift), refreshes the reports,
  prefetches draw sheets for events starting within three days, and publishes.
* **weekly** does the same plus the walk-forward backtest and a production refit.
* **odds** captures prices every three hours into `odds_snapshots.parquet`. Only
  the price nearest a match's start is a closing line, so frequency is the point.
  It first probes tennisexplorer and fails loudly if GitHub's IPs are refused;
  if that happens, hide the Betting page on the public site.
* The app runs **cache-only for draws**: visitors never spend the RapidAPI quota.
  Set `RAPIDAPI_KEY` as an app secret only if you want live fetches (capped by
  `RAPIDAPI_DAILY_BUDGET`).

### One-time setup

1. Make the repository public (Actions minutes are unlimited for public repos).
2. Add the repo secret `RAPIDAPI_KEY` (optional; without it draws are not prefetched).
   Add the repo *variable* `APP_URL` (the `*.streamlit.app` address) once the app is
   deployed: `keepalive.yml` uses it to wake a sleeping app with a headless browser
   every 6 hours (a plain GET cannot), and re-enables the other scheduled workflows
   so GitHub's 60-days-of-inactivity rule cannot switch them off. The re-enable
   trick restarts that clock as far as the GitHub docs and similar keep-alive
   tools indicate; it is not something this repo has been able to test yet.
3. Seed the cloud with this machine's history, so it starts from your odds
   snapshots instead of from nothing:

   ```bash
   .venv/bin/python -m tennis.ops.bundle pack-state /tmp/state.tar.zst
   .venv/bin/python -m tennis.ops.bundle export-snapshots /tmp/odds_snapshots.parquet
   gh release create data-latest --prerelease --title "Pipeline data" --notes "Rolling assets"
   gh release upload data-latest /tmp/state.tar.zst /tmp/odds_snapshots.parquet
   ```
4. Run **weekly** once from the Actions tab (`workflow_dispatch`): it publishes
   `serve.tar.zst`. Check the run's wall time and peak memory in its log.
5. On share.streamlit.io, create an app from this repo with main file
   `dashboard/app.py` and the secret
   `DATA_BUNDLE_URL = "https://github.com/<owner>/<repo>/releases/download/data-latest/serve.tar.zst"`.
6. Once the scheduled runs are green for a few days, retire the laptop agents:
   `launchctl unload ~/Library/LaunchAgents/com.tennis.{odds,daily,weekly}.plist`.
   The odds *backfill* (`odds_te_hist`) stays local research and is unaffected.

### Things that can go wrong

* **Memory on Streamlit Cloud.** Serving peaks around 560 MB, but a *replay*
  rebuilds feature state from every match. If it does not fit, move the same app
  to a Hugging Face Space (16 GB); nothing else changes.
* **Cron jitter.** Scheduled runs often start 5-30 minutes late. Harmless for
  daily; the 3-hourly capture drifts slightly.
* **60-day inactivity.** Scheduled workflows are disabled after 60 days without
  repository activity. The weekly job makes a small empty commit in the first
  week of each month to prevent it (this needs the default branch to accept
  pushes from `github-actions`).

## Local machine (research and fallback)

The sections below describe running everything on this laptop. It remains the
research machine: the odds backfill and the 7.5 GB scrape cache live here.

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
