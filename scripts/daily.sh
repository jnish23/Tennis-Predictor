#!/usr/bin/env bash
# Entrypoint for the ingestion job (launchd, cron, or by hand).
#
# Deliberately does not redirect its own output. The launchd agents in
# deploy/launchd/ set StandardOutPath per agent, so the daily and weekly runs
# land in separate logs; a redirect here would swallow both into one file and
# leave those empty. Run by hand and you get the output on your terminal, which
# is what you want. cron has no equivalent, so deploy/crontab.example does its
# own redirect.
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python -m tennis.ingest.daily "$@"
