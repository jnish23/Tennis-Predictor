#!/usr/bin/env bash
# Live odds capture only -- deliberately separate from daily.sh.
#
# The daily job is heavy (download, reload, rebuild features, ~30s+) and only
# needs to run once, after play. This is two HTTP requests and a handful of
# inserts, and its value depends entirely on running *often*: the price nearest
# a match's start is the only one that can honestly be called a closing line,
# and measured on 4,365 resolved fixtures the last capture before play beats
# the first by 0.00410 log loss. A price captured 12 hours early is an opening
# line, and benchmarking against one flatters whatever it is compared to.
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python -m tennis.ingest.odds_live
