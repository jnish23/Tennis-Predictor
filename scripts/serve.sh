#!/usr/bin/env bash
# Serve the dashboard on the tailnet, so it is reachable from anywhere without
# exposing it to the internet.
#
# Why this rather than a host: the collector is stateful -- a ~10 GB database
# once the odds backfill lands, a gzipped scrape cache, long-running jobs -- so
# the machine holding it stays put and only the *view* needs to travel.
# Tailscale gives that for free, privately, with no change to the pipeline.
#
# Setup, once:
#   brew install --cask tailscale     # then sign in and enable the extension
#   tailscale ip -4                   # the address this machine answers on
#
# Then from any signed-in device: http://<that-ip>:8502
#
# `tailscale serve` (commented below) would additionally give it an HTTPS name
# on the tailnet. It is off by default because it requires HTTPS certificates
# to be enabled for the tailnet, which is a per-account choice.
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${PORT:-8502}"

if command -v tailscale >/dev/null 2>&1; then
  TS_IP="$(tailscale ip -4 2>/dev/null | head -1 || true)"
  if [ -n "$TS_IP" ]; then
    echo "tailnet:  http://${TS_IP}:${PORT}"
  else
    echo "tailscale installed but not connected — run 'tailscale up' first" >&2
  fi
else
  echo "tailscale not found; serving on the local network only" >&2
fi
echo "local:    http://localhost:${PORT}"

# 0.0.0.0 so the tailnet interface is included. The tailnet is private by
# default -- only devices signed into your account can reach it -- but this
# does also expose the port to the local network, so do not run it on an
# untrusted one.
exec .venv/bin/streamlit run dashboard/app.py \
  --server.address 0.0.0.0 \
  --server.port "${PORT}" \
  --server.headless true \
  --browser.gatherUsageStats false
