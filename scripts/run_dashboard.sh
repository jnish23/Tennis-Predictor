#!/usr/bin/env bash
# Launch the Streamlit dashboard.
#
# PORT=... to override. If the port is taken, say what is holding it rather
# than letting Streamlit fail with a bare "Port N is not available" — the usual
# cause is an earlier copy of this same dashboard still running.
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${PORT:-8502}"
ADDRESS="${ADDRESS:-0.0.0.0}"

holder=$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null | head -1 || true)
if [ -n "$holder" ]; then
  cmd=$(ps -o command= -p "$holder" 2>/dev/null || echo "unknown")
  echo "Port $PORT is already in use by PID $holder:" >&2
  echo "  $cmd" >&2
  echo >&2
  if echo "$cmd" | grep -q "streamlit run dashboard/app.py"; then
    if [ "${KILL_EXISTING:-0}" = "1" ]; then
      echo "KILL_EXISTING=1 set — stopping PID $holder and relaunching." >&2
      kill "$holder" 2>/dev/null || true
      # let the socket clear before rebinding
      for _ in $(seq 1 10); do
        lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t >/dev/null 2>&1 || break
        sleep 0.5
      done
    else
      echo "That is another copy of this dashboard. Either:" >&2
      echo "  kill $holder            # stop it" >&2
      echo "  KILL_EXISTING=1 $0      # stop it and relaunch" >&2
      echo "  PORT=8503 $0            # run alongside it" >&2
      exit 1
    fi
  else
    echo "Not our dashboard. Use PORT=8503 $0 to pick a different port." >&2
    exit 1
  fi
fi

exec .venv/bin/streamlit run dashboard/app.py \
  --server.port "$PORT" \
  --server.address "$ADDRESS" \
  --server.headless true
