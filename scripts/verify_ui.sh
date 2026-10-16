#!/usr/bin/env bash
# Verify the live app in one shell session.
#
# The service is started, exercised through a real browser, and stopped inside a
# single invocation. That is deliberate: this environment reaps processes when
# the shell that owns them exits, so a server started in one call is gone by the
# next one. Everything that needs the server therefore has to happen here.
#
#   bash scripts/verify_ui.sh
#
# Screenshots land in artifacts/screens/.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Resolved from PATH, with overrides for the case where a tool lives somewhere
# unusual. Hard-coding an absolute interpreter path would make this script work
# on exactly one machine, which is the opposite of what a verification script is
# for.
PY="${STRATA_PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY="python"
NODE="${STRATA_NODE:-node}"
export NODE_PATH="${STRATA_NODE_PATH:-$(npm root -g 2>/dev/null || echo)}"

PORT="${STRATA_VERIFY_PORT:-8077}"
LOG="artifacts/logs/server.log"

mkdir -p artifacts/logs artifacts/screens
rm -f "$LOG"

echo "==> building the web app if it is missing"
if [ ! -f web/out/index.html ]; then
  ( cd web && NEXT_TELEMETRY_DISABLED=1 npm run build ) || exit 1
fi

echo "==> starting the service on port $PORT"
PORT="$PORT" STRATA_DATA_DIR="artifacts/verify-data" STRATA_LOG_LEVEL="info" \
  "$PY" scripts/serve.py > "$LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
  fi
}
trap cleanup EXIT

for i in $(seq 1 40); do
  if curl -s --noproxy '*' -o /dev/null --max-time 2 "http://127.0.0.1:$PORT/api/health"; then
    echo "    service up after ${i}s"
    break
  fi
  sleep 1
done

if ! curl -s --noproxy '*' -o /dev/null --max-time 3 "http://127.0.0.1:$PORT/api/health"; then
  echo "!! service never came up; log follows"
  tail -30 "$LOG"
  exit 1
fi

echo "==> driving the app in a browser"
STRATA_VERIFY_URL="http://127.0.0.1:$PORT/" "$NODE" scripts/verify_ui.js
RC=$?

echo "==> service log (last 25 lines)"
tail -25 "$LOG"

exit $RC
