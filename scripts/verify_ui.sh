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

PY="C:/Users/User/.workbuddy-ai/binaries/python/envs/opencv5/Scripts/python.exe"
NODE="C:/Users/User/.workbuddy-ai/binaries/node/versions/22.22.2-6/node.exe"
PW_MODULES="C:/Users/User/.workbuddy-ai/binaries/node/workspace/node_modules"
PORT="${STRATA_VERIFY_PORT:-8077}"
LOG="artifacts/logs/server.log"

mkdir -p artifacts/logs artifacts/screens
rm -f "$LOG"

echo "==> building the web app if it is missing"
if [ ! -f web/out/index.html ]; then
  ( cd web && PATH="C:/Users/User/.workbuddy-ai/binaries/node/versions/22.22.2-6:$PATH" \
      NEXT_TELEMETRY_DISABLED=1 npm run build ) || exit 1
fi

echo "==> starting the service on port $PORT"
STRATA_STATIC_DIR="web/out" STRATA_DATA_DIR="artifacts/verify-data" \
  STRATA_LOG_LEVEL="info" "$PY" scripts/serve.py --port "$PORT" > "$LOG" 2>&1 &
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
STRATA_VERIFY_URL="http://127.0.0.1:$PORT/" \
NODE_PATH="$PW_MODULES" "$NODE" scripts/verify_ui.js
RC=$?

echo "==> service log (last 25 lines)"
tail -25 "$LOG"

exit $RC
