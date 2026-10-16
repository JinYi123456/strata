#!/usr/bin/env bash
# Build the operator console and bundle it with the service.
#
#   bash scripts/build_console.sh
#
# The built console is copied to service/static/, which is committed. That is a
# deliberate product decision rather than an accident of tooling: it means the
# service runs from a checkout with Python alone, with no Node toolchain present,
# which is what makes `python scripts/serve.py` work immediately after
# `pip install -r requirements.txt`.
#
# The cost is that service/static/ is generated and can drift from web/. Run this
# script after any change under web/ - and if you forget, the symptom is a stale
# interface rather than a broken one, which is why it is worth a script rather
# than a note in a document.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

NODE_DIR="C:/Users/User/.workbuddy-ai/binaries/node/versions/22.22.2-6"
if [ -d "$NODE_DIR" ]; then
  export PATH="$NODE_DIR:$PATH"
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "npm not found; install Node 20+ to build the console" >&2
  exit 1
fi

echo "==> installing web dependencies"
(cd web && npm install --no-audit --no-fund)

echo "==> building"
(cd web && NEXT_TELEMETRY_DISABLED=1 npm run build)

if [ ! -f web/out/index.html ]; then
  echo "the build produced no web/out/index.html" >&2
  exit 1
fi

echo "==> bundling into service/static"
rm -rf service/static
mkdir -p service/static
cp -r web/out/. service/static/

count=$(find service/static -type f | wc -l)
size=$(du -sh service/static | cut -f1)
echo "    $count files, $size"
echo "done. The service will serve it automatically."
