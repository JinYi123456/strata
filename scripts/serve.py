"""Start the STRATA service.

    python scripts/serve.py                    # http://127.0.0.1:8000
    python scripts/serve.py --port 9000
    python scripts/serve.py --reload           # development
    python scripts/serve.py --host 0.0.0.0     # expose on the network

Environment variables (see ``service/app.py`` for the full list) control where
state is stored and whether the service runs against local disk or managed
infrastructure. Nothing here reads a secret: the model-backed decision layer is
optional and picks up its own key from the environment if one is present, and
falls back to the deterministic policy when it is not.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The vision core lives beside the service rather than inside it, so both roots
# are put on the path here instead of relying on the caller's PYTHONPATH.
for candidate in (ROOT / "vision", ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the STRATA service.")
    parser.add_argument("--host", default=os.environ.get("STRATA_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("STRATA_PORT", "8000")))
    parser.add_argument(
        "--reload", action="store_true", help="Restart on source changes (development only)"
    )
    parser.add_argument(
        "--log-level", default=os.environ.get("STRATA_LOG_LEVEL", "info")
    )
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print(
            "uvicorn is not installed.\n"
            "  pip install -r vision/requirements.txt -r service/requirements.txt",
            file=sys.stderr,
        )
        return 1

    print(f"STRATA service on http://{args.host}:{args.port}  (docs at /api/docs)")
    uvicorn.run(
        "service.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
