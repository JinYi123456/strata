"""Start the STRATA service.

    python scripts/serve.py                    # http://127.0.0.1:8000
    python scripts/serve.py --port 9000
    python scripts/serve.py --reload           # development
    python scripts/serve.py --host 0.0.0.0     # expose on the network

When a ``PORT`` environment variable is present - which is how a hosting platform
tells a service where to listen - it is used for the port and the bind address
defaults to ``0.0.0.0``, because a service bound to loopback is unreachable from
outside its own container. Locally neither applies and it stays on 127.0.0.1.

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


def _default_host() -> str:
    if os.environ.get("STRATA_HOST"):
        return os.environ["STRATA_HOST"]
    # A platform that sets PORT expects the service to be reachable on all
    # interfaces; binding loopback there is a silent, total failure.
    return "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"


def _default_port() -> int:
    for key in ("STRATA_PORT", "PORT"):
        value = os.environ.get(key)
        if value:
            try:
                return int(value)
            except ValueError:
                print(f"ignoring non-numeric {key}={value!r}", file=sys.stderr)
    return 8000


def _default_static_dir() -> str | None:
    """Where the built console lives, if it has been built.

    Checked in order: an explicit override, the console bundled with the service,
    then the raw export directory. The bundled copy is what makes the service
    runnable from a checkout with Python alone, with no Node toolchain present.
    """
    explicit = os.environ.get("STRATA_STATIC_DIR")
    if explicit:
        return explicit
    for candidate in (ROOT / "service" / "static", ROOT / "web" / "out"):
        if (candidate / "index.html").is_file():
            return str(candidate)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the STRATA service.")
    parser.add_argument("--host", default=_default_host())
    parser.add_argument("--port", type=int, default=_default_port())
    parser.add_argument(
        "--reload", action="store_true", help="Restart on source changes (development only)"
    )
    parser.add_argument("--log-level", default=os.environ.get("STRATA_LOG_LEVEL", "info"))
    args = parser.parse_args()

    from service.observability import configure_logging

    configure_logging()

    static_dir = _default_static_dir()
    if static_dir:
        os.environ["STRATA_STATIC_DIR"] = static_dir
        print(f"serving the console from {static_dir}")
    else:
        print(
            "no built console found; serving the API only.\n"
            "  build it with: bash scripts/build_console.sh",
            file=sys.stderr,
        )

    try:
        import uvicorn
    except ImportError:
        print(
            "uvicorn is not installed.\n"
            "  pip install -r requirements.txt",
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
