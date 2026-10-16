"""The STRATA service: HTTP access to the vision core and the perception agent.

The package is organised so that each layer has one job:

``store``     where bytes and records live, local or on managed infrastructure
``demo``      deterministic sites, so the app is usable with nothing uploaded
``encoding``  the untrusted-input boundary for images
``runners``   running the pipeline and the agent, and nothing else
``payloads``  the wire format, and the coordinate conventions it guarantees
``sessions``  session lifecycle and the ledger
``app``       HTTP routing and error mapping

``service.app:app`` is the ASGI entry point.
"""

from __future__ import annotations

__all__ = ["create_app"]

__version__ = "1.0.0"


def create_app(*args, **kwargs):
    """Lazy re-export so importing the package does not build the app."""
    from .app import create_app as _create_app

    return _create_app(*args, **kwargs)
