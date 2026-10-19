"""Structured logging for the service.

The deployment document claims per-stage timings in the logs. A claim like that
is only worth making if something emits it, so this module is that something:
one JSON object per line, on stdout, with the fields a log explorer turns into
a graph.

JSON rather than a formatted message because the point of a structured log is
that nobody parses prose. Text stays available (``STRATA_LOG_FORMAT=text``) for
a developer watching a terminal, where JSON is unreadable noise.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any

_CONFIGURED = False

# Fields that must never appear in a log line, whatever a caller passes. The
# optional model layer takes a key from the environment; a log that printed its
# configuration would put that key in a log explorer, which is a second place
# secrets live and the harder one to clean up.
_REDACT = ("api_key", "authorization", "token", "secret", "password", "access_key")


class JsonLogFormatter(logging.Formatter):
    """One JSON object per line, with the request-correlation fields kept flat."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "fields", None)
        if isinstance(extra, dict):
            for key, value in extra.items():
                name, redacted = _clean_pair(key, value)
                payload[name] = redacted
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, separators=(",", ":"))


def _is_secret(key: str) -> bool:
    lowered = key.lower()
    return any(needle in lowered for needle in _REDACT)


def _clean_pair(key: str, value: Any) -> tuple[str, Any]:
    """Rename *and* drop: a secret kept under a harmless key is still leaked.

    Relabelling alone was the first version of this, and a test that only looked
    at the key names let it through.
    """
    if _is_secret(key):
        return f"{key.lower()}_redacted", "[redacted]"
    return key.lower(), _safe(value)


def _safe(value: Any) -> Any:
    if isinstance(value, float):
        return None if value != value or value in (float("inf"), float("-inf")) else value
    if isinstance(value, (int, bool, str)) or value is None:
        return value
    if isinstance(value, dict):
        return dict(_clean_pair(k, v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    return str(value)


def log_fields(logger: logging.Logger, message: str, level: int = logging.INFO, **fields: Any) -> None:
    """Log one structured event: ``log_fields(log, "scan", total_ms=..., ...)``."""
    logger.log(level, message, extra={"fields": fields})


def configure_logging(level: str | int | None = None, fmt: str | None = None) -> None:
    """Attach one handler to the root logger. Idempotent, and safe to call twice."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    resolved_level = level or os.environ.get("STRATA_LOG_LEVEL", "INFO")
    resolved_fmt = fmt or os.environ.get("STRATA_LOG_FORMAT", "json")

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonLogFormatter()
        if resolved_fmt.lower() == "json"
        else logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )
    root = logging.getLogger()
    root.handlers = [h for h in root.handlers if h is not handler]
    root.addHandler(handler)
    root.setLevel(getattr(logging, str(resolved_level).upper(), logging.INFO))
    _CONFIGURED = True


def scan_log_fields(summary: dict[str, Any]) -> dict[str, Any]:
    """The subset of a scan summary worth indexing: numbers, not pixels.

    Event boxes and coverage grids belong in the stored record, not in every
    log line - a log that carries the whole payload costs more to read than it
    is worth, and the timings are the part that answers "why was that slow".
    """
    timings = summary.get("timings_ms") or {}
    registration = summary.get("registration") or {}
    coverage = summary.get("coverage") or {}
    return {
        "event_count": len(summary.get("events") or []),
        "registration_model": registration.get("model"),
        "registration_confidence": registration.get("confidence"),
        "observed_fraction": coverage.get("observed_fraction"),
        "feature_backend": summary.get("feature_backend"),
        "match_backend": summary.get("match_backend"),
        "total_ms": timings.get("total_ms"),
        "stage_ms": {
            k: v for k, v in timings.items() if k.endswith("_ms") and k != "total_ms"
        },
    }
