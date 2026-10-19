"""
Lambda entry point for the STRATA vision workload.

One invocation runs one scan: read the two visits from S3, run the pipeline, put
the change set in the ledger, return the summary.

The handler is deliberately thin. All of the vision lives in ``strata_core``,
which knows nothing about AWS, and all of the AWS lives here, which knows
nothing about vision. That split is why the same pipeline runs unchanged in the
console, in the HTTP service and in this container.

Environment
-----------
STRATA_S3_BUCKET     bucket holding session objects
STRATA_DDB_TABLE     ledger table
STRATA_AWS_REGION    region (the runtime sets this, but allow an override)
STRATA_BACKEND       feature backend: auto | aliked | sift | orb
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import replace
from typing import Any

import boto3
import cv2
import numpy as np

from strata_core.config import FeatureConfig, PipelineConfig
from strata_core.pipeline import run_scan

log = logging.getLogger("strata.lambda")
log.setLevel(os.environ.get("STRATA_LOG_LEVEL", "INFO"))

_S3 = None
_DDB = None


def s3():
    global _S3
    if _S3 is None:
        _S3 = boto3.client("s3")
    return _S3


def ddb():
    global _DDB
    if _DDB is None:
        _DDB = boto3.resource("dynamodb")
    return _DDB


def load_image(bucket: str, key: str) -> np.ndarray:
    """Read one image out of S3 and decode it.

    ``imdecode`` rather than ``imread``: the function has no filesystem, which
    keeps the container read-only-safe and avoids a temp file per frame.
    """
    body = s3().get_object(Bucket=bucket, Key=key)["Body"].read()
    buf = np.frombuffer(body, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not decode s3://{bucket}/{key}")
    return image


def build_config() -> PipelineConfig:
    """Build the pipeline config from the environment.

    ``PipelineConfig`` is frozen, so this composes a new one rather than
    mutating a default - which also means a handler invocation cannot leak
    configuration into the next one.
    """
    backend = os.environ.get("STRATA_BACKEND", "auto")
    return replace(
        PipelineConfig(),
        features=replace(
            FeatureConfig(),
            backend=backend,  # type: ignore[arg-type]
            model_dir=os.environ.get("STRATA_MODEL_DIR", "/var/task/models"),
        ),
    )


def _num(value: Any, ndigits: int = 3) -> float | None:
    """A JSON-safe float.

    ``inf`` and ``nan`` are not valid JSON, and a strict serialiser rejects the
    whole payload because of one of them. They are also not numbers a caller can
    use: an identity registration has no reprojection error at all, so the
    honest encoding is null rather than a large sentinel. This is the same defect
    the service hit — the one case that most needs a clean answer was the one
    that broke — so it is handled at the boundary here too.
    """
    number = float(value)
    return round(number, ndigits) if math.isfinite(number) else None


def summarise(result: Any) -> dict[str, Any]:
    """Shape the scan into a JSON-safe payload.

    Floats are rounded to a fixed precision so two runs that agree to within
    noise produce identical JSON - otherwise every replay looks like a change.
    """
    reg = result.registration
    return {
        "backend": result.feature_backend,
        "registration": {
            "model": reg.model,
            "inliers": int(reg.inlier_count),
            "matches": int(reg.match_count),
            "inlier_ratio": _num(reg.inlier_ratio, 4),
            "reprojection_px": _num(reg.reproj_error_px),
            "confidence": _num(reg.confidence, 4),
            "notes": list(reg.notes),
        },
        "events": [
            {
                "kind": e.kind,
                "confidence": _num(e.confidence, 4),
                "box": [int(v) for v in e.bbox],
                "area_px": int(e.bbox[2]) * int(e.bbox[3]),
                "displacement_px": _num(e.displacement_px, 2),
                "appearance_delta": _num(e.appearance_delta),
                "summary": e.summary,
            }
            for e in result.events
        ],
        "coverage": result.to_summary()["coverage"],
        "timings_ms": result.timings.to_dict(),
    }


def ledger_row(
    prefix: str, site_id: str, session_id: str, payload: dict[str, Any], *, now_ms: int
) -> dict[str, str]:
    """One DynamoDB item, in the shape the deployed table accepts.

    Same ``{pk, sk, doc}`` contract as ``service/store.py``, because both writers
    target the one table that ``deploy/deploy.sh`` creates. The distinct prefix
    keeps Lambda-written rows out of the service's own ledger listing rather than
    mixing two record types into one partition.
    """
    return {
        "pk": f"{prefix}/ledger/{site_id}",
        "sk": f"{session_id}-{now_ms}.json",
        "doc": json.dumps(payload),
    }


def handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    """Run one scan.

    Event shape:
        {"site_id": str, "session_id": str,
         "baseline": [s3 key, ...], "rescan": [s3 key, ...],
         "record": bool}

    Returns the change set plus per-stage timings. A failure returns a typed
    error rather than raising, because a 500 with a stack trace is not something
    an operator can act on.
    """
    started = time.perf_counter()
    bucket = os.environ["STRATA_S3_BUCKET"]

    try:
        site_id = str(event["site_id"])
        session_id = str(event["session_id"])
        baseline_keys = list(event["baseline"])
        rescan_keys = list(event["rescan"])
    except (KeyError, TypeError) as exc:
        return {"ok": False, "error": "bad_request", "detail": str(exc)}

    try:
        baseline = [load_image(bucket, k) for k in baseline_keys]
        rescan = [load_image(bucket, k) for k in rescan_keys]
    except Exception as exc:
        log.exception("could not read session objects")
        return {"ok": False, "error": "unreadable_session", "detail": str(exc)}

    try:
        result = run_scan(baseline, rescan, config=build_config())
    except ValueError as exc:
        # An empty capture set, or a pair that cannot be registered.
        return {"ok": False, "error": "unregistrable", "detail": str(exc)}

    payload = summarise(result)
    payload["ok"] = True
    payload["site_id"] = site_id
    payload["session_id"] = session_id
    payload["wall_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
    # Always present, so a caller can branch on it without a key check.
    payload["recorded"] = False

    if event.get("record"):
        table = os.environ.get("STRATA_DDB_TABLE")
        if not table:
            payload["recorded"] = False
            payload["record_error"] = "no ledger table configured"
        else:
            try:
                prefix = os.environ.get("STRATA_LEDGER_PREFIX", "lambda")
                ddb().Table(table).put_item(
                    Item=ledger_row(
                        prefix, site_id, session_id, payload,
                        now_ms=int(time.time() * 1000),
                    )
                )
                payload["recorded"] = True
            except Exception as exc:  # pragma: no cover - network path
                log.exception("ledger write failed")
                payload["recorded"] = False
                payload["record_error"] = str(exc)

    return payload
