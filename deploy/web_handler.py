"""AWS Lambda entry point for the console's HTTP API.

The service is FastAPI; Lambda speaks an event JSON dialect. Mangum is the
adapter, and it is the only thing here, which is the point: ``service/app.py``
runs unchanged under uvicorn in a container, under ``TestClient`` in the test
suite and under Lambda behind this two-line handler. Nothing about the deployment
is allowed to reach into the application.

Handler: ``web_handler.handler``

Environment, all read by the app itself (see ``service/app.py``):

    STRATA_EXECUTOR   aws  -> S3 + DynamoDB store; anything else -> filesystem
    STRATA_S3_BUCKET  session objects
    STRATA_DDB_TABLE  the site ledger
    STRATA_MODEL_DIR  where the ONNX weights are staged (Lambda: /tmp/models)
    STRATA_CORS_ORIGINS  comma-separated browser origins allowed to call this

Payload sizes: API Gateway HTTP caps a request at 10 MB, and the console sends
base64 JPEG inside JSON, so a phone photo of a shelf fits and a raw 12 MP HEIC
does not. ``service/encoding.py`` enforces its own smaller limit first, so the
client gets a typed error rather than a gateway 413.
"""

from __future__ import annotations

import logging
import os

from mangum import Mangum

from handler import ensure_weights
from service.app import app

log = logging.getLogger("strata.web")

# Stage the ONNX weights before the first request, using the same helper the
# vision function uses. Without this the app still *configures* the neural
# backend, and then resolves down to SIFT at run time because the model files are
# not on disk - which the health document reports honestly, but which would make
# the console quietly slower than every number in the report. The zip cannot
# carry the weights (see deploy/deploy.py), so /tmp is filled from S3 here: one
# fetch per execution environment, not per request.
try:
    staged = ensure_weights(os.environ.get("STRATA_MODEL_DIR", ""))
    if staged:
        log.info("staged %d weight(s) for the web function: %s", len(staged), staged)
except Exception as exc:  # pragma: no cover - network path
    log.warning("weight staging failed (%s); the classical backend will be used", exc)

handler = Mangum(app, lifespan="off")
