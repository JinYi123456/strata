"""The STRATA HTTP API.

The service is the only place the vision core is reachable from outside, and it
is deliberately thin: it validates input, delegates to :mod:`sessions`, and
serialises the result. All the interesting decisions live in the core and the
agent, where they can be tested without a server.

Run locally:

    uvicorn service.app:app --reload --port 8000

Configuration is environment-only, so the same image runs unchanged on a laptop
and on managed infrastructure:

===========================  ==================================================
``STRATA_EXECUTOR``          ``local`` (default) or ``aws``
``STRATA_DATA_DIR``          Where the local store writes. Default ``./data``
``STRATA_S3_BUCKET``         Bucket for frames, when ``executor=aws``
``STRATA_DDB_TABLE``         Table for records, when ``executor=aws``
``STRATA_AWS_REGION``        Region for the above
``STRATA_CORS_ORIGINS``      Comma-separated allowed origins. Default ``*``
``STRATA_STATIC_DIR``        Built web app to serve, if any
===========================  ==================================================
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from strata_core.config import FeatureConfig, PipelineConfig
from strata_core.features import neural_available, resolve_backend

from . import demo, encoding
from .models import (
    CommitRequest,
    CreateSessionRequest,
    ErrorOut,
    Health,
    ImageRef,
    LedgerEntry,
    LedgerOut,
    RescanRequest,
    RescanResponse,
    SessionState,
    SiteList,
)
from .payloads import build_rescan_response
from .sessions import SessionError, SessionManager
from .store import StoreError, build_store

log = logging.getLogger("strata.service")

API_PREFIX = "/api"


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def create_app(
    *,
    executor: str | None = None,
    data_dir: str | Path | None = None,
    manager: SessionManager | None = None,
    static_dir: str | Path | None = None,
) -> FastAPI:
    """Build the application. Injectable so tests need no environment."""
    executor = (executor or os.environ.get("STRATA_EXECUTOR", "local")).lower()
    data_dir = Path(data_dir or os.environ.get("STRATA_DATA_DIR", "data"))

    if manager is None:
        store = build_store(
            executor,
            root=data_dir,
            bucket=os.environ.get("STRATA_S3_BUCKET", ""),
            table=os.environ.get("STRATA_DDB_TABLE", ""),
            region=os.environ.get("STRATA_AWS_REGION"),
            prefix=os.environ.get("STRATA_S3_PREFIX", "strata"),
        )
        manager = SessionManager(store)

    app = FastAPI(
        title="STRATA",
        version="1.0",
        description=(
            "Version control for the physical world. Record a site once, re-scan "
            "it later, and get a typed, confidence-scored account of what changed."
        ),
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )

    origins = [
        o.strip()
        for o in os.environ.get("STRATA_CORS_ORIGINS", "*").split(",")
        if o.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins or ["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    app.state.manager = manager
    app.state.executor = executor

    # --- Error mapping -------------------------------------------------
    #
    # Failures are returned in the shape the schema declares, so the client has
    # one code path. A bare 500 with an HTML body is the single most common way
    # a UI ends up showing "Unexpected token < in JSON".

    @app.exception_handler(SessionError)
    async def _session_error(_: Request, exc: SessionError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content=ErrorOut(
                error="session_error", detail=str(exc), hint="Check the session id and its state."
            ).model_dump(),
        )

    @app.exception_handler(encoding.ImageError)
    async def _image_error(_: Request, exc: encoding.ImageError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content=ErrorOut(
                error="invalid_image",
                detail=str(exc),
                hint="Send PNG or JPEG frames as base64 data URLs.",
            ).model_dump(),
        )

    @app.exception_handler(StoreError)
    async def _store_error(_: Request, exc: StoreError) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content=ErrorOut(error="not_found", detail=str(exc)).model_dump(),
        )

    # --- Health --------------------------------------------------------

    @app.get(f"{API_PREFIX}/health", response_model=Health, tags=["meta"])
    def health() -> Health:
        import cv2

        fcfg = FeatureConfig()
        backend = resolve_backend(fcfg)
        return Health(
            vision_backend=backend,
            executor=app.state.executor,
            storage=manager.store.name,
            opencv_version=str(cv2.__version__),
            model_weights_present=neural_available(fcfg),
            limits={
                "max_upload_bytes": encoding.MAX_UPLOAD_BYTES,
                "max_frames_per_upload": encoding.MAX_FRAMES_PER_UPLOAD,
                "max_dimension": encoding.MAX_DIMENSION,
                "max_concurrent_scans": 2,
            },
        )

    # --- Sites ---------------------------------------------------------

    @app.get(f"{API_PREFIX}/sites", response_model=SiteList, tags=["sites"])
    def list_sites() -> SiteList:
        return SiteList(sites=manager.list_sites())

    @app.get(
        f"{API_PREFIX}/sites/{{site_id}}/ledger", response_model=LedgerOut, tags=["sites"]
    )
    def site_ledger(site_id: str) -> LedgerOut:
        return LedgerOut(site_id=site_id, entries=manager.ledger(site_id))

    # --- Sessions ------------------------------------------------------

    @app.post(f"{API_PREFIX}/sessions", response_model=SessionState, tags=["sessions"])
    def create_session(req: CreateSessionRequest) -> SessionState:
        frames = encoding.decode_frames(req.baseline_frames)
        session = manager.create_session(
            site_id=req.site_id, name=req.name, baseline_frames=frames or None
        )
        return _session_state(manager, session)

    @app.get(
        f"{API_PREFIX}/sessions/{{session_id}}", response_model=SessionState, tags=["sessions"]
    )
    def get_session(session_id: str) -> SessionState:
        return _session_state(manager, manager.get_session(session_id))

    @app.post(
        f"{API_PREFIX}/sessions/{{session_id}}/rescan",
        response_model=RescanResponse,
        tags=["sessions"],
    )
    def rescan(session_id: str, req: RescanRequest) -> RescanResponse:
        frames = encoding.decode_frames(req.frames)
        if not frames and not req.use_site_rescan:
            raise SessionError("no frames were supplied and the stored rescan was declined")

        outcome, scan_index = manager.rescan(
            session_id, frames=frames or None, min_confidence=req.min_confidence
        )

        session = manager.get_session(session_id)
        ground_truth = None
        site = demo.get_demo_site(session.site_id) if session.is_demo else None
        if site is not None:
            ground_truth = demo.ground_truth_payload(site)

        def image_url(panel: str) -> str:
            return f"{API_PREFIX}/sessions/{session_id}/scans/{scan_index}/panels/{panel}"

        return build_rescan_response(
            outcome,
            session_id=session_id,
            scan_index=scan_index,
            image_url=image_url,
            min_confidence=req.min_confidence,
            ground_truth=ground_truth,
        )

    @app.get(
        f"{API_PREFIX}/sessions/{{session_id}}/scans/{{scan_index}}", tags=["sessions"]
    )
    def scan_record(session_id: str, scan_index: int) -> dict[str, Any]:
        return manager.scan_record(session_id, scan_index)

    @app.get(
        f"{API_PREFIX}/sessions/{{session_id}}/scans/{{scan_index}}/panels/{{panel}}",
        tags=["sessions"],
    )
    def panel(session_id: str, scan_index: int, panel: str) -> Response:
        raw = manager.read_panel(session_id, scan_index, panel)
        media = "image/png" if raw[:4] == b"\x89PNG" else "image/jpeg"
        return Response(
            content=raw,
            media_type=media,
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )

    @app.get(f"{API_PREFIX}/sessions/{{session_id}}/baseline", tags=["sessions"])
    def baseline_image(session_id: str, frame: int = 0) -> Response:
        raw = manager.read_baseline(session_id, frame)
        return Response(
            content=raw,
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )

    # --- Commit gate ---------------------------------------------------

    @app.post(
        f"{API_PREFIX}/sessions/{{session_id}}/commit", response_model=LedgerEntry, tags=["ledger"]
    )
    def commit(session_id: str, req: CommitRequest) -> LedgerEntry:
        entry = manager.commit(session_id, approved=req.approved, note=req.note)
        return LedgerEntry(**entry)

    # --- Optional static front end -------------------------------------

    static_root = Path(
        static_dir or os.environ.get("STRATA_STATIC_DIR", "web/out")
    )
    if static_root.is_dir():
        _mount_static(app, static_root)
    else:
        log.info("no built web app at %s; serving the API only", static_root)

    return app


def _session_state(manager: SessionManager, session) -> SessionState:
    """Compose the client-visible state, including the coverage and guidance views."""
    coverage = session.baseline_coverage
    guidance_url = None
    baseline_url = f"{API_PREFIX}/sessions/{session.session_id}/baseline?frame=0"
    bw, bh = session.baseline_size

    # Coverage and guidance are produced by the first scan, so they are read back
    # from the most recent one rather than recomputed here.
    if session.scans:
        try:
            record = manager.scan_record(session.session_id, session.scans - 1)
            coverage = record.get("coverage") or coverage
            guidance_url = (
                f"{API_PREFIX}/sessions/{session.session_id}/scans/"
                f"{session.scans - 1}/panels/guidance"
            )
        except (SessionError, StoreError):
            pass

    return SessionState(
        session_id=session.session_id,
        site_id=session.site_id,
        name=session.name,
        created_at=session.created_at,
        canonical_size=session.canonical_size,
        baseline_frames=session.baseline_frames,
        baseline=ImageRef(url=baseline_url, width=int(bw), height=int(bh)),
        coverage=coverage,
        guidance=ImageRef(url=guidance_url, width=int(bw), height=int(bh))
        if guidance_url
        else None,
        scans=session.scans,
        committed=session.committed,
    )


def _mount_static(app: FastAPI, root: Path) -> None:
    """Serve a built front end from the same origin as the API.

    Same-origin matters: it removes CORS from the deployment entirely and lets
    the app be published as one URL rather than two that must be kept in step.
    """
    from fastapi.staticfiles import StaticFiles

    class _SpaStaticFiles(StaticFiles):
        """Serve index.html for unknown paths so client-side routing works."""

        async def get_response(self, path: str, scope):  # type: ignore[override]
            try:
                return await super().get_response(path, scope)
            except Exception:
                return await super().get_response("index.html", scope)

    app.mount("/", _SpaStaticFiles(directory=str(root), html=True), name="web")
    log.info("serving the built web app from %s", root)


app = create_app()
