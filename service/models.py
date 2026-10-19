"""Request and response schemas for the STRATA HTTP API.

Every payload the service accepts or returns is declared here, so the contract
the web client codes against is a single readable file rather than something
that has to be inferred from handler bodies. The schemas are deliberately
explicit about units and coordinate spaces: a box in canonical pixels and a box
in source pixels look identical on the wire, and mixing them silently is the
easiest way to build a UI that displays confidently wrong numbers.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# --------------------------------------------------------------------------
# Shared
# --------------------------------------------------------------------------


class Health(BaseModel):
    """Liveness plus the provenance a client needs to interpret results."""

    status: Literal["ok"] = "ok"
    service: str = "strata"
    api_version: str = "1.0"
    vision_backend: str = Field(description="Feature backend the pipeline resolved")
    executor: str = Field(description="local or aws")
    storage: str = Field(description="Where session state is persisted")
    opencv_version: str
    model_weights_present: bool
    limits: dict[str, int] = Field(
        default_factory=dict,
        description="Server-enforced limits, so the client can validate before uploading",
    )


class ImageRef(BaseModel):
    """A reference to an image the client can fetch, plus its size."""

    url: str
    width: int
    height: int


class BoundingBox(BaseModel):
    """A box in canonical pixels, origin top-left."""

    x: int
    y: int
    width: int
    height: int


# --------------------------------------------------------------------------
# Sites and demo data
# --------------------------------------------------------------------------


class SiteSummary(BaseModel):
    """A site the operator can scan. Demo sites need no upload."""

    site_id: str
    name: str
    description: str
    is_demo: bool
    baseline_frames: int
    thumbnail: str | None = Field(
        default=None, description="Data URL of the baseline thumbnail, when available"
    )


class SiteList(BaseModel):
    sites: list[SiteSummary]


# --------------------------------------------------------------------------
# Sessions
# --------------------------------------------------------------------------


class CreateSessionRequest(BaseModel):
    """Open a session against a site.

    A session is one site under observation. Its baseline is the record every
    later rescan is compared against, so it is written once and then immutable -
    that immutability is what makes "what changed since?" a well-posed question.
    """

    site_id: str | None = Field(
        default=None, description="An existing site, including a demo site"
    )
    name: str | None = Field(default=None, max_length=120)
    baseline_frames: list[str] | None = Field(
        default=None,
        description="Base64 data URLs of the baseline frames, if not using a site",
    )

    @field_validator("baseline_frames")
    @classmethod
    def _frames_not_empty(cls, v: list[str] | None) -> list[str] | None:
        if v is not None and len(v) == 0:
            raise ValueError("baseline_frames was provided but empty")
        return v


class SessionState(BaseModel):
    """The full client-visible state of a session."""

    session_id: str
    site_id: str
    name: str
    created_at: str
    canonical_size: tuple[int, int]
    baseline_frames: int
    baseline: ImageRef | None = None
    coverage: dict[str, Any] | None = None
    guidance: ImageRef | None = None
    scans: int = 0
    committed: bool = False


# --------------------------------------------------------------------------
# Rescan
# --------------------------------------------------------------------------


class RescanRequest(BaseModel):
    """Supply rescan frames, or omit them to re-run the site's stored rescan."""

    frames: list[str] | None = Field(
        default=None, description="Base64 data URLs of the rescan frames"
    )
    use_site_rescan: bool = Field(
        default=True, description="Fall back to the site's own rescan when no frames are given"
    )
    min_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Hide events below this confidence"
    )


class EventOut(BaseModel):
    """One typed change, with the evidence that produced it."""

    event_id: str
    kind: str
    confidence: float
    bbox: BoundingBox
    centroid: tuple[float, float]
    displacement_px: float
    appearance_delta: float
    summary: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    uncertain: bool = Field(
        default=False, description="The agent flagged this as not confidently resolved"
    )


class TraceStepOut(BaseModel):
    """One step of the agent's reasoning, in the order it happened."""

    index: int
    phase: str
    action: str | None = None
    rationale: str | None = None
    decided_by: str | None = None
    confidence: float | None = None
    duration_ms: float | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)


class TraceOut(BaseModel):
    steps: list[TraceStepOut]
    brains_used: list[str]
    used_fallback: bool
    decision_count: int
    action_count: int


class RegistrationOut(BaseModel):
    model: str
    confidence: float
    inlier_count: int
    match_count: int
    inlier_ratio: float
    reproj_error_px: float | None = Field(
        default=None,
        description=(
            "Mean symmetric transfer error of the inliers, in canonical pixels. "
            "Null when there is no model to measure - an aborted registration has "
            "no error, which is different from an error of zero."
        ),
    )
    notes: list[str] = Field(default_factory=list)


class RescanResponse(BaseModel):
    """Everything one rescan produced."""

    session_id: str
    scan_index: int
    outcome: str = Field(description="finalized, declined (awaiting a decision), or aborted")
    feature_backend: str
    match_backend: str = Field(
        default="",
        description="The matcher that actually ran for this pair. It differs from "
                    "feature_backend when the structure guard diverts a thin pair "
                    "to the classical path or refuses a flat one.",
    )
    extra_views: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Further frames of the second visit, whether each was "
                    "registered, and what it contributed to coverage.",
    )
    registration: RegistrationOut
    events: list[EventOut]
    event_totals: dict[str, int]
    coverage: dict[str, Any]
    guidance: list[dict[str, Any]] = Field(
        default_factory=list, description="Ordered capture instructions for the operator"
    )
    trace: TraceOut
    timings_ms: dict[str, float]
    images: dict[str, ImageRef] = Field(
        default_factory=dict,
        description="baseline, rescan, overlay, side_by_side, guidance",
    )
    ground_truth: list[dict[str, Any]] | None = Field(
        default=None,
        description="Only present for demo sites, where the true changes are known",
    )


# --------------------------------------------------------------------------
# Commit gate
# --------------------------------------------------------------------------


class CommitRequest(BaseModel):
    """Approve or withhold the write of a scan into the site's record."""

    approved: bool
    note: str = Field(default="", max_length=500)


class LedgerEntry(BaseModel):
    """One immutable line of a site's history."""

    entry_id: str
    session_id: str
    scan_index: int
    recorded_at: str
    outcome: str
    approved: bool
    note: str
    event_totals: dict[str, int]
    registration_confidence: float
    decision_made_by: str = Field(
        description="human when the gate was answered, otherwise the agent"
    )


class LedgerOut(BaseModel):
    site_id: str
    entries: list[LedgerEntry]


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class ErrorOut(BaseModel):
    """A structured failure, so the client never has to parse prose."""

    error: str
    detail: str = ""
    hint: str = ""
