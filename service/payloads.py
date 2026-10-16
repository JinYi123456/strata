"""Turning pipeline and agent output into the wire format.

Kept apart from :mod:`runners` so that "run the vision" and "shape the response"
can change independently, and so the coordinate-space conventions are stated in
exactly one place.

Two conventions are enforced here, because getting them wrong produces a UI that
looks right and is wrong:

* **Boxes are ``(x, y, width, height)`` in canonical pixels.** The core reports
  them that way and the schema names the fields, so a client cannot mistake a
  width for an x-coordinate.
* **Ground truth is only ever attached to demo sites, and is labelled.** It
  travels in a separate field from detections and carries its own coordinate
  space, so it can never be read as something the system inferred.
"""

from __future__ import annotations

import math
from typing import Any, Callable

from strata_core.events import ChangeEvent

from .models import (
    BoundingBox,
    EventOut,
    ImageRef,
    RegistrationOut,
    RescanResponse,
    TraceOut,
    TraceStepOut,
)
from .runners import PANELS, ScanOutcome


def _num(value: Any, digits: int = 4) -> float | None:
    """Round a float for the wire, or null if JSON cannot represent it.

    An identity registration reports an infinite reprojection error, because
    there is no model to measure. JSON has no infinity and encoding one raises
    ``ValueError: Out of range float values are not JSON compliant`` - so the
    aborted-registration case, which is precisely the case that most needs a
    clean response, would return a 500 instead. Null is the honest encoding: it
    says "not measured", which is what happened. NaN is treated the same way,
    for the same reason.

    Applied to every float that leaves the service rather than only to the one
    that was observed to misbehave, because the next non-finite value will come
    from somewhere nobody predicted.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return round(v, digits) if math.isfinite(v) else None


def _bbox(raw: tuple[int, int, int, int]) -> BoundingBox:
    x, y, w, h = (int(v) for v in raw)
    return BoundingBox(x=x, y=y, width=w, height=h)


def event_out(event: ChangeEvent, *, uncertain: bool) -> EventOut:
    return EventOut(
        event_id=event.event_id,
        kind=event.kind,
        confidence=_num(event.confidence) or 0.0,
        bbox=_bbox(event.bbox),
        centroid=(round(float(event.centroid[0]), 2), round(float(event.centroid[1]), 2)),
        displacement_px=_num(event.displacement_px) or 0.0,
        appearance_delta=_num(event.appearance_delta) or 0.0,
        summary=event.summary,
        evidence=dict(event.evidence or {}),
        uncertain=uncertain,
    )


def trace_out(outcome: ScanOutcome) -> TraceOut:
    trace = outcome.run.trace
    steps = [
        TraceStepOut(
            index=int(s.index),
            phase=str(s.phase),
            action=s.action,
            rationale=s.rationale,
            decided_by=s.decided_by,
            confidence=_num(s.confidence),
            duration_ms=_num(s.duration_ms, digits=3),
            args=dict(s.args or {}),
            result=dict(s.result or {}),
        )
        for s in trace.steps
    ]
    return TraceOut(
        steps=steps,
        brains_used=list(trace.brains_used),
        used_fallback=bool(trace.used_fallback),
        decision_count=len(trace.decisions),
        action_count=len(trace.actions),
    )


def registration_out(outcome: ScanOutcome) -> RegistrationOut:
    reg = outcome.result.registration
    return RegistrationOut(
        model=str(reg.model),
        confidence=_num(reg.confidence) or 0.0,
        inlier_count=int(reg.inlier_count),
        match_count=int(reg.match_count),
        inlier_ratio=_num(reg.inlier_ratio) or 0.0,
        reproj_error_px=_num(reg.reproj_error_px),
        notes=list(reg.notes),
    )


def image_refs(image_url: Callable[[str], str], panels: dict[str, Any]) -> dict[str, ImageRef]:
    """One :class:`ImageRef` per panel that was actually produced."""
    out: dict[str, ImageRef] = {}
    for name in PANELS:
        img = panels.get(name)
        if img is None:
            continue
        h, w = img.shape[:2]
        out[name] = ImageRef(url=image_url(name), width=int(w), height=int(h))
    return out


def build_rescan_response(
    outcome: ScanOutcome,
    *,
    session_id: str,
    scan_index: int,
    image_url: Callable[[str], str],
    min_confidence: float = 0.0,
    ground_truth: list[dict[str, Any]] | None = None,
) -> RescanResponse:
    """Assemble the full response for one scan.

    The filter is applied to the *presentation* of events, never to the count of
    what was found: ``event_totals`` always reflects everything the system
    reported, so hiding low-confidence rows in the UI cannot quietly change the
    headline number.
    """
    uncertain = outcome.uncertain_ids
    all_events = outcome.events
    shown = [e for e in all_events if float(e.confidence) >= min_confidence]

    totals: dict[str, int] = {}
    for e in all_events:
        totals[e.kind] = totals.get(e.kind, 0) + 1

    if outcome.has_result:
        result = outcome.result
        assert result is not None  # for type checkers; has_result guarantees it
        return RescanResponse(
            session_id=session_id,
            scan_index=scan_index,
            outcome=str(outcome.run.outcome),
            feature_backend=str(result.feature_backend),
            registration=registration_out(outcome),
            events=[event_out(e, uncertain=e.event_id in uncertain) for e in shown],
            event_totals=totals,
            coverage={
                "status": result.coverage.status,
                "observed_fraction": round(float(result.coverage.observed_fraction), 4),
                "frame_count": int(result.coverage.frame_count),
                "grid": [int(result.coverage.rows), int(result.coverage.cols)],
                "view_requests": [r.to_dict() for r in result.view_requests],
            },
            guidance=[r.to_dict() for r in result.view_requests],
            trace=trace_out(outcome),
            timings_ms=result.timings.to_dict(),
            images=image_refs(image_url, outcome.panels),
            ground_truth=ground_truth,
        )

    # No scan was produced - the loop stopped first. The trace says why, and the
    # response says so plainly rather than inventing an empty result that would
    # read as "nothing changed", which is a different and much worse claim.
    return RescanResponse(
        session_id=session_id,
        scan_index=scan_index,
        outcome=str(outcome.run.outcome),
        feature_backend="",
        registration=RegistrationOut(
            model="identity",
            confidence=0.0,
            inlier_count=0,
            match_count=0,
            inlier_ratio=0.0,
            reproj_error_px=None,
            notes=["no scan was produced"],
        ),
        events=[],
        event_totals={},
        coverage={},
        guidance=[],
        trace=trace_out(outcome),
        timings_ms={},
        images={},
        ground_truth=ground_truth,
    )
