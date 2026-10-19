"""End-to-end orchestration of the STRATA pipeline.

One call does everything a site re-scan needs:

1. extract features from the baseline and the rescan;
2. match and robustly register the rescan into the baseline's canonical frame;
3. accumulate the baseline's coverage map and derive view requests;
4. warp the rescan into the canonical frame;
5. diff the two canonical frames, gated by registration residual and coverage;
6. resolve the difference into typed, confidence-scored events;
7. return a single result object with every artefact and every provenance
   field a report needs.

The pipeline is deliberately pure: it takes arrays and returns arrays plus
JSON-serialisable metadata. No filesystem, no network, no model downloads at
call time. That is what makes it deployable unchanged inside a Lambda
function, a container, or the local service.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import cv2
import numpy as np

from .config import PipelineConfig
from .coverage import (
    CoverageAccumulator,
    CoverageMap,
    ViewRequest,
    coverage_summary,
    view_requests,
)
from .diff import DiffResult, compute_diff
from .events import ChangeEvent, event_totals, resolve_events
from .features import FeatureSet, Matcher, build_extractor, resolve_backend, to_gray
from .geometry import Registration, register, render_canonical_baseline, warp_to_canonical

log = logging.getLogger("strata.pipeline")

ProgressFn = Callable[[str, float], None]


@dataclass
class StageTiming:
    """Wall-clock cost of each stage, in milliseconds.

    Recorded on every run rather than only under a profiler, so that "which stage
    is slow, and on which hardware" is answered by normal operation instead of
    requiring a separate benchmarking rig. A deployment that keeps these numbers
    accumulates a performance history for free.
    """

    extract_baseline_ms: float = 0.0
    extract_rescan_ms: float = 0.0
    match_ms: float = 0.0
    register_ms: float = 0.0
    coverage_ms: float = 0.0
    warp_ms: float = 0.0
    diff_ms: float = 0.0
    events_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> dict:
        return {k: round(v, 3) for k, v in self.__dict__.items()}


@dataclass
class ScanResult:
    """Everything one registration + diff produces."""

    baseline: np.ndarray                  # canonical baseline frame
    baseline_source: np.ndarray           # original baseline frame
    rescan_source: np.ndarray             # original rescan frame
    rescan_canonical: np.ndarray          # rescan warped into canonical frame
    registration: Registration
    diff: DiffResult
    events: list[ChangeEvent]
    coverage: CoverageMap
    view_requests: list[ViewRequest]
    timings: StageTiming
    feature_backend: str
    config: PipelineConfig
    # Which matcher actually produced this pair's correspondences. It is not
    # always ``feature_backend``: the structure guard can divert a thin pair to
    # the classical matcher, and a flat pair is refused outright. Reporting the
    # resolved backend alone would attribute a measurement to a matcher that
    # never ran.
    match_backend: str = ""
    config: PipelineConfig
    # What each further frame of the second visit contributed to coverage, and
    # why any of them did not. Empty unless extra frames were supplied.
    extra_views: list[dict[str, Any]] = field(default_factory=list)

    # --- Derived views -------------------------------------------------

    @property
    def event_totals(self) -> dict[str, int]:
        return event_totals(self.events)

    def overlay(self, min_confidence: float = 0.0) -> np.ndarray:
        """The annotated diff image a human looks at."""
        return _annotate(self.baseline, self.diff, self.events, min_confidence)

    def side_by_side(self) -> np.ndarray:
        """Baseline | rescan-canonical | diff, stacked horizontally."""
        h = self.baseline.shape[0]
        def fit(img: np.ndarray) -> np.ndarray:
            if img.shape[0] != h:
                s = h / float(img.shape[0])
                img = cv2.resize(img, (int(img.shape[1] * s), h), interpolation=cv2.INTER_AREA)
            return img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        panels = [fit(self.baseline), fit(self.rescan_canonical), fit(self.overlay())]
        sep = np.full((h, 3, 3), 230, np.uint8)
        out = panels[0]
        for p in panels[1:]:
            out = np.hstack([out, sep, p])
        return out

    def to_summary(self) -> dict[str, Any]:
        """A complete, JSON-serialisable account of the run.

        Everything a report or an integration needs, with no numpy arrays and no
        paths: the registration provenance, the diff statistics, every typed
        event, the coverage state, and the per-stage timings. If a number appears
        anywhere in this project's evaluation, it comes from here.
        """
        return {
            "version": "1.0",
            "feature_backend": self.feature_backend,
            "match_backend": self.match_backend,
            "executor": self.config.executor,
            "registration": self.registration.to_summary(),
            "diff": self.diff.to_summary(),
            "events": [e.to_dict() for e in self.events],
            "event_totals": self.event_totals,
            "coverage": coverage_summary(self.coverage, self.view_requests),
            "extra_views": list(self.extra_views),
            "timings_ms": self.timings.to_dict(),
        }


def _annotate(
    baseline: np.ndarray,
    diff: DiffResult,
    events: list[ChangeEvent],
    min_confidence: float,
) -> np.ndarray:
    from .events import render_events

    canvas = baseline.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    heat = cv2.applyColorMap(diff.diff_gray, cv2.COLORMAP_INFERNO)
    mask = diff.mask > 0
    blended = cv2.addWeighted(canvas, 0.55, heat, 0.45, 0)
    canvas[mask] = blended[mask]
    return render_events(canvas, events, min_confidence=min_confidence)


def _fit_to(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize an image to ``(width, height)`` unless it already matches."""
    w, h = size
    if image.shape[1] == w and image.shape[0] == h:
        return image
    interp = cv2.INTER_AREA if (image.shape[1] > w or image.shape[0] > h) else cv2.INTER_LINEAR
    return cv2.resize(image, (w, h), interpolation=interp)


def _coverage_accumulator(
    frames: list[np.ndarray],
    cfg: PipelineConfig,
    canonical_size: tuple[int, int],
    extractor,
) -> CoverageAccumulator:
    """Start a coverage map from one or more frames already in canonical space.

    Multi-frame coverage is what makes the guidance meaningful: a single frame
    sees part of the scene, and the union of several views sees more. The
    accumulator is fed each frame's own keypoints so cells that repeatedly
    yield matches count as genuinely observed.
    """
    acc = CoverageAccumulator(cfg.coverage, canonical_size)
    for frame in frames[: cfg.coverage.max_frames]:
        if frame.shape[:2] != (canonical_size[1], canonical_size[0]):
            frame = cv2.resize(frame, canonical_size, interpolation=cv2.INTER_AREA)
        gray = to_gray(frame)
        acc.add_frame(gray, extractor.detect(gray))
    return acc


def _add_extra_views(
    acc: CoverageAccumulator,
    frames: list[np.ndarray],
    base_features: FeatureSet,
    base_shape: tuple[int, int],
    canonical_size: tuple[int, int],
    extractor,
    matcher: Matcher,
    cfg: PipelineConfig,
) -> list[dict[str, Any]]:
    """Register each further frame of the second visit, and credit what it saw.

    This is what closes the perception loop. When the agent asks for another
    capture because coverage is too thin, the frame it receives has to change
    what the system knows it observed - otherwise "act" is theatre and the
    coverage bar can never be met. Each extra view is matched against the
    baseline, registered with the same estimator the primary frame uses, and
    warped in; only its inlier correspondences are credited, because those are
    the places where the view actually tells us something.

    A view that cannot be registered is reported as attempted and discarded
    rather than silently adding coverage: an unaligned frame makes a cell look
    observed when nothing in it can be trusted.
    """
    reports: list[dict[str, Any]] = []
    if not frames:
        return reports

    canon_w, canon_h = canonical_size
    base_h, base_w = base_shape
    scale = np.array([canon_w / float(base_w), canon_h / float(base_h)], dtype=np.float64)

    for index, frame in enumerate(frames, start=2):
        report: dict[str, Any] = {
            "frame": index,
            "registered": False,
            "confidence": 0.0,
            "inliers": 0,
            "points_credited": 0,
        }
        try:
            feats = extractor.detect(to_gray(frame))
            matches = matcher.match(feats, base_features)
            reg = register(
                matches, src_shape=frame.shape[:2], dst_shape=base_shape, cfg=cfg.geometry
            )
        except Exception as exc:  # a bad view must not fail the whole scan
            report["reason"] = f"could not be processed: {exc}"
            reports.append(report)
            continue

        report["confidence"] = round(float(reg.confidence), 4)
        report["backend"] = str(getattr(matches, "backend", "") or "")
        mask = reg.inliers_mask
        aligned = mask is not None and mask.shape[0] == len(matches) and bool(mask.any())
        if reg.model == "identity" or not aligned:
            report["reason"] = "no alignment found for this view"
            reports.append(report)
            continue
        if reg.confidence < cfg.coverage.extra_view_min_confidence:
            report["reason"] = (
                f"registered at {reg.confidence:.2f}, below the "
                f"{cfg.coverage.extra_view_min_confidence:.2f} bar for crediting coverage"
            )
            reports.append(report)
            continue

        warped, _valid = warp_to_canonical(frame, reg, canonical_size)
        points = matches.dst_points[mask].astype(np.float64) * scale
        acc.add_canonical_view(to_gray(warped), points.astype(np.float32))
        report.update(
            registered=True,
            inliers=int(mask.sum()),
            points_credited=int(len(points)),
        )
        reports.append(report)
    return reports


def run_scan(
    baseline_frames: list[np.ndarray],
    rescan_frames: list[np.ndarray],
    *,
    config: PipelineConfig | None = None,
    progress: ProgressFn | None = None,
) -> ScanResult:
    """Run the full registration + diff pipeline.

    Args:
        baseline_frames: frames of the first visit. The first frame anchors the
            canonical frame; the rest contribute to the coverage map.
        rescan_frames: frames of the second visit. The first is registered and
            diffed; the rest contribute coverage evidence.
        config: pipeline configuration; defaults are tuned for hand-held
            re-visits of a shelf-sized scene.
        progress: optional ``(stage, fraction)`` callback for a live UI.
    """
    cfg = config or PipelineConfig()
    timings = StageTiming()
    t_start = time.perf_counter()

    def tick(stage: str, frac: float) -> None:
        if progress:
            try:
                progress(stage, frac)
            except Exception:  # pragma: no cover - UI must never break a run
                log.debug("progress callback raised", exc_info=True)

    if not baseline_frames or not rescan_frames:
        raise ValueError("both baseline_frames and rescan_frames must be non-empty")

    tick("features", 0.05)
    extractor = build_extractor(cfg.features)
    backend = resolve_backend(cfg.features)
    matcher = Matcher(cfg.features, backend=backend)

    base0 = baseline_frames[0]
    res0 = rescan_frames[0]

    t0 = time.perf_counter()
    f_base = extractor.detect(to_gray(base0))
    timings.extract_baseline_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    f_res = extractor.detect(to_gray(res0))
    timings.extract_rescan_ms = (time.perf_counter() - t0) * 1000.0

    tick("match", 0.30)
    t0 = time.perf_counter()
    matches = matcher.match(f_res, f_base)  # rescan -> baseline
    timings.match_ms = (time.perf_counter() - t0) * 1000.0

    tick("register", 0.45)
    t0 = time.perf_counter()
    reg = register(
        matches, src_shape=res0.shape[:2], dst_shape=base0.shape[:2], cfg=cfg.geometry
    )
    timings.register_ms = (time.perf_counter() - t0) * 1000.0

    tick("warp", 0.58)
    t0 = time.perf_counter()
    canonical_size = reg.canonical_size
    canon_res, rescan_valid = warp_to_canonical(res0, reg, canonical_size)
    # The baseline owns the canonical frame, but it is still routed through the
    # same warp path so that both canonical images share one resampling
    # history. Resizing one and warping the other leaves a difference ring
    # around every high-contrast edge in the scene. See render_canonical_baseline.
    canon_base = render_canonical_baseline(base0, canonical_size)
    timings.warp_ms = (time.perf_counter() - t0) * 1000.0

    tick("coverage", 0.68)
    t0 = time.perf_counter()
    baseline_canonical = [canon_base] + [
        cv2.resize(f, canonical_size, interpolation=cv2.INTER_AREA)
        for f in baseline_frames[1:]
    ]
    accumulator = _coverage_accumulator(
        baseline_canonical, cfg, canonical_size, extractor
    )
    # The second visit's further frames are registered in and counted too. This
    # is the difference between an agent that can ask for another look and one
    # that can only ask and ignore the answer.
    extra_views = _add_extra_views(
        accumulator, rescan_frames[1:], f_base, base0.shape[:2],
        canonical_size, extractor, matcher, cfg,
    )
    cmap = accumulator.build()
    requests = view_requests(cmap, cfg.coverage)
    timings.coverage_ms = (time.perf_counter() - t0) * 1000.0

    tick("diff", 0.80)
    t0 = time.perf_counter()
    # The gate answers "was this area imaged", not "was it imaged well". A
    # smooth surface the camera covered is still a place where an object could
    # have appeared, so gating on feature quality would discard exactly the
    # evidence that matters most. See CoverageMap.imaged_mask.
    gate = cmap.imaged_mask(canonical_size)
    if rescan_valid is not None:
        gate = cv2.bitwise_and(gate, rescan_valid)
    diff_forward = compute_diff(
        canon_base, canon_res, cfg.diff,
        residual=reg.residual_map, coverage_mask=gate, valid_mask=rescan_valid,
    )
    timings.diff_ms = (time.perf_counter() - t0) * 1000.0

    tick("events", 0.92)
    t0 = time.perf_counter()
    events = resolve_events(
        diff_forward, cfg.events,
        baseline_shape=canon_base.shape[:2],
        rescan_shape=canon_res.shape[:2],
        baseline_image=canon_base,
        rescan_image=canon_res,
    )
    timings.events_ms = (time.perf_counter() - t0) * 1000.0

    timings.total_ms = (time.perf_counter() - t_start) * 1000.0
    tick("done", 1.0)

    return ScanResult(
        baseline=canon_base,
        baseline_source=base0,
        rescan_source=res0,
        rescan_canonical=canon_res,
        registration=reg,
        diff=diff_forward,
        events=events,
        coverage=cmap,
        view_requests=requests,
        timings=timings,
        feature_backend=backend,
        config=cfg,
        extra_views=extra_views,
        match_backend=str(getattr(matches, "backend", "") or ""),
    )


def analyse_capture_quality(frame: np.ndarray) -> dict[str, Any]:
    """Cheap per-frame quality signals for live capture guidance.

    Runs on every incoming frame, so it stays deliberately inexpensive: blur
    via Laplacian variance, exposure via mean/percentiles, texture via edge
    density. These drive the 'hold steady' / 'more light' / 'too close'
    messages an operator sees while filming.
    """
    gray = to_gray(frame)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean = float(np.mean(gray))
    p5, p95 = np.percentile(gray, [5, 95])
    edges = cv2.Canny(gray, 80, 180)
    edge_density = float(np.count_nonzero(edges)) / float(edges.size)

    messages: list[str] = []
    status = "good"
    if lap_var < 45:
        status = "blur"
        messages.append("Hold steadier - image is soft")
    if mean < 48:
        status = "dark"
        messages.append("More light needed")
    elif mean > 212 or (p95 - p5) < 30:
        status = "exposure"
        messages.append("Overexposed or flat - adjust exposure")
    if edge_density < 0.010:
        if status == "good":
            status = "low-texture"
        messages.append("Too little detail - move closer")

    return {
        "status": status,
        "sharpness": round(lap_var, 2),
        "mean_luma": round(mean, 2),
        "dynamic_range": round(float(p95 - p5), 2),
        "edge_density": round(edge_density, 5),
        "messages": messages,
    }
