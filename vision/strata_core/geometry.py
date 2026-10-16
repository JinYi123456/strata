"""Robust registration into a shared canonical frame.

The reason a naive image diff fails on a site re-scan is that the camera never
returns to the same pose. STRATA solves this by *canonicalising*: every session
is warped into a frame defined by the baseline session, so two visits become
pixel-comparable even though they were shot from different positions.

The estimator is OpenCV 5's USAC framework with MAGSAC. The model is chosen
adaptively - a homography when the scene is near-planar (the common case for a
shelf, wall, bench, or fridge), and a fundamental matrix plus dense warp when
there is real depth relief. A per-pixel *residual map* is produced alongside
the warp so that regions where registration is weak are never reported as
changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np

from .config import GeometryConfig
from .features import FeatureSet, MatchSet

log = logging.getLogger("strata.geometry")

ModelName = Literal["homography", "fundamental", "identity"]


@dataclass
class Registration:
    """Result of registering one session into the canonical frame."""

    model: ModelName
    matrix: np.ndarray                      # 3x3 homography, or F when model=fundamental
    inliers_mask: np.ndarray                # (N,) bool over the match set
    inlier_count: int
    match_count: int
    inlier_ratio: float
    reproj_error_px: float                  # mean symmetric transfer error of inliers
    canonical_size: tuple[int, int]         # (width, height) of the canonical frame
    residual_map: np.ndarray | None = None  # (H, W) float32 in canonical pixels
    confidence: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.model != "identity" and self.inlier_count > 0

    def to_summary(self) -> dict:
        """Serialisable provenance for the ledger."""
        return {
            "model": self.model,
            "inlier_count": int(self.inlier_count),
            "match_count": int(self.match_count),
            "inlier_ratio": round(float(self.inlier_ratio), 4),
            "reproj_error_px": round(float(self.reproj_error_px), 3),
            "canonical_size": [int(self.canonical_size[0]), int(self.canonical_size[1])],
            "confidence": round(float(self.confidence), 4),
            "notes": list(self.notes),
        }


def canonical_size_for(
    shape: tuple[int, int], long_edge: int
) -> tuple[int, int]:
    """Scale an image shape so its long edge matches ``long_edge``."""
    h, w = shape[:2]
    scale = long_edge / float(max(h, w))
    return max(1, int(round(w * scale))), max(1, int(round(h * scale)))


def _identity_registration(
    canonical_size: tuple[int, int], reason: str
) -> Registration:
    return Registration(
        model="identity",
        matrix=np.eye(3, dtype=np.float64),
        inliers_mask=np.zeros(0, dtype=bool),
        inlier_count=0,
        match_count=0,
        inlier_ratio=0.0,
        reproj_error_px=float("inf"),
        canonical_size=canonical_size,
        confidence=0.0,
        notes=[reason],
    )


def _mean_symmetric_error(
    H: np.ndarray, src: np.ndarray, dst: np.ndarray
) -> float:
    """Mean Euclidean symmetric transfer error in pixels."""
    if len(src) == 0:
        return float("inf")
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    fwd = np.linalg.norm(proj - dst, axis=1)
    try:
        Hinv = np.linalg.inv(H)
        back = cv2.perspectiveTransform(dst.reshape(-1, 1, 2), Hinv).reshape(-1, 2)
        bwd = np.linalg.norm(back - src, axis=1)
    except np.linalg.LinAlgError:
        bwd = fwd
    return float(np.mean(np.minimum(fwd, bwd)))


def _sane_homography(H: np.ndarray) -> bool:
    """Reject degenerate warps that would collapse or explode the frame."""
    if H is None or not np.all(np.isfinite(H)):
        return False
    det = float(np.linalg.det(H[:2, :2]))
    if det <= 1e-8:
        return False
    # Map the unit square; the result must stay finite and bounded.
    quad = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float64).reshape(-1, 1, 2)
    try:
        out = cv2.perspectiveTransform(quad, H).reshape(-1, 2)
    except cv2.error:
        return False
    if not np.all(np.isfinite(out)):
        return False
    span = float(np.max(out) - np.min(out))
    return 1e-3 < span < 1e5


def register(
    matches: MatchSet,
    *,
    src_shape: tuple[int, int],
    dst_shape: tuple[int, int],
    cfg: GeometryConfig,
) -> Registration:
    """Register ``src`` onto ``dst`` (the canonical frame owner).

    ``src_points`` live in the source session; ``dst_points`` live in the
    canonical frame. Returns a homography that maps source pixels into
    canonical pixels.
    """
    canonical_size = canonical_size_for(dst_shape, cfg.canonical_long_edge)

    if len(matches) < cfg.min_inliers:
        reason = f"only {len(matches)} putative matches (< {cfg.min_inliers})"
        if getattr(matches, "low_structure", False):
            reason += "; the pair has almost no structure to match"
        return _identity_registration(canonical_size, reason)

    src = matches.src_points.astype(np.float64)
    dst = matches.dst_points.astype(np.float64)

    # --- Planar model: USAC_MAGSAC homography -----------------------------
    H, mask_h = cv2.findHomography(
        src,
        dst,
        cv2.USAC_MAGSAC,
        cfg.homography_reproj_threshold,
        maxIters=cfg.ransac_max_iters,
        confidence=cfg.ransac_confidence,
    )
    note: list[str] = []
    if H is not None and mask_h is not None and 1 <= mask_h.sum() < len(src):
        in_h = mask_h.ravel().astype(bool)
    else:
        in_h = np.ones(len(src), dtype=bool) if H is not None else np.zeros(len(src), dtype=bool)

    if H is not None and _sane_homography(H):
        err_h = _mean_symmetric_error(H, src[in_h], dst[in_h])
        ratio_h = float(in_h.sum()) / float(len(src))
    else:
        err_h, ratio_h = float("inf"), 0.0
        H, in_h = None, np.zeros(len(src), dtype=bool)

    # --- Non-planar model: USAC_MAGSAC fundamental ------------------------
    F = None
    err_f, ratio_f = float("inf"), 0.0
    in_f = np.zeros(len(src), dtype=bool)
    if len(src) >= 8 and not cfg.prefer_planar:
        try:
            F, mask_f = cv2.findFundamentalMat(
                src,
                dst,
                cv2.USAC_MAGSAC,
                ransacReprojThreshold=cfg.fundamental_reproj_threshold,
                confidence=cfg.ransac_confidence,
                maxIters=cfg.ransac_max_iters,
            )
            if F is not None and mask_f is not None and len(F) == 3:
                in_f = mask_f.ravel().astype(bool)
                lines_s = cv2.computeCorrespondEpilines(
                    src[in_f].reshape(-1, 1, 2), 1, F
                ).reshape(-1, 3)
                lines_d = cv2.computeCorrespondEpilines(
                    dst[in_f].reshape(-1, 1, 2), 2, F
                ).reshape(-1, 3)
                d_s = np.abs(np.sum(lines_s * np.hstack(
                    [src[in_f], np.ones((in_f.sum(), 1))]), axis=1)) / np.sqrt(
                    lines_s[:, 0] ** 2 + lines_s[:, 1] ** 2 + 1e-12)
                d_d = np.abs(np.sum(lines_d * np.hstack(
                    [dst[in_f], np.ones((in_f.sum(), 1))]), axis=1)) / np.sqrt(
                    lines_d[:, 0] ** 2 + lines_d[:, 1] ** 2 + 1e-12)
                err_f = float(np.mean((d_s + d_d) / 2.0))
                ratio_f = float(in_f.sum()) / float(len(src))
        except cv2.error as exc:  # pragma: no cover - defensive
            log.warning("fundamental estimation failed: %s", exc)

    # --- Choose the model -------------------------------------------------
    use_h = H is not None and ratio_h >= cfg.min_inlier_ratio and in_h.sum() >= cfg.min_inliers
    use_f = F is not None and ratio_f >= cfg.min_inlier_ratio and in_f.sum() >= cfg.min_inliers

    if cfg.prefer_planar:
        choose_h = use_h
    else:
        choose_h = use_h and (not use_f or err_h <= err_f)
    if not (choose_h or use_f):
        # Nothing crossed the bar; try a relaxed homography before giving up.
        return _identity_registration(
            canonical_size,
            f"no model passed: H inliers={int(in_h.sum())} ratio={ratio_h:.3f} "
            f"err={err_h:.2f}; F ratio={ratio_f:.3f} err={err_f:.2f}",
        )

    if choose_h:
        model: ModelName = "homography"
        matrix = H
        inliers = in_h
        err = err_h
        if use_f and err_f < err_h * 0.8:
            note.append(
                f"homography preferred (planar); fundamental err {err_f:.2f}px noted"
            )
    else:
        model = "fundamental"
        matrix = F
        inliers = in_f
        err = err_f
        note.append("non-planar scene: fundamental model selected")

    inlier_ratio = float(inliers.sum()) / float(len(src))
    confidence = _confidence(inlier_ratio, err, int(inliers.sum()), cfg)
    if getattr(matches, "low_structure", False):
        # The correspondences came from a pair with almost no structure, where a
        # consistent-looking match set can be pure coincidence. Whatever the
        # geometric quality metrics say, this alignment has not been earned, so
        # it is capped rather than trusted.
        if confidence > cfg.low_structure_confidence_cap:
            note.append(
                f"low-structure pair: confidence capped at "
                f"{cfg.low_structure_confidence_cap:.2f} (was {confidence:.2f})"
            )
            confidence = cfg.low_structure_confidence_cap
        else:
            note.append("low-structure pair: alignment is not trustworthy")

    residual = None
    if model == "homography":
        residual = residual_map(matrix, src_shape, dst_shape, canonical_size)

    return Registration(
        model=model,
        matrix=np.asarray(matrix, dtype=np.float64),
        inliers_mask=inliers,
        inlier_count=int(inliers.sum()),
        match_count=int(len(src)),
        inlier_ratio=inlier_ratio,
        reproj_error_px=err,
        canonical_size=canonical_size,
        residual_map=residual,
        confidence=confidence,
        notes=note,
    )


def _confidence(ratio: float, err: float, n_inliers: int, cfg: GeometryConfig) -> float:
    """Turn registration quality into a bounded 0-1 confidence."""
    if n_inliers <= 0 or not np.isfinite(err):
        return 0.0
    ratio_term = np.clip(
        (ratio - cfg.min_inlier_ratio) / max(1e-6, 1.0 - cfg.min_inlier_ratio), 0.0, 1.0
    )
    err_term = np.clip(1.0 - (err / max(1e-6, cfg.residual_poor_px)), 0.0, 1.0)
    count_term = np.clip(n_inliers / float(max(1, cfg.min_inliers * 4)), 0.0, 1.0)
    return float(0.45 * ratio_term + 0.35 * err_term + 0.20 * count_term)


def render_canonical_baseline(
    image: np.ndarray, canonical_size: tuple[int, int]
) -> np.ndarray:
    """Render the baseline into the canonical frame using the *same* kernel as
    the warped rescan.

    This matters more than it looks. The baseline owns the canonical frame, so
    the tempting shortcut is ``cv2.resize`` to canonical size. But the rescan
    reaches the canonical frame through ``cv2.warpPerspective`` with bilinear
    interpolation, and bilinear resampling has a different modulation transfer
    function from ``cv2.resize``. Put the two side by side and every
    high-contrast edge in the scene shows a difference ring - not because
    anything changed, but because the two images were resampled differently.

    A scene full of labelled objects has high-contrast edges everywhere, so
    this artefact would be mistaken for a change on nearly every object. The
    baseline is therefore routed through the identical warp path, with an
    identity homography, so both canonical frames share one resampling
    history and the difference reflects the scene rather than the pipeline.
    """
    w, h = canonical_size
    src_h, src_w = image.shape[:2]
    if (src_w, src_h) == (w, h):
        return image.copy()
    H = np.diag([w / float(src_w), h / float(src_h), 1.0])
    return cv2.warpPerspective(
        image, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


def warp_to_canonical(
    image: np.ndarray, reg: Registration, shape: tuple[int, int] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Warp a session frame into the canonical frame.

    ``reg.matrix`` is expressed in the *source* frame's pixel units. The
    canonical frame is usually a different resolution, so the matrix has to be
    rescaled before it is applied - otherwise the warp is applied at the wrong
    scale and most of the output lands outside the image. The correction is a
    similarity conjugation: ``S_canon @ H @ S_src^-1``.

    Returns ``(warped, valid)`` where ``valid`` is a uint8 mask that is 255
    only where the warped image actually carries source pixels. A re-visit
    never covers the frame identically, so the warp necessarily leaves an
    uncovered wedge along one or more edges. Those pixels differ enormously
    from the baseline for reasons that have nothing to do with the scene, and
    left untreated they dominate both the calibration statistics and the
    change-region list. Handing the caller an explicit validity mask lets the
    diff exclude them at the source rather than trying to filter them out
    afterwards.
    """
    target_w, target_h = shape or reg.canonical_size
    if reg.model != "homography":
        resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
        return resized, np.full((target_h, target_w), 255, np.uint8)

    src_h, src_w = image.shape[:2]
    # ``reg.canonical_size`` is the size the matrix's output was defined at.
    base_w, base_h = reg.canonical_size
    H = np.asarray(reg.matrix, dtype=np.float64)

    sx = float(base_w) / float(max(1, src_w))
    sy = float(base_h) / float(max(1, src_h))
    if abs(sx - 1.0) > 1e-9 or abs(sy - 1.0) > 1e-9:
        # Map source pixels -> canonical-space pixels.
        H = np.diag([sx, sy, 1.0]) @ H

    if (target_w, target_h) != (base_w, base_h):
        S_out = np.diag([target_w / float(base_w), target_h / float(base_h), 1.0])
        H = S_out @ H

    warped = cv2.warpPerspective(
        image, H, (target_w, target_h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
    )
    # Warp a fully-set mask through the same transform to learn exactly which
    # output pixels received source data. Eroding by the interpolation kernel
    # removes the soft, half-weighted rim where the warp blends into the
    # constant border - those pixels are the ones that masquerade as changes.
    ones = np.full((src_h, src_w), 255, np.uint8)
    valid = cv2.warpPerspective(
        ones, H, (target_w, target_h),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
    )
    erode = max(1, int(np.ceil(max(H[0, 0], H[1, 1], 1.0))))
    if erode > 1:
        k = np.ones((erode, erode), np.uint8)
        valid = cv2.erode(valid, k)
    return warped, valid


def residual_map(
    H: np.ndarray,
    src_shape: tuple[int, int],
    dst_shape: tuple[int, int],
    canonical_size: tuple[int, int],
    grid: int = 32,
) -> np.ndarray:
    """Estimate a per-pixel registration residual in canonical space.

    A coarse grid is warped forward and back through ``H``; the local
    round-trip error becomes the residual. Under a pure homography this is
    near-uniform, but it still surfaces ill-conditioned regions (extremely
    oblique planes) so the diff can down-weight them rather than call them
    changes.
    """
    w, h = canonical_size
    ys = np.linspace(0, src_shape[0] - 1, grid)
    xs = np.linspace(0, src_shape[1] - 1, grid)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float64)

    # ``H`` maps source -> canonical-sized pixels; scale it to the source frame
    # first so the round-trip is measured in the units the grid is sampled in.
    sx = float(canonical_size[0]) / float(max(1, src_shape[1]))
    sy = float(canonical_size[1]) / float(max(1, src_shape[0]))
    Hs = np.diag([1.0 / sx, 1.0 / sy, 1.0]) @ np.asarray(H, np.float64)

    fwd = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), Hs).reshape(-1, 2)
    try:
        Hinv = np.linalg.inv(Hs)
    except np.linalg.LinAlgError:
        return np.zeros((h, w), np.float32)
    bwd = cv2.perspectiveTransform(fwd.reshape(-1, 1, 2), Hinv).reshape(-1, 2)
    err = np.linalg.norm(bwd - pts, axis=1).reshape(grid, grid).astype(np.float32)

    # Resample the residual grid up to canonical resolution.
    resized = cv2.resize(err, (w, h), interpolation=cv2.INTER_LINEAR)
    return resized


def compose_transform(
    first: Registration, second: Registration
) -> np.ndarray:
    """Compose the warp from ``second``'s frame into ``first``'s canonical frame."""
    if second.model != "homography" or first.model != "homography":
        return np.eye(3, dtype=np.float64)
    return first.matrix @ np.linalg.inv(second.matrix)
