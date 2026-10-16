"""Capture-coverage and uncertainty mapping for active perception.

This is the part of STRATA that makes it more than a scanner. A passive system
takes whatever frames it is given and hopes. STRATA maintains a live map of
which parts of the canonical frame are *well observed*, *weakly observed*, or
*not observed at all*, and then hands the agent a concrete, ranked list of
view requests: "you have not covered the lower-left shelf", "that corner is
occluded", "step back and re-shoot the top-right".

The map is coarse by design (a grid over the canonical frame) because its job
is to be *legible to a human on a phone screen* and *actionable by the agent*
in real time, not to be a high-resolution uncertainty field.

Two signals are accumulated per cell:

* **keypoint density** - cells that repeatedly yield matched features are
  genuinely observed.
* **texture / edge energy** - a cell with real structure but no keypoints is
  observed-but-unmatched (blur, motion, occlusion).

The result is a coverage score in ``[0, 1]`` per cell plus a list of the
worst-covered cells converted into directional guidance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import CoverageConfig
from .features import FeatureSet

log = logging.getLogger("strata.coverage")


@dataclass
class CoverageMap:
    """Per-cell observation quality over the canonical frame.

    Two distinct questions live in this one map, and conflating them is a
    mistake worth naming:

    * **Was this area imaged at all?** A cell is *imaged* when a session's
      field of view reached it. This is the permissive question, and it is the
      one that belongs in the diff gate: a pixel outside the field of view
      carries no information and must not be differenced.
    * **Was this area imaged *well*?** A cell is *well observed* when it yielded
      matched features, so its geometry is trustworthy. This is the demanding
      question, and it belongs in confidence and in the view requests the agent
      issues - "you have not covered the lower-left shelf" is guidance, not a
      reason to discard evidence.

    Using the demanding question as the diff gate silently throws away every
    smooth, featureless region - which is precisely where a newly placed object
    shows up against a plain surface. ``imaged`` therefore gates the diff, while
    ``score`` and the derived status drive guidance.
    """

    rows: int
    cols: int
    keypoint_counts: np.ndarray      # (rows, cols) int
    texture_energy: np.ndarray       # (rows, cols) float 0-1
    observed_fraction: float = 0.0
    status: str = "unknown"          # observed | weak | unobserved
    frame_count: int = 0
    # The canonical frame this grid covers. Carried explicitly so a consumer can
    # map a point in canonical pixels onto a cell without assuming a size - the
    # grid is resolution-independent, but the coordinates fed into it are not.
    canonical_size: tuple[int, int] = (1600, 1200)

    def heatmap(self, size: tuple[int, int] | None = None) -> np.ndarray:
        """Render the coverage score as a BGR heatmap at canonical size."""
        score = self.score()
        norm = np.clip(score * 255, 0, 255).astype(np.uint8)
        colour = cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS)
        if size is not None:
            colour = cv2.resize(colour, size, interpolation=cv2.INTER_NEAREST)
        return colour

    def mask(self, size: tuple[int, int] | None = None, threshold: float = 0.30) -> np.ndarray:
        """Binary canonical-size mask of cells considered *observed*.

        This is the demanding mask, suitable for reporting and for shading a
        confidence display. It is **not** the appropriate diff gate - use
        :meth:`imaged_mask` for that.
        """
        observed = (self.score() >= threshold).astype(np.uint8) * 255
        if size is not None:
            observed = cv2.resize(observed, size, interpolation=cv2.INTER_NEAREST)
        return observed

    def imaged_mask(self, size: tuple[int, int] | None = None) -> np.ndarray:
        """Binary mask of cells the camera's field of view actually reached.

        A cell counts as imaged when it registered *any* evidence at all -
        matched features or simply local contrast above the sensor's noise.
        The bar is deliberately low: this mask exists to exclude regions
        outside the capture, not to make a judgement about image quality.
        """
        kp = self.keypoint_counts.astype(np.float32)
        tex = np.clip(self.texture_energy, 0.0, 1.0)
        imaged = ((kp > 0) | (tex > 0.012)).astype(np.uint8) * 255
        if size is not None:
            imaged = cv2.resize(imaged, size, interpolation=cv2.INTER_NEAREST)
        return imaged

    def score(self) -> np.ndarray:
        """Per-cell observation *quality* in ``[0, 1]``.

        Quality is about how much trustworthy geometric evidence a cell
        yielded, which is what guidance and confidence need. It is not a
        statement about whether the area was imaged - see :meth:`imaged_mask`.
        """
        kp = self.keypoint_counts.astype(np.float32)
        kp_term = np.clip(kp / max(1.0, float(np.percentile(kp, 90) or 1.0)), 0.0, 1.0)
        return np.clip(0.65 * kp_term + 0.35 * np.clip(self.texture_energy, 0.0, 1.0), 0.0, 1.0)

    def to_dict(self) -> dict:
        return {
            "rows": self.rows,
            "cols": self.cols,
            "observed_fraction": round(float(self.observed_fraction), 4),
            "status": self.status,
            "frame_count": int(self.frame_count),
            "canonical_size": [int(self.canonical_size[0]), int(self.canonical_size[1])],
            "keypoint_counts": self.keypoint_counts.astype(int).tolist(),
            "texture_energy": np.round(self.texture_energy, 3).tolist(),
        }


@dataclass
class ViewRequest:
    """A concrete instruction for the operator to improve coverage."""

    direction: str          # e.g. "pan left", "step back"
    target_cell: tuple[int, int]
    deficit: float          # how far below the confident bar this cell sits
    priority: float         # higher = more urgent
    reason: str

    def to_dict(self) -> dict:
        return {
            "direction": self.direction,
            "target_cell": [int(self.target_cell[0]), int(self.target_cell[1])],
            "deficit": round(float(self.deficit), 4),
            "priority": round(float(self.priority), 4),
            "reason": self.reason,
        }


class CoverageAccumulator:
    """Accumulates per-cell observation evidence across a session's frames."""

    def __init__(self, cfg: CoverageConfig, canonical_size: tuple[int, int]) -> None:
        self.cfg = cfg
        self.canonical_size = canonical_size
        self.keypoint_counts = np.zeros((cfg.grid_rows, cfg.grid_cols), np.int32)
        self.texture_energy = np.zeros((cfg.grid_rows, cfg.grid_cols), np.float32)
        self.frame_count = 0

    def add_frame(self, canonical_gray: np.ndarray, features: FeatureSet | None = None) -> None:
        """Accumulate evidence from one canonicalised frame."""
        if self.frame_count >= self.cfg.max_frames:
            return
        self.frame_count += 1
        h, w = canonical_gray.shape[:2]
        self._add_texture(canonical_gray)

        if features is not None and len(features) > 0:
            pts = features.points.astype(np.float32)
            # features come from the source frame; scale into canonical space
            sh, sw = features.image_shape
            if sw > 0 and sh > 0:
                pts = pts * np.array(
                    [w / float(sw), h / float(sh)], dtype=np.float32
                )
            self._add_keypoints(pts, w, h)

    def _add_texture(self, gray: np.ndarray) -> None:
        h, w = gray.shape[:2]
        grad = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad = np.abs(grad) + np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
        rows = np.linspace(0, h, self.cfg.grid_rows + 1).astype(int)
        cols = np.linspace(0, w, self.cfg.grid_cols + 1).astype(int)
        for r in range(self.cfg.grid_rows):
            for c in range(self.cfg.grid_cols):
                cell = grad[rows[r]:rows[r + 1], cols[c]:cols[c + 1]]
                if cell.size == 0:
                    continue
                energy = float(np.mean(cell)) / 255.0
                self.texture_energy[r, c] += np.clip(energy, 0.0, 1.0)

    def _add_keypoints(self, pts: np.ndarray, w: int, h: int) -> None:
        if pts.size == 0:
            return
        col = np.clip((pts[:, 0] / max(1, w) * self.cfg.grid_cols).astype(int), 0, self.cfg.grid_cols - 1)
        row = np.clip((pts[:, 1] / max(1, h) * self.cfg.grid_rows).astype(int), 0, self.cfg.grid_rows - 1)
        for i in range(len(row)):
            self.keypoint_counts[row[i], col[i]] += 1

    def build(self) -> CoverageMap:
        """Finalise the accumulated evidence into a coverage map."""
        if self.frame_count > 1:
            texture = self.texture_energy / float(self.frame_count)
        else:
            texture = self.texture_energy
        cmap = CoverageMap(
            rows=self.cfg.grid_rows,
            cols=self.cfg.grid_cols,
            keypoint_counts=self.keypoint_counts,
            texture_energy=np.clip(texture, 0.0, 1.0),
            frame_count=self.frame_count,
            canonical_size=self.canonical_size,
        )
        cmap.observed_fraction = self._observed_fraction(cmap)
        cmap.status = _status_for(cmap.observed_fraction, self.cfg)
        return cmap

    def coverage_mask(self, threshold: float = 0.3) -> np.ndarray:
        """Binary canonical-size mask of observed cells (for diff gating)."""
        cmap = self.build()
        return cmap.mask(self.canonical_size, threshold=threshold)

    def _observed_fraction(self, cmap: CoverageMap) -> float:
        score = cmap.score()
        observed = score >= 0.30
        return float(np.count_nonzero(observed)) / float(observed.size)


def _status_for(fraction: float, cfg: CoverageConfig) -> str:
    if fraction >= cfg.confident_fraction:
        return "observed"
    if fraction >= cfg.weak_fraction:
        return "weak"
    return "unobserved"


def view_requests(
    cmap: CoverageMap,
    cfg: CoverageConfig,
    *,
    max_requests: int = 3,
) -> list[ViewRequest]:
    """Convert under-covered cells into ranked, directional instructions.

    The direction is derived from where the deficit sits relative to the
    centre of the frame, which is the mapping an operator intuitively follows:
    deficit on the left -> "pan left"; deficit at the edges -> "step back";
    deficit only high in the frame -> "tilt down".
    """
    score = cmap.score()
    rows, cols = score.shape
    confident = cfg.confident_fraction
    centre_r, centre_c = (rows - 1) / 2.0, (cols - 1) / 2.0

    deficits: list[tuple[float, int, int]] = []
    for r in range(rows):
        for c in range(cols):
            s = float(score[r, c])
            if s >= 0.30:
                continue
            # Weight by how far from the centre the cell is: edge misses matter
            # more because they usually mean an un-captured part of the scene.
            radial = np.hypot((r - centre_r) / max(1e-6, centre_r),
                              (c - centre_c) / max(1e-6, centre_c))
            deficit = (1.0 - s) * (0.6 + 0.4 * min(1.0, radial))
            deficits.append((deficit, r, c))

    if not deficits:
        return []

    deficits.sort(reverse=True)
    requests: list[ViewRequest] = []
    covered_cells: set[tuple[int, int]] = set()

    for deficit, r, c in deficits:
        if len(requests) >= max_requests:
            break
        # Skip cells adjacent to an already-requested one so guidance is spread.
        if any(abs(r - rr) <= 1 and abs(c - cc) <= 1 for rr, cc in covered_cells):
            continue
        covered_cells.add((r, c))
        direction, reason = _direction_for(r, c, centre_r, centre_c, rows, cols)
        requests.append(
            ViewRequest(
                direction=direction,
                target_cell=(r, c),
                deficit=deficit,
                priority=deficit,
                reason=reason,
            )
        )
    return requests


def _direction_for(
    r: int, c: int, centre_r: float, centre_c: float, rows: int, cols: int
) -> tuple[str, str]:
    dr = (r - centre_r) / max(1e-6, centre_r)
    dc = (c - centre_c) / max(1e-6, centre_c)

    if abs(dc) < 0.34 and abs(dr) < 0.34:
        return "step closer", "central region under-observed; move in for detail"
    if abs(dc) >= abs(dr):
        if dc < 0:
            return "pan left", f"left-of-frame cell ({r},{c}) is under-observed"
        return "pan right", f"right-of-frame cell ({r},{c}) is under-observed"
    if dr < 0:
        return "tilt up", f"upper cell ({r},{c}) is under-observed"
    return "tilt down", f"lower cell ({r},{c}) is under-observed"


def direction_to_cell(cmap: CoverageMap, cell: tuple[int, int]) -> tuple[str, str]:
    """Directional guidance for one specific cell.

    The same mapping :func:`view_requests` uses, exposed for callers that already
    know which cell they care about - the agent aiming a capture at an unresolved
    change, for instance, rather than at the worst-covered area in general.
    """
    rows, cols = cmap.rows, cmap.cols
    r, c = int(cell[0]), int(cell[1])
    return _direction_for(r, c, (rows - 1) / 2.0, (cols - 1) / 2.0, rows, cols)


def coverage_summary(cmap: CoverageMap, requests: list[ViewRequest]) -> dict:
    """Compact, serialisable coverage state for the agent and the UI."""
    return {
        "status": cmap.status,
        "observed_fraction": round(float(cmap.observed_fraction), 4),
        "frame_count": int(cmap.frame_count),
        "grid": [int(cmap.rows), int(cmap.cols)],
        "view_requests": [r.to_dict() for r in requests],
    }


def render_guidance(
    frame: np.ndarray,
    cmap: CoverageMap,
    requests: list[ViewRequest],
) -> np.ndarray:
    """Draw the coverage overlay and the primary guidance arrow onto a frame."""
    out = frame.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    h, w = out.shape[:2]

    overlay = cmap.heatmap((w, h))
    faint = cv2.addWeighted(out, 0.72, overlay, 0.28, 0)
    out = faint

    for req in requests:
        r, c = req.target_cell
        x1 = int(c / cmap.cols * w)
        x2 = int((c + 1) / cmap.cols * w)
        y1 = int(r / cmap.rows * h)
        y2 = int((r + 1) / cmap.rows * h)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 215, 255), 2)

    if requests:
        top = requests[0]
        r, c = top.target_cell
        cx = int((c + 0.5) / cmap.cols * w)
        cy = int((r + 0.5) / cmap.rows * h)
        cv2.arrowedLine(
            out, (w // 2, h - 40), (cx, cy), (0, 215, 255), 4, tipLength=0.18
        )
        banner = f"NEXT: {top.direction.upper()}"
        cv2.rectangle(out, (0, 0), (w, 42), (18, 18, 18), -1)
        cv2.putText(
            out, banner, (14, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 215, 255), 2, cv2.LINE_AA
        )
        sub = f"coverage {cmap.observed_fraction * 100:.0f}%  status {cmap.status}"
        cv2.putText(
            out, sub, (14, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA
        )
    return out
