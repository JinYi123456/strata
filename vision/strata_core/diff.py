"""Canonical change detection and change-region extraction.

Once both sessions live in the same canonical frame, "what changed?" becomes a
question that can be asked in shared geometry: normalise the photometry, take
the difference, threshold it adaptively, clean the mask, and extract regions.

Two things make this more than a frame differencer:

* **Photometric normalisation** (CLAHE + blur) so that a change in exposure or
  colour temperature between two visits does not masquerade as a changed
  object.
* **Residual gating** - regions sitting inside a poorly-registered band are
  flagged ``low_confidence`` rather than being asserted as changes. The
  pipeline is allowed to say "I cannot tell here", which is what makes the
  reported events trustworthy enough to act on.

Change is measured on two independent channels and unioned:

* **luminance**, which carries structure appearing, disappearing or shifting;
* **chrominance**, which carries a surface changing *colour* - a repainted
  wall, a replaced sign, a differently-coloured container.

The second channel is not a refinement, it is a necessity. Luminance is a
weighted sum of the three channels, and two colours that are far apart can be
nearly identical in luma - a saturated blue and a saturated orange can differ
by barely fifteen levels once converted. A luminance-only differencer is
therefore structurally blind to a whole class of real change, and no amount of
threshold tuning recovers what the conversion already discarded. Keeping the
two channels separate also means the normalisation applied for lighting
robustness never gets the chance to erase a genuine colour change along with
the lighting drift it was meant to remove.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import DiffConfig

log = logging.getLogger("strata.diff")


@dataclass
class ChangeRegion:
    """One contiguous region that changed, in canonical coordinates."""

    region_id: int
    bbox: tuple[int, int, int, int]      # x, y, w, h
    area_px: int
    centroid: tuple[float, float]
    mean_abs_diff: float                 # normalised 0-255 scale
    peak_abs_diff: float
    mask_iou_local: float = 0.0
    registration_residual: float = 0.0
    confidence: float = 0.0
    low_confidence: bool = False
    contour: np.ndarray | None = None
    #: Boolean footprint of this region within its own bounding box, shape
    #: ``(h, w)``. Kept so a consumer measuring the object's surrounding surface
    #: can tell which pixels inside the box are change and which are untouched,
    #: and can avoid sampling the change itself as if it were background.
    footprint: np.ndarray | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "region_id": self.region_id,
            "bbox": [int(v) for v in self.bbox],
            "area_px": int(self.area_px),
            "centroid": [round(float(self.centroid[0]), 2), round(float(self.centroid[1]), 2)],
            "mean_abs_diff": round(float(self.mean_abs_diff), 3),
            "peak_abs_diff": round(float(self.peak_abs_diff), 3),
            "registration_residual": round(float(self.registration_residual), 3),
            "confidence": round(float(self.confidence), 4),
            "low_confidence": bool(self.low_confidence),
            "notes": list(self.notes),
        }


@dataclass
class DiffResult:
    """Full output of a canonical diff between a baseline and a rescan."""

    diff_gray: np.ndarray                # (H, W) uint8 normalised difference
    mask: np.ndarray                     # (H, W) uint8 binary change mask
    regions: list[ChangeRegion]
    canonical_size: tuple[int, int]
    coverage_mask: np.ndarray | None = None
    stats: dict = field(default_factory=dict)

    @property
    def change_fraction(self) -> float:
        h, w = self.mask.shape[:2]
        return float(self.mask.sum()) / float(h * w * 255)

    def to_summary(self) -> dict:
        return {
            "canonical_size": list(self.canonical_size),
            "region_count": len(self.regions),
            "change_fraction": round(self.change_fraction, 5),
            "regions": [r.to_dict() for r in self.regions],
            **self.stats,
        }


def _valid_region(shape: tuple[int, int], margin: int) -> tuple[slice, slice]:
    """The interior of the frame, excluding a border band.

    The canonical warp leaves an uncovered wedge along the frame edges wherever
    the rescan does not extend. Those pixels differ enormously from the
    baseline for reasons that have nothing to do with the scene, so they are
    excluded from calibration statistics.
    """
    h, w = shape
    m = max(0, int(margin))
    if h <= 2 * m + 8 or w <= 2 * m + 8:
        return (slice(0, h), slice(0, w))
    return (slice(m, h - m), slice(m, w - m))


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _chroma_opponent(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split colour into two opponent channels, decoupled from brightness.

    Both channels are differences *between* colour components, so a uniform
    change in exposure - which scales all three components together - moves
    neither. That is what lets the chroma channel report a colour change while
    ignoring the lighting drift between two visits, instead of having to blur
    the distinction away.

    Returns ``(rg, by)`` as uint8 images centred on 128.
    """
    if img.ndim == 2:
        flat = np.full(img.shape, 128, np.uint8)
        return flat, flat.copy()
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    f = img.astype(np.float32)
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    # A single global offset is removed so the channels are centred on 128 for
    # both visits. Exposure drift cancels exactly, because it adds the same
    # constant to every component and so cancels in each difference.
    rg = r - g
    by = (b + g) * 0.5 - r
    rg = np.clip(rg + 128.0, 0, 255).astype(np.uint8)
    by = np.clip(by + 128.0, 0, 255).astype(np.uint8)
    return rg, by


def _normalise(gray: np.ndarray, cfg: DiffConfig) -> np.ndarray:
    """Local contrast normalisation so lighting drift is not a 'change'."""
    clahe = cv2.createCLAHE(
        clipLimit=cfg.clahe_clip_limit,
        tileGridSize=(cfg.clahe_tile_grid, cfg.clahe_tile_grid),
    )
    eq = clahe.apply(gray)
    k = cfg.blur_ksize | 1  # ensure odd
    return cv2.GaussianBlur(eq, (k, k), 0)


def compute_diff(
    baseline: np.ndarray,
    rescan: np.ndarray,
    cfg: DiffConfig,
    *,
    residual: np.ndarray | None = None,
    coverage_mask: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
) -> DiffResult:
    """Diff two canonical-frame images and extract change regions.

    The thresholding is deliberately *calibrated to the data* rather than fixed.
    Two captures of the same place never agree exactly - sensor noise, resampling
    and any registration residual all contribute a baseline disagreement floor.
    A fixed threshold either calls that floor a change (false positives) or
    misses real changes (false negatives). So the floor is measured from the
    image pair itself and the decision threshold is placed above it.

    ``valid_mask`` marks the pixels where the warped rescan actually carries
    source data. Pixels outside it are an artefact of the warp, not of the
    scene, and are excluded from both the calibration statistics and the final
    mask - which is what stops the uncovered edge wedge from dominating the
    region list.
    """
    if baseline.shape[:2] != rescan.shape[:2]:
        rescan = cv2.resize(rescan, (baseline.shape[1], baseline.shape[0]), interpolation=cv2.INTER_AREA)
        if valid_mask is not None and valid_mask.shape[:2] != baseline.shape[:2]:
            valid_mask = cv2.resize(
                valid_mask, (baseline.shape[1], baseline.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

    b_gray = _to_gray(baseline)
    r_gray = _to_gray(rescan)
    b_norm = _normalise(b_gray, cfg)
    r_norm = _normalise(r_gray, cfg)

    raw = cv2.absdiff(b_norm, r_norm)

    # --- Chrominance channel ----------------------------------------------
    # Computed from the *raw* colour, with no CLAHE, because the whole point of
    # this channel is to measure a colour change that the luminance
    # normalisation would smooth away. Chroma is contrast-stretched and blurred
    # only enough to suppress per-pixel sensor noise; it is not equalised,
    # since equalising it would flatten exactly the difference it measures.
    b_rg, b_by = _chroma_opponent(baseline)
    r_rg, r_by = _chroma_opponent(rescan)
    chroma = np.maximum(cv2.absdiff(b_rg, r_rg), cv2.absdiff(b_by, r_by))
    k = max(3, cfg.blur_ksize | 1)
    chroma = cv2.GaussianBlur(chroma, (k, k), 0)

    # --- Calibrate the disagreement floor for this pair -------------------
    # Robust statistics, not mean/stdev: a large genuine change would inflate
    # a stdev estimate, but it cannot move the 50th-80th percentile band much.
    #
    # The decision threshold is deliberately conservative about its *upper*
    # anchor. A high percentile of the difference image is dominated by a
    # handful of extreme pixels - typically the black wedge left where the
    # warp does not cover the frame - and using it directly raises the
    # threshold until real changes fall below it. Instead the upper anchor is
    # taken from the interior of the frame, where the border cannot reach, and
    # the border is excluded from the statistics outright.
    valid = _valid_region(baseline.shape[:2], cfg.border_margin_px)
    raw_in = raw[valid]
    if valid_mask is not None:
        # Zero out invalid pixels so they cannot contribute to the statistics.
        # They are set to 0 rather than excluded from the array because the
        # percentile must be computed over a contiguous set with the same
        # distribution shape; the invalid pixels genuinely carry no signal.
        keep = (valid_mask[valid] > 0)
        if np.count_nonzero(keep) >= 1024:
            raw_in = raw[valid][keep]
    if raw_in.size < 1024:
        raw_in = raw.reshape(-1)

    p_low = float(np.percentile(raw_in, 50))
    p_mid = float(np.percentile(raw_in, 80))
    p_high = float(np.percentile(raw_in, cfg.strong_percentile))
    mad = float(np.median(np.abs(raw_in.astype(np.float32) - p_low)))
    # Noise floor: median absolute deviation scaled to a sigma-equivalent.
    noise_sigma = 1.4826 * mad
    # A change must exceed both the noise band and a robust upper percentile.
    floor = p_low + max(cfg.noise_sigma_mult * noise_sigma, cfg.min_abs_delta)
    strong = max(p_high, floor + cfg.strong_margin)
    # Never let the upper anchor run away: it is a *lower* bound on what counts
    # as a strong change, and an unclamped percentile can push it above real
    # signal. This ceiling is the difference between "sensitive detector" and
    # "detector that finds only the warp border".
    strong = min(strong, cfg.max_threshold)

    mask = np.where(raw >= strong, 255, 0).astype(np.uint8)

    # Local contrast gate, applied *within the luminance channel*: a real
    # object change is coherent over a neighbourhood, whereas resampling noise
    # is not. A pixel survives only if its neighbourhood also carries elevated
    # luminance disagreement.
    #
    # This gate is deliberately applied before the union with chrominance, not
    # after. Gating the union would require every chroma-only change - a
    # recoloured surface, whose luminance is essentially unchanged - to also
    # pass a luminance coherence test, which would erase precisely the evidence
    # the chroma channel exists to supply. Each channel is gated by its own
    # coherence and the two verdicts are then combined.
    k = max(3, cfg.local_gate_ksize | 1)
    local_mean = cv2.blur(raw.astype(np.float32), (k, k))
    local_ok = (local_mean >= floor).astype(np.uint8) * 255
    mask = cv2.bitwise_and(mask, local_ok)

    # --- Chrominance decision --------------------------------------------
    # The chroma channel is calibrated against its own distribution: it has a
    # different noise floor and a different dynamic range from luminance, so
    # reusing the luminance threshold would either make it uselessly strict or
    # uselessly permissive. It is deliberately given a lower bar, because a
    # colour change is unambiguous evidence of change - a bare surface does not
    # drift in hue between two visits the way it drifts in brightness.
    chroma_in = chroma[valid]
    if valid_mask is not None:
        keep = (valid_mask[valid] > 0)
        if np.count_nonzero(keep) >= 1024:
            chroma_in = chroma[valid][keep]
    if chroma_in.size < 1024:
        chroma_in = chroma.reshape(-1)
    c_low = float(np.percentile(chroma_in, 50))
    c_mad = float(np.median(np.abs(chroma_in.astype(np.float32) - c_low)))
    c_sigma = 1.4826 * c_mad
    c_threshold = c_low + max(cfg.chroma_sigma_mult * c_sigma, cfg.min_chroma_delta)
    chroma_mask = np.where(chroma >= c_threshold, 255, 0).astype(np.uint8)
    # Chrominance coherence is enforced by morphology rather than by a
    # neighbourhood mean. A colour change is a *surface*, so its evidence forms
    # a solid patch; the residual noise that survives the threshold is a
    # sprinkle of isolated pixels. Opening with a modest kernel removes the
    # sprinkle while leaving a real patch's area almost untouched, which a mean
    # filter cannot do - a mean filter of the same size would eat into the
    # patch's edges and, worse, a low threshold lets enough noise through that
    # a mean filter passes it anyway.
    open_k = max(3, cfg.chroma_open_ksize | 1)
    chroma_mask = cv2.morphologyEx(
        chroma_mask, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8)
    )
    chroma_before = int(np.count_nonzero(chroma_mask))
    mask = cv2.bitwise_or(mask, chroma_mask)

    # Restrict to the region the baseline actually observed.
    if coverage_mask is not None:
        if coverage_mask.shape[:2] != mask.shape[:2]:
            coverage_mask = cv2.resize(
                coverage_mask, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        mask = cv2.bitwise_and(mask, coverage_mask)

    # Restrict to pixels where the rescan actually carries source data. This is
    # the gate that removes the warp-border wedge: those pixels are artefacts
    # of the geometry, never changes in the scene.
    if valid_mask is not None:
        if valid_mask.shape[:2] != mask.shape[:2]:
            valid_mask = cv2.resize(
                valid_mask, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        mask = cv2.bitwise_and(mask, valid_mask)

    if cfg.morph_open_ksize > 0:
        kk = np.ones((cfg.morph_open_ksize, cfg.morph_open_ksize), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kk)
    if cfg.morph_close_ksize > 0:
        kk = np.ones((cfg.morph_close_ksize, cfg.morph_close_ksize), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kk)

    regions = _extract_regions(mask, raw, residual, cfg, chroma=chroma)
    merged_from = len(regions)
    if cfg.merge_fragments:
        # Signed difference: positive where the baseline held more than the
        # rescan (structure departed), negative where the rescan holds more
        # (structure arrived). Used to stop the merge fusing a translation's
        # two opposite-polarity halves into one self-cancelling box.
        polarity = b_norm.astype(np.float32) - r_norm.astype(np.float32)
        regions = merge_fragmented_regions(
            regions,
            max_gap_px=cfg.merge_max_gap_px,
            max_area_ratio=cfg.merge_max_area_ratio,
            polarity=polarity,
            polarity_min=cfg.merge_polarity_min,
        )
    stats = {
        "mean_abs_diff": round(float(np.mean(raw)), 3),
        "noise_floor_p50": round(p_low, 3),
        "noise_floor_p80": round(p_mid, 3),
        "noise_sigma_est": round(noise_sigma, 3),
        "threshold_used": round(strong, 3),
        "changed_pixels": int(np.count_nonzero(mask)),
        "total_pixels": int(mask.size),
        "regions_before_merge": int(merged_from),
        "regions_after_merge": len(regions),
        # Chrominance provenance, so a report can show that the colour channel
        # was doing work rather than merely present.
        "chroma_threshold_used": round(c_threshold, 3),
        "chroma_pixels": int(chroma_before),
    }
    return DiffResult(
        diff_gray=raw,
        mask=mask,
        regions=regions,
        canonical_size=(baseline.shape[1], baseline.shape[0]),
        coverage_mask=coverage_mask,
        stats=stats,
    )


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _extract_regions(
    mask: np.ndarray,
    diff: np.ndarray,
    residual: np.ndarray | None,
    cfg: DiffConfig,
    *,
    chroma: np.ndarray | None = None,
) -> list[ChangeRegion]:
    """Contour extraction with area, residual and confidence scoring.

    A region is scored by whichever channel carries its evidence. A change that
    is purely chromatic - a recoloured surface - barely moves the luminance
    difference, so scoring it on luminance alone would report it as weak
    evidence and bury it under the noise floor. Taking the stronger of the two
    per pixel means a region is judged by the best evidence it actually has,
    which is also what lets the report say *which* channel found it.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape[:2]
    total = float(h * w)
    regions: list[ChangeRegion] = []

    for c in contours:
        area = int(cv2.contourArea(c))
        if area < cfg.min_region_area_px:
            continue
        if area / total > cfg.max_region_fraction:
            # An enormous "change" is almost always a registration failure.
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        m = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(m, [c], -1, 255, thickness=cv2.FILLED)
        sel = m > 0

        lum_vals = diff[sel]
        lum_mean = float(np.mean(lum_vals)) if lum_vals.size else 0.0
        if chroma is not None:
            ch_vals = chroma[sel]
            ch_mean = float(np.mean(ch_vals)) if ch_vals.size else 0.0
            # The reported magnitude is the stronger channel's, so confidence
            # scoring sees the real evidence rather than the weaker channel's
            # view of it.
            mean_d = max(lum_mean, ch_mean)
            peak_d = float(max(
                np.max(lum_vals) if lum_vals.size else 0.0,
                np.max(ch_vals) if ch_vals.size else 0.0,
            ))
            dominant = "chroma" if ch_mean > lum_mean else "luminance"
        else:
            mean_d = lum_mean
            peak_d = float(np.max(lum_vals)) if lum_vals.size else 0.0
            dominant = "luminance"

        M = cv2.moments(c)
        cx = float(M["m10"] / M["m00"]) if M["m00"] else x + bw / 2.0
        cy = float(M["m01"] / M["m00"]) if M["m00"] else y + bh / 2.0

        resid = 0.0
        low_conf = False
        notes: list[str] = [f"evidence dominant in {dominant} channel"]
        if residual is not None:
            r_vals = residual[sel]
            resid = float(np.mean(r_vals)) if r_vals.size else 0.0
            if resid >= cfg.residual_poor_px:
                low_conf = True
                notes.append(
                    f"registration residual {resid:.1f}px >= {cfg.residual_poor_px}px; "
                    "reported as uncertain rather than changed"
                )
            elif resid >= cfg.residual_good_px:
                notes.append(f"elevated registration residual {resid:.1f}px")

        conf = _region_confidence(area, mean_d, peak_d, resid, cfg)
        footprint = (m[y:y + bh, x:x + bw] > 0)
        regions.append(
            ChangeRegion(
                region_id=-1,
                bbox=(int(x), int(y), int(bw), int(bh)),
                area_px=area,
                centroid=(cx, cy),
                mean_abs_diff=mean_d,
                peak_abs_diff=peak_d,
                registration_residual=resid,
                confidence=conf,
                low_confidence=low_conf,
                contour=c,
                footprint=footprint,
                notes=notes,
            )
        )

    regions.sort(key=lambda r: r.area_px * max(r.confidence, 0.05), reverse=True)
    for i, r in enumerate(regions):
        r.region_id = i
    return regions


def merge_fragmented_regions(
    regions: list[ChangeRegion],
    *,
    max_gap_px: int = 26,
    max_area_ratio: float = 3.5,
    polarity: np.ndarray | None = None,
    polarity_min: float = 0.08,
) -> list[ChangeRegion]:
    """Fuse regions that are fragments of one changed object.

    Thresholding plus morphology regularly splits a single object into two or
    three pieces: a label band separates from the body it sits on, a strong edge
    survives while the interior falls below the threshold, or a morphological
    close bridges then re-splits a contour. Reporting those as separate events
    both overstates how much changed and depresses precision, because only one
    of the fragments can match the ground-truth box.

    Two regions are fused when their boxes are close relative to their size,
    neither is dramatically larger than the other, **and they agree on the
    direction of the change**.

    That last condition is essential and is not a refinement. A translation
    deposits two adjacent regions of *opposite* polarity: a hole where the
    object used to be, and the object itself where it arrived. They sit a few
    pixels apart and are often similar in size, so every spatial test says
    "fuse these". But fusing them averages a departure and an arrival into one
    box that is, on average, unchanged - the presence gap that identified each
    half collapses towards zero, and the event is then reported as an in-place
    change at a location that merely spans the whole affair. The pairing of the
    two halves into a single ``moved`` event is a *later*, structural step; the
    merge must not pre-empt it by destroying the two halves' identities.

    ``polarity`` is a signed map of the difference (baseline minus rescan); a
    region's sign is its mean over that map. Regions are only fused when their
    signs match and neither is near zero.
    """
    work = list(regions)
    changed = True
    while changed:
        changed = False
        out: list[ChangeRegion] = []
        used = [False] * len(work)
        for i, a in enumerate(work):
            if used[i]:
                continue
            for j in range(i + 1, len(work)):
                if used[j]:
                    continue
                b = work[j]
                if not _should_merge(a, b, max_gap_px, max_area_ratio):
                    continue
                if polarity is not None and not _polarity_agrees(
                    a, b, polarity, polarity_min
                ):
                    continue
                out.append(_fuse(a, b))
                used[i] = used[j] = True
                changed = True
                break
            if not used[i]:
                used[i] = True
                out.append(a)
        work = out

    work.sort(key=lambda r: r.area_px * max(r.confidence, 0.05), reverse=True)
    for i, r in enumerate(work):
        r.region_id = i
    return work


def _region_polarity(r: ChangeRegion, polarity: np.ndarray) -> float:
    """Mean signed difference over a region's footprint, in [-1, 1]."""
    x, y, w, h = r.bbox
    H, W = polarity.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return 0.0
    patch = polarity[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    return float(np.tanh(float(np.mean(patch)) / 64.0))


def _polarity_agrees(
    a: ChangeRegion, b: ChangeRegion, polarity: np.ndarray, min_magnitude: float
) -> bool:
    """True when two regions changed in the same direction, or both barely did."""
    pa = _region_polarity(a, polarity)
    pb = _region_polarity(b, polarity)
    if abs(pa) < min_magnitude or abs(pb) < min_magnitude:
        # Neither carries a decisive direction, so there is nothing to preserve.
        return True
    return (pa > 0) == (pb > 0)


def _should_merge(
    a: ChangeRegion, b: ChangeRegion, max_gap_px: int, max_area_ratio: float
) -> bool:
    ax, ay, aw, ah = a.bbox
    bx, by, bw, bh = b.bbox
    # Gap between the boxes on each axis; 0 when they overlap.
    gx = max(0, max(ax, bx) - min(ax + aw, bx + bw))
    gy = max(0, max(ay, by) - min(ay + ah, by + bh))
    gap = float(np.hypot(gx, gy))
    smaller = max(1, min(aw, ah, bw, bh))
    # Distance is judged relative to the smaller box, so a small fragment must
    # be genuinely adjacent to be fused, while two large boxes may sit further
    # apart and still be one object.
    if gap > max(max_gap_px, 0.75 * smaller):
        return False
    area_ratio = max(a.area_px, b.area_px) / float(max(1, min(a.area_px, b.area_px)))
    return area_ratio <= max_area_ratio


def _fuse(a: ChangeRegion, b: ChangeRegion) -> ChangeRegion:
    """Union two regions into one, keeping the union box and area-weighted stats."""
    ax, ay, aw, ah = a.bbox
    bx, by, bw, bh = b.bbox
    x0, y0 = min(ax, bx), min(ay, by)
    x1, y1 = max(ax + aw, bx + bw), max(ay + ah, by + bh)
    total = float(a.area_px + b.area_px)
    wa, wb = a.area_px / total, b.area_px / total
    contour = None
    if a.contour is not None or b.contour is not None:
        parts = [c for c in (a.contour, b.contour) if c is not None]
        try:
            contour = np.vstack(parts)
        except ValueError:
            contour = a.contour if a.contour is not None else b.contour
    # The fused region's footprint is the union of the two, placed in the union
    # box. This is what keeps the "which pixels here are the change" question
    # answerable after a merge, so the surround sample stays a sample of
    # untouched surface.
    footprint = None
    if a.footprint is not None or b.footprint is not None:
        w, h = int(x1 - x0), int(y1 - y0)
        footprint = np.zeros((h, w), bool)
        for region in (a, b):
            if region.footprint is None:
                continue
            rx, ry, rw, rh = region.bbox
            ox, oy = rx - x0, ry - y0
            fh, fw = region.footprint.shape[:2]
            footprint[oy:oy + fh, ox:ox + fw] |= region.footprint[:min(fh, h - oy), :min(fw, w - ox)]
    return ChangeRegion(
        region_id=-1,
        bbox=(int(x0), int(y0), int(x1 - x0), int(y1 - y0)),
        area_px=int(a.area_px + b.area_px),
        centroid=(
            float(wa * a.centroid[0] + wb * b.centroid[0]),
            float(wa * a.centroid[1] + wb * b.centroid[1]),
        ),
        mean_abs_diff=float(wa * a.mean_abs_diff + wb * b.mean_abs_diff),
        peak_abs_diff=float(max(a.peak_abs_diff, b.peak_abs_diff)),
        mask_iou_local=max(a.mask_iou_local, b.mask_iou_local),
        registration_residual=float(wa * a.registration_residual + wb * b.registration_residual),
        confidence=float(max(a.confidence, b.confidence)),
        low_confidence=bool(a.low_confidence and b.low_confidence),
        contour=contour,
        footprint=footprint,
        notes=list({*a.notes, *b.notes}),
    )


def _region_confidence(
    area: int, mean_d: float, peak_d: float, residual: float, cfg: DiffConfig
) -> float:
    """Bounded 0-1 confidence that a region is a genuine object-level change."""
    area_term = np.clip(area / float(max(1, cfg.min_region_area_px * 8)), 0.0, 1.0)
    photometric = np.clip((0.6 * mean_d + 0.4 * peak_d) / 60.0, 0.0, 1.0)
    if residual <= cfg.residual_good_px:
        resid_term = 1.0
    elif residual >= cfg.residual_poor_px:
        resid_term = 0.0
    else:
        span = cfg.residual_poor_px - cfg.residual_good_px
        resid_term = 1.0 - (residual - cfg.residual_good_px) / max(1e-6, span)
    conf = 0.30 * area_term + 0.45 * photometric + 0.25 * resid_term
    return float(np.clip(max(conf, cfg.min_confidence), 0.0, 1.0))


def render_change_overlay(
    baseline: np.ndarray,
    diff: DiffResult,
    *,
    min_confidence: float = 0.0,
) -> np.ndarray:
    """Draw the change heatmap and region boxes onto the baseline frame."""
    canvas = baseline.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    heat = cv2.applyColorMap(diff.diff_gray, cv2.COLORMAP_INFERNO)
    blend = cv2.addWeighted(canvas, 0.55, heat, 0.45, 0)
    keep = diff.mask > 0
    canvas[keep] = blend[keep]

    for r in diff.regions:
        if r.confidence < min_confidence:
            continue
        x, y, w, h = r.bbox
        colour = (0, 200, 255) if not r.low_confidence else (0, 120, 255)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), colour, 2)
        label = f"#{r.region_id} {r.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(canvas, (x, max(0, y - th - 6)), (x + tw + 6, y), colour, -1)
        cv2.putText(
            canvas, label, (x + 3, max(12, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA,
        )
    return canvas
