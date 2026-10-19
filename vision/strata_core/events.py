"""Change regions resolved into typed, explained events.

A region of differing pixels is not yet an answer. The useful unit of output is
an *event*: "the blue box that was here is gone", "a new object appeared where
nothing was", "this object shifted", "this object is still here but its state
changed".

Classification is a **two-image comparison in canonical geometry**. The two
canonical images carry independent evidence, whereas two difference maps of the
same pair carry identical evidence and cannot be compared to each other.

The decisive design problem is *presence*: deciding whether a box contains an
object or merely the scene's own background texture. An absolute "is there
texture here" test cannot work, because a real shelf, wall or bench is itself
full of structure - edges, seams, banding, grain. A patch of empty shelf is not
flat, so any absolute measure of structure will call it an object.

The fix is to make presence **local and comparative**. For a box, we compare
the structure *inside* it against the structure of a *reference band* drawn
from the same image immediately around it:

    presence = structure_inside / (structure_inside + structure_surround)

If the box is an object sitting on background, its interior out-structures its
surround and presence approaches 1. If the box is a patch of background, its
interior and its surround look alike and presence approaches 0.5. If the box is
*less* structured than its surround (a smooth patch cutting into a busy scene),
presence falls below 0.5.

This is scale-free and content-free: it never has to know what "an object" looks
like, only whether this patch stands out from its own immediate neighbourhood.
That is what makes it work on a textured shelf and a plain wall alike.

Rules, all evaluated in canonical geometry:

* **missing**       - presence high in the baseline, low in the rescan.
* **added**         - presence high in the rescan, low in the baseline.
* **moved**         - presence high in both, and the baseline interior matches
                      the rescan at a translated location far better than it
                      matches the rescan in place.
* **state_changed** - presence high in both, matching in place, appearance
                      differs beyond the re-visit floor.
* **uncertain**     - registration is poor here, or neither capture holds
                      structure, so the system declines to assert a kind.
                      Reported honestly rather than guessed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np

from .config import EventConfig
from .diff import ChangeRegion, DiffResult

log = logging.getLogger("strata.events")

EventKind = Literal["added", "missing", "moved", "state_changed", "uncertain"]


@dataclass
class ChangeEvent:
    """A typed, explained, confidence-scored change."""

    event_id: str
    kind: EventKind
    confidence: float
    regions: list[int]
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    displacement_px: float
    appearance_delta: float
    summary: str
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "kind": self.kind,
            "confidence": round(float(self.confidence), 4),
            "regions": [int(r) for r in self.regions],
            "bbox": [int(v) for v in self.bbox],
            "centroid": [round(float(self.centroid[0]), 2), round(float(self.centroid[1]), 2)],
            "displacement_px": round(float(self.displacement_px), 2),
            "appearance_delta": round(float(self.appearance_delta), 3),
            "summary": self.summary,
            "evidence": self.evidence,
        }


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _patch_variance(gray: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 3) -> float:
    """Gradient energy inside a box - a proxy for 'is there an object here'."""
    x, y, w, h = bbox
    H, W = gray.shape[:2]
    x0, y0 = max(0, x + pad), max(0, y + pad)
    x1, y1 = min(W, x + w - pad), min(H, y + h - pad)
    if x1 - x0 < 4 or y1 - y0 < 4:
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return 0.0
    patch = gray[y0:y1, x0:x1]
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(np.abs(gx)) + np.mean(np.abs(gy)))


def _local_std(gray: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    x, y, w, h = bbox
    H, W = gray.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return 0.0
    return float(np.std(gray[y0:y1, x0:x1]))


def _structure_energy(patch: np.ndarray) -> float:
    """Local structure energy of a patch: gradient magnitude plus dispersion.

    Both terms are scale-normalised so patches of different sizes are
    comparable, and both are computed on the patch's own pixel values so the
    result is invariant to overall brightness and to what the scene contains.
    """
    if patch.size < 16:
        return 0.0
    pf = patch.astype(np.float32)
    gx = cv2.Sobel(pf, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(pf, cv2.CV_32F, 0, 1, ksize=3)
    grad = float(np.mean(np.sqrt(gx * gx + gy * gy)))
    # Interquartile dispersion resists a single bright edge dominating the score.
    q75, q25 = np.percentile(pf, [75.0, 25.0])
    disp = float(q75 - q25)
    return grad + 0.5 * disp


def _presence(
    gray: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    ring: int | None = None,
    ring_fraction: float = 0.45,
    ring_min: int = 10,
    ring_max: int = 90,
    change_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """How much does this box stand out from its own immediate surround?

    The measure is deliberately comparative rather than absolute, because a
    real scene's background is itself full of texture. See the module docstring
    for why an absolute test cannot work.

    The surround band is sized *as a fraction of the box* rather than as a
    fixed number of pixels. That keeps the comparison meaningful across
    resolutions: at canonical scale a box may be three times larger than it was
    in the source frame, and a fixed-width ring would then sample a sliver that
    says more about the box than about its surroundings.

    ``change_mask``, when supplied, excludes pixels of the change itself from
    the **surround** band. This matters because a change region's bounding box
    is not a tight fit around the object: it is the bounding rectangle of the
    mask of differing pixels, so a ring drawn just outside it can still fall
    *inside* the object that appeared or departed. When that happens the
    reference is drawn from the very thing being measured, the two energies
    converge, and the comparison silently loses its power - an arrival whose box
    sits slightly inside the object reads as no arrival at all.

    The **interior** is deliberately measured over the whole box, including any
    changed pixels. The question being asked is "does this location hold an
    object", and the object *is* the changed pixels; excluding them would
    replace the measurement with a reading of whatever surface lies beneath,
    which is the same in both captures and therefore carries no information
    about whether something is there.

    Returns a dict with:

    * ``presence``  - ``inside / (inside + surround)`` in ``[0, 1]``. ~0.5 means
      the box looks like its surroundings; >0.5 means the box holds more
      structure than its surround; <0.5 means it holds less.
    * ``inside`` / ``surround`` - the raw structure energies, for evidence.
    * ``contrast``  - inside minus surround, normalised; a signed sharpness of
      the comparison around the 0.5 pivot.
    """
    H, W = gray.shape[:2]
    x, y, w, h = bbox
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return {"presence": 0.5, "inside": 0.0, "surround": 0.0, "contrast": 0.0}

    inside = _structure_energy(gray[y0:y1, x0:x1])

    if ring is None:
        ring = int(np.clip(
            round(ring_fraction * min(x1 - x0, y1 - y0)), ring_min, ring_max
        ))

    # Surround band: the ring of pixels immediately outside the box, clipped to
    # the image. Taken from the *same* image so it carries that image's own
    # background statistics.
    rx0, ry0 = max(0, x0 - ring), max(0, y0 - ring)
    rx1, ry1 = min(W, x1 + ring), min(H, y1 + ring)
    outer = gray[ry0:ry1, rx0:rx1]
    if outer.size == 0:
        return {"presence": 0.5, "inside": inside, "surround": 0.0, "contrast": 0.0}

    # Build the ring by masking out the interior so the object itself does not
    # contaminate its own reference.
    mask = np.ones(outer.shape[:2], np.uint8)
    ix0, iy0 = x0 - rx0, y0 - ry0
    ix1, iy1 = ix0 + (x1 - x0), iy0 + (y1 - y0)
    mask[iy0:iy1, ix0:ix1] = 0
    # Also exclude the rest of the change itself, so the reference is the
    # unchanged surface rather than the object being measured.
    if change_mask is not None:
        cm_outer = change_mask[ry0:ry1, rx0:rx1]
        if cm_outer.shape == mask.shape:
            mask[cm_outer > 0] = 0
    ring_pixels = outer[mask > 0]
    if ring_pixels.size < 16:
        return {"presence": 0.5, "inside": inside, "surround": 0.0, "contrast": 0.0}

    # Structure energy of the ring, computed on the ring's own footprint so the
    # two numbers are directly comparable.
    surround = _ring_energy(outer.astype(np.float32), mask)

    denom = inside + surround
    if denom <= 1e-6:
        return {"presence": 0.5, "inside": inside, "surround": surround, "contrast": 0.0}
    presence = float(np.clip(inside / denom, 0.0, 1.0))
    contrast = float(np.clip((inside - surround) / denom, -1.0, 1.0))
    return {
        "presence": presence,
        "inside": round(inside, 4),
        "surround": round(surround, 4),
        "contrast": contrast,
    }


def _structure_energy_masked(patch: np.ndarray, keep: np.ndarray) -> float:
    """Structure energy of a patch, computed only over the kept pixels."""
    if patch.size < 16 or not np.any(keep):
        return 0.0
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    grad_term = float(np.mean(grad[keep]))
    vals = patch[keep]
    q75, q25 = np.percentile(vals, [75.0, 25.0])
    return grad_term + 0.5 * float(q75 - q25)


def _ring_energy(outer: np.ndarray, mask: np.ndarray) -> float:
    """Structure energy computed only over the ring pixels of a masked patch.

    Sobel is applied to the full outer block (so gradients at the interior
    boundary are real), then averaged only where the mask is set.
    """
    of = outer.astype(np.float32, copy=False)
    gx = cv2.Sobel(of, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(of, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    sel = mask > 0
    if not np.any(sel):
        return 0.0
    grad_term = float(np.mean(grad[sel]))
    vals = of[sel]
    q75, q25 = np.percentile(vals, [75.0, 25.0])
    return grad_term + 0.5 * float(q75 - q25)


#: Presence above this reads as "this box holds its own structure" - an object.
PRESENCE_FLOOR = 0.56
#: Presence above this is a confident object rather than marginal structure.
PRESENCE_STRONG = 0.64
#: Below this, a box reads as flatter than its surround (a smooth patch).
PRESENCE_ABSENT = 0.50

# Backwards-compatible aliases for callers that referenced the old constant.
OBJECTNESS_FLOOR = PRESENCE_FLOOR
OBJECTNESS_STRONG = PRESENCE_STRONG


def resolve_events(
    baseline_diff: DiffResult,
    cfg: EventConfig,
    *,
    baseline_shape: tuple[int, int],
    rescan_shape: tuple[int, int],
    baseline_image: np.ndarray | None = None,
    rescan_image: np.ndarray | None = None,
    motion_tolerance_px: float = 16.0,
    event_prefix: str = "evt",
) -> list[ChangeEvent]:
    """Turn change regions into typed events.

    When ``baseline_image`` and ``rescan_image`` are supplied, each region is
    classified by comparing what each image actually contains at that location.
    Without them the function degrades gracefully to presence-only typing, which
    is what the synthetic evaluation harness exercises.
    """
    regions = list(baseline_diff.regions)
    if not regions:
        return []

    base_gray = _to_gray(baseline_image) if baseline_image is not None else None
    res_gray = _to_gray(rescan_image) if rescan_image is not None else None

    events: list[ChangeEvent] = []
    for i, r in enumerate(regions):
        kind, conf, disp, delta, reason, extra = _classify(
            r, cfg, base_gray, res_gray, motion_tolerance_px
        )
        events.append(
            ChangeEvent(
                event_id=f"{event_prefix}-{i+1:03d}",
                kind=kind,
                confidence=conf,
                regions=[r.region_id],
                bbox=r.bbox,
                centroid=r.centroid,
                displacement_px=disp,
                appearance_delta=delta,
                summary=_summarise(kind, r, disp, delta),
                evidence={
                    "reason": reason,
                    "region_id": r.region_id,
                    "area_px": r.area_px,
                    "mean_abs_diff": round(r.mean_abs_diff, 3),
                    "registration_residual": round(r.registration_residual, 3),
                    **extra,
                },
            )
        )

    events = _resolve_translations(events, cfg, base_gray, res_gray)

    events.sort(key=lambda e: (e.kind == "uncertain", -e.confidence))
    for n, e in enumerate(events):
        e.event_id = f"{event_prefix}-{n+1:03d}"
    return events


def _resolve_translations(
    events: list[ChangeEvent],
    cfg: EventConfig,
    base_gray: np.ndarray | None = None,
    res_gray: np.ndarray | None = None,
) -> list[ChangeEvent]:
    """Promote a departure/arrival pair into a single ``moved`` event.

    A translated object leaves a *paired* signature: a departure where it used
    to stand and an arrival where it now stands, close together and of similar
    shape. Neither half is a removal or an insertion on its own - the object
    still exists, it simply is not where it was.

    Single-region classification cannot see this, and not merely because it
    looks at one box at a time. A translation's departure is *pixel-identical*
    to a removal's: in both cases an object was standing there and now bare
    surface is. The same is true of the arrival, which cannot be told from an
    insertion by looking at it alone. The two cases become distinguishable only
    when the whole region set is in hand, and only by their *co-occurrence* -
    a removal's counterpart arrival does not exist. That is why this pass runs
    here rather than in the per-region classifier.

    An arrival is not required to have been classified ``added``. It was, once,
    and the measurement says that requirement was wrong. A moved object lands
    where structure is present in *both* captures, so the presence gap at its
    destination is small by construction - measured across the benchmark, not
    one destination in six was ever labelled ``added``, which meant the pairing
    pass never once saw a candidate and every ``moved`` the system reported came
    from the classifier's independent shift test rather than from pairing at
    all. Whether a destination is typed ``added`` or ``state_changed`` is
    genuinely ambiguous from a single region: both readings are defensible, and
    the truth is only recoverable by looking at the departure. So the kind is
    not a filter here; it is *evidence*, recorded on the merged event and
    overridden once the pair is established.

    Pairing is solved as a global assignment rather than first-fit. When two
    departures sit within range of one arrival - or of each other's arrivals -
    taking the first acceptable partner lets a move claim another move's
    destination, which is visible in the benchmark as two correctly-localised
    events with transposed kinds. Every candidate pair is therefore enumerated,
    scored, and matched in ascending cost order so the cheapest pairs are made
    first and no region is used twice.

    The merged event's confidence reflects how well the two halves corroborate
    each other, and the displacement is the shift between their centroids.
    """
    departures = [
        e for e in events
        if e.kind == "missing" and e.evidence.get("area_px", 0) > 0
    ]
    # Candidate arrivals: any region of comparable extent that is not itself a
    # departure. The kind is deliberately not constrained - see above.
    arrivals = [
        e for e in events
        if e is not None
        and e.evidence.get("area_px", 0) > 0
        and e.kind != "missing"
    ]
    if not departures or not arrivals:
        return events

    # --- Enumerate every admissible pair ------------------------------------
    candidates: list[tuple[float, float, int, int]] = []  # (cost, sep, dep_i, arr_i)
    for di, dep in enumerate(departures):
        for ai, arr in enumerate(arrivals):
            if arr is dep:
                continue
            # The separation that matters is between the two *boxes*, not
            # between their centroids. A change region is bounded by the mask of
            # differing pixels, and that mask is not symmetric around the object:
            # an object sitting on a textured surface produces most of its
            # evidence inside its own footprint, while the bare patch it vacates
            # produces evidence across the whole exposed area. Fragmentation and
            # merging then move the two centroids towards each other, so a real
            # translation of a few hundred pixels can present as a centroid gap
            # of under a hundred. Measuring box-edge separation is invariant to
            # that, because both boxes still sit where the object was and is.
            gap_x = max(dep.bbox[0] - (arr.bbox[0] + arr.bbox[2]),
                        arr.bbox[0] - (dep.bbox[0] + dep.bbox[2]), 0.0)
            gap_y = max(dep.bbox[1] - (arr.bbox[1] + arr.bbox[3]),
                        arr.bbox[1] - (dep.bbox[1] + dep.bbox[3]), 0.0)
            sep = float(np.hypot(gap_x, gap_y))
            if sep > cfg.translation_max_px:
                continue
            # Centroid distance must be *positive*: two events at the same place
            # are not a translation, however they are typed.
            dx = dep.centroid[0] - arr.centroid[0]
            dy = dep.centroid[1] - arr.centroid[1]
            centroid_dist = float(np.hypot(dx, dy))
            if centroid_dist < cfg.translation_min_px:
                continue
            # ...and it must be bounded, which the box-edge test above cannot
            # ensure on its own. Box-edge separation shrinks as the regions
            # grow, so an oversized arrival - a region that is several things
            # merged together rather than one object - can keep its edge near
            # the departure while its centre sits most of a scene away.
            #
            # The bound is flat, and that is a deliberate, measured trade-off
            # rather than an oversight. Allowing the two regions' own extents as
            # slack was tried first, on the reasoning that the centroid distance
            # is the gap plus those extents. It re-admitted exactly the case
            # this guard exists for: a bloated arrival earns a *larger*
            # allowance precisely because it is bloated, so three of the seven
            # benchmark scenes produced a spurious translation again. A flat
            # bound costs one genuine pair - a 436px displacement, just over the
            # limit - and produces none. For a system whose value rests on not
            # inventing changes, missing one is the better error.
            if centroid_dist > cfg.translation_max_px:
                continue
            # The arrival must show *some* arrival evidence - a region that
            # recorded no positive change in presence is not something that
            # appeared, whatever else is true of it. This is the weakest
            # possible version of the test, deliberately: the measured
            # distributions of every stronger version overlapped between
            # correct and incorrect pairs, so a threshold placed inside that
            # overlap would reject real translations.
            arr_gap = float(arr.evidence.get("presence_gap", 0.0) or 0.0)
            if arr_gap <= cfg.pair_arrival_gap_min:
                continue
            # A loose sanity bound on relative size. This is not a
            # discriminator - correct and incorrect pairs both spanned
            # 0.49-0.99 - it only rules out the physically impossible, where one
            # region is several times the other and no single object could
            # account for both. The bound is set well below the smallest ratio
            # observed for a correct pair so it cannot reject one.
            if not _size_compatible(dep, arr, cfg.translation_size_ratio_min):
                continue
            # Aspect-ratio agreement, used as a *cost* and deliberately NOT as
            # a gate.
            #
            # It is tempting to gate on it: the two halves of a translation
            # bound the same object, so their proportions ought to match, and
            # the pairs the algorithm actually chose did separate cleanly
            # (0.81-0.99 for correct pairs against 0.53 for the wrong one). But
            # that measurement was circular - the algorithm had partly chosen
            # them *because* of this score. Measured over all *candidate* pairs
            # instead, the classes overlap completely: correct pairs ranged
            # 0.40-1.00 and incorrect ones 0.32-0.99. No threshold separates
            # them, so gating here would reject real translations while
            # admitting false ones.
            #
            # As a cost term it remains useful, because it only has to rank
            # candidates that are all admissible. Squaring it gives shape more
            # influence than raw distance, which is what lets a well-shaped
            # partner 197px away beat a wrongly-shaped blob 145px away.
            shape = _box_similarity(dep.bbox, arr.bbox)
            # An arrival with strong evidence of its own is a better partner
            # than one that barely registered.
            evidence_weight = 1.0 + 2.0 * max(0.0, cfg.pair_arrival_gap_min * 2.0 - arr_gap)
            cost = sep * (2.0 - shape) ** 2 * evidence_weight
            candidates.append((cost, sep, di, ai))

    if not candidates:
        return events

    # --- Solve as a global assignment ---------------------------------------
    #
    # Ascending cost, first-fit. A full Hungarian solve would be tidier, but
    # the pair count here is small and the cost is already a distance: taking
    # the globally cheapest pair first, then the next cheapest that does not
    # reuse a region, gives the same answer whenever the costs are meaningfully
    # separated, and cannot do worse than first-fit-by-departure when they are
    # not. What it fixes is the crossing case, where two moves are close enough
    # that a departure's nearest arrival belongs to the other move.
    #
    # Crucially, pairing is not free and an unpaired departure is not a failure.
    # A scene with one removal and one translation produces *two* departures,
    # and only one of them has a partner: the removal genuinely has none, and
    # leaving it as ``missing`` is the correct answer. An assignment that pairs
    # whenever any arrival happens to be in range will invent a translation for
    # every removal that shares a scene with one, which is what the benchmark
    # showed - two departures where the truth held one move. So each pair must
    # additionally *earn* its place, by explaining the evidence better than the
    # alternative reading of "a removal happened here and an insertion happened
    # there". That is the comparison in _pair_beats_alternatives below.
    candidates.sort(key=lambda c: c[0])
    used_dep: set[int] = set()
    used_arr: set[int] = set()
    matched: list[tuple[int, int, float]] = []
    for cost, sep, di, ai in candidates:
        if di in used_dep or ai in used_arr:
            continue
        if not _pair_beats_alternatives(departures[di], arrivals[ai], sep, cfg):
            continue
        used_dep.add(di)
        used_arr.add(ai)
        matched.append((di, ai, sep))

    merged: list[ChangeEvent] = []
    consumed: set[int] = set()

    for di, ai, _sep in matched:
        dep = departures[di]
        best = arrivals[ai]
        consumed.add(id(best))
        # Displacement is measured centre-to-centre, which is the object's own
        # travel. The box-edge separation used for pairing is not the same
        # quantity - it shrinks with the size of the two regions - so the two
        # are deliberately kept apart.
        dist = float(np.hypot(
            dep.centroid[0] - best.centroid[0], dep.centroid[1] - best.centroid[1]
        ))
        # Corroboration: both halves independently saw a clear direction of
        # change, and their magnitudes are comparable.
        g_dep = abs(float(dep.evidence.get("presence_gap", 0.0)))
        g_arr = abs(float(best.evidence.get("presence_gap", 0.0)))
        corroboration = float(np.clip(
            1.0 - abs(g_dep - g_arr) / max(1e-6, max(g_dep, g_arr)), 0.0, 1.0
        ))
        conf = float(np.clip(
            0.35 * corroboration
            + 0.30 * min(1.0, dist / (cfg.translation_min_px * 3))
            + 0.35 * min(g_dep, g_arr) / 0.25,
            0.0, 1.0,
        ))
        merged.append(
            ChangeEvent(
                event_id=dep.event_id,
                kind="moved",
                confidence=conf,
                regions=dep.regions + best.regions,
                bbox=dep.bbox,
                centroid=dep.centroid,
                displacement_px=dist,
                appearance_delta=dep.appearance_delta,
                summary=(
                    f"Object moved {dist:.0f}px: vacated its recorded position and "
                    f"reappeared at a new location"
                ),
                evidence={
                    "reason": (
                        "paired signature: a departure at the old location and a "
                        "similar-sized arrival nearby"
                    ),
                    "region_id": dep.evidence.get("region_id"),
                    "paired_region_id": best.evidence.get("region_id"),
                    "area_px": dep.evidence.get("area_px"),
                    "paired_area_px": best.evidence.get("area_px"),
                    "arrival_initial_kind": best.kind,
                    "arrival_bbox": [int(v) for v in best.bbox],
                    "displacement_px": round(dist, 2),
                    "corroboration": round(corroboration, 4),
                    "shape_similarity": round(_box_similarity(dep.bbox, best.bbox), 4),
                    "appearance_score": round(
                        float((_translation_corroboration(
                            base_gray, res_gray, dep.bbox, best.bbox
                        ) or {}).get("score", 0.0)), 4
                    ),
                    "departure_presence_gap": round(g_dep, 4),
                    "arrival_presence_gap": round(g_arr, 4),
                },
            )
        )

    out: list[ChangeEvent] = []
    for e in events:
        if id(e) in consumed:
            continue
        if any(m.event_id == e.event_id and m.kind == "moved" for m in merged):
            continue
        out.append(e)
    out.extend(merged)
    return out


def _translation_corroboration(
    base_gray: np.ndarray | None,
    res_gray: np.ndarray | None,
    dep_box: tuple[int, int, int, int],
    arr_box: tuple[int, int, int, int],
) -> dict | None:
    """Compare the object at a departure against the object at an arrival.

    A translation makes a specific, checkable prediction: the content standing
    at the departure in the baseline should be the content standing at the
    arrival in the rescan. The two patches are compared at *matched intrinsic
    size* - both resampled to a common grid before correlating - so the test
    asks whether the same thing is depicted rather than whether the same pixels
    line up.

    This value is reported as **evidence**, and is deliberately not used as a
    gate on the pairing. It was measured against the benchmark and does not
    discriminate: genuine translations scored 0.65-0.99 while unrelated
    removal/insertion pairs scored 0.54-0.87, and the ranges overlap enough that
    no threshold separates them. The reason is structural - resampling both
    patches to a common small grid leaves the scene's smooth backdrop as a
    shared component, so any two boxes in the same region correlate somewhat.
    Gating on a measurement that does not separate the classes would reject real
    translations while admitting false ones, which is worse than not testing.

    The pairing decision rests instead on the properties that *do* separate
    them: the two halves sit close together, and they must have opposite
    polarity - which is established upstream, by polarity-aware merging in the
    diff stage, and cannot happen for two unrelated events of the same sign.

    Returns ``None`` when the image data needed for the check is absent.
    """
    if base_gray is None or res_gray is None:
        return None
    a = _resample_patch(base_gray, dep_box)
    b = _resample_patch(res_gray, arr_box)
    if a is None or b is None:
        return None
    if a.std() < 1e-6 or b.std() < 1e-6:
        return {"score": 0.5, "correlation": 0.0}
    corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    if not np.isfinite(corr):
        return {"score": 0.5, "correlation": 0.0}
    return {"score": float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0)), "correlation": corr}


def _resample_patch(
    gray: np.ndarray, bbox: tuple[int, int, int, int], size: int = 48
) -> np.ndarray | None:
    """Crop a box and resample it to a fixed grid, for shape-independent matching."""
    x, y, w, h = bbox
    if w < 6 or h < 6:
        return None
    H, W = gray.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 - x0 < 6 or y1 - y0 < 6:
        return None
    patch = gray[y0:y1, x0:x1].astype(np.float32)
    patch = cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)
    return patch


def _pair_beats_alternatives(
    dep: ChangeEvent,
    arr: ChangeEvent,
    sep: float,
    cfg: EventConfig,
) -> bool:
    """Whether pairing two regions explains them better than leaving both alone.

    Every departure/arrival pair has a rival explanation: a removal happened at
    the departure, and an unrelated insertion happened at the arrival. That
    reading is always *available* - it invokes no relationship between the two -
    so a pairing rule that fires merely because an arrival is in range is not
    testing anything. It is asserting a relationship by default.

    The two readings differ in what they predict about the evidence:

    * Under **pairing**, one object left and arrived. Both regions are the same
      object, so they should be of comparable extent, and the arrival should be
      a substantial region - an object's worth of structure, not a fragment.
    * Under **removal plus insertion**, two unrelated things happened, and the
      fact that they are near each other and similarly sized is a coincidence.
      Coincidences of that kind are the thing this test is trying to rule out.

    So the pair is accepted when the arrival is a real region and the two agree
    in scale. This is not a claim that the evidence is decisive - it plainly is
    not, which is why the per-pair confidence is reported separately - only that
    the pairing is the better of the two explanations available.

    The thresholds are set from the *structural* requirement (an object's worth
    of arrival, comparable extent) rather than fitted to the benchmark, because
    a rule fitted to six scenes would not survive the seventh.
    """
    area_arr = float(arr.evidence.get("area_px", 0) or 0)
    area_dep = float(dep.evidence.get("area_px", 0) or 0)
    if area_arr <= 0 or area_dep <= 0:
        return False
    # The arrival has to be a real object-sized region. A pair whose arrival is
    # a scattering of pixels is a removal plus noise, not a translation.
    if area_arr < cfg.pair_arrival_area_px:
        return False
    # ...and the two have to agree in scale. Both readings are possible for a
    # mismatched pair, but only the removal-plus-insertion reading explains why
    # one region is many times the other's size.
    ratio = min(area_arr, area_dep) / max(area_arr, area_dep)
    if ratio < cfg.pair_scale_ratio_min:
        return False
    # Separation is already bounded by the caller; this is the last sanity check
    # that the pair is a plausible single displacement rather than two events
    # that merely happen to share a scene.
    return sep <= cfg.translation_max_px


def _size_compatible(dep: ChangeEvent, arr: ChangeEvent, ratio_min: float) -> bool:
    """Whether a departure and an arrival are plausibly the same object's area.

    This is a genuinely different test from :func:`_box_similarity`, and it is
    needed because the two boxes describe *different things*. An arrival box
    bounds the object itself. A departure box bounds a hole: the bare surface
    the object used to cover, which on a curved or textured surface is not the
    object's silhouette and can be substantially larger or smaller than it.

    So the guard has to be loose. A tight one would reject exactly the pairs
    that should be joined - a physically accurate departure box is routinely
    half again the area of the arrival - while a loose one still catches the
    case it exists for, which is a small region paired with a region an order of
    magnitude bigger, where no single object could be responsible for both.
    """
    da = float(dep.evidence.get("area_px", 0) or 0)
    aa = float(arr.evidence.get("area_px", 0) or 0)
    if da <= 0 or aa <= 0:
        return True
    ratio = min(da, aa) / max(da, aa)
    return ratio >= ratio_min


def _box_similarity(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> float:
    """How alike two boxes are in shape, in ``[0, 1]``.

    Aspect ratio is the useful signal here, not raw area: the same object seen
    at a different pose projects to nearly the same proportions, while its
    absolute size along the ground plane can shift by a factor of two. A box
    similarity of 1 means the two are the same shape, whatever their area.
    """
    aw, ah = max(1.0, float(a[2])), max(1.0, float(a[3]))
    bw, bh = max(1.0, float(b[2])), max(1.0, float(b[3]))
    ra, rb = aw / ah, bw / bh
    ratio = min(ra, rb) / max(ra, rb)
    return float(np.clip(ratio, 0.0, 1.0))


def _change_mask_for(r: ChangeRegion, shape: tuple[int, int]) -> np.ndarray | None:
    """Place a region's footprint back into a full-frame mask.

    The presence measure needs to know which pixels around the box are part of
    the change and which are untouched surface, so it can draw its reference
    from surface alone. The region stores that footprint within its own bounding
    box; this expands it to the frame so the measure can index it with ordinary
    absolute coordinates.
    """
    if r.footprint is None:
        return None
    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    x, y, bw, bh = r.bbox
    fh, fw = r.footprint.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + fw), min(h, y + fh)
    if x1 <= x0 or y1 <= y0:
        return mask
    mask[y0:y1, x0:x1] = r.footprint[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.uint8) * 255
    return mask


def _classify(
    r: ChangeRegion,
    cfg: EventConfig,
    base_gray: np.ndarray | None,
    res_gray: np.ndarray | None,
    motion_tolerance_px: float,
) -> tuple[EventKind, float, float, float, str, dict]:
    """Decide the event kind for one region.

    Returns ``(kind, confidence, displacement_px, appearance_delta, reason, extra)``.
    """
    if r.low_confidence:
        return (
            "uncertain", min(r.confidence, 0.35), 0.0, 0.0,
            "registration residual too high to assert a change kind",
            {"residual_px": round(r.registration_residual, 3)},
        )

    # Without the images we can only report that something changed here.
    if base_gray is None or res_gray is None:
        return (
            "state_changed", r.confidence, 0.0, 0.0,
            "appearance differs at this location",
            {},
        )

    b_pre = _presence(
        base_gray, r.bbox,
        change_mask=_change_mask_for(r, base_gray.shape[:2]),
    )
    s_pre = _presence(
        res_gray, r.bbox,
        change_mask=_change_mask_for(r, res_gray.shape[:2]),
    )
    b_score, s_score = b_pre["presence"], s_pre["presence"]
    gap = s_score - b_score  # positive: structure *arrived*; negative: it *left*

    extra = {
        "baseline_presence": round(b_score, 4),
        "rescan_presence": round(s_score, 4),
        "presence_gap": round(gap, 4),
        # Retained under their historical keys so downstream consumers keep working.
        "baseline_objectness": round(b_score, 4),
        "rescan_objectness": round(s_score, 4),
        "baseline_inside": b_pre["inside"],
        "baseline_surround": b_pre["surround"],
        "rescan_inside": s_pre["inside"],
        "rescan_surround": s_pre["surround"],
    }

    # --- Direction of the change -------------------------------------------
    #
    # Presence is comparative: a box that looks like its own surround sits at
    # 0.5. In a scene whose background is textured, a *discrete object* reads
    # BELOW 0.5 (its interior is smoother and more uniform than the busy
    # background around it), and bare background reads near 0.5. The sign of
    # the gap therefore carries the direction of the change:
    #
    #   gap > 0  structure *arrived*   (baseline object-ish -> rescan background-ish)
    #   gap < 0  structure *departed*  (baseline background-ish -> rescan object-ish)
    #
    # What this sign cannot do on its own is separate a removal from a
    # translation, because both leave a departure at the old location. That
    # distinction is made structurally, by the caller, from the *paired*
    # signature: a translation also deposits an arrival nearby. Here we type
    # the single-region case, where the sign is the whole story.
    departed = gap <= -cfg.presence_gap_min
    arrived = gap >= cfg.presence_gap_min

    if departed:
        conf = float(np.clip(0.35 * r.confidence + 0.65 * _gap_term(-gap), 0.0, 1.0))
        return (
            "missing", conf, 0.0, float(abs(gap) * 255.0),
            "background is exposed here in the rescan where the baseline held an "
            "object, and nothing comparable appeared nearby",
            extra,
        )

    if arrived:
        conf = float(np.clip(0.35 * r.confidence + 0.65 * _gap_term(gap), 0.0, 1.0))
        return (
            "added", conf, 0.0, float(abs(gap) * 255.0),
            "an object occupies this location in the rescan that was background "
            "in the baseline",
            extra,
        )

    # --- No clear direction: separate a translation from an in-place change --
    #
    # The presence gap is small, which rules out a simple appearance or
    # disappearance. Two possibilities remain - the object translated, or it
    # changed in place - and the test is whether the baseline interior explains
    # the rescan *better somewhere else* than it explains it *in place*.
    disp, shifted_score = _best_shift(
        base_gray, res_gray, r.bbox, search=int(motion_tolerance_px * 3)
    )
    in_place = _in_place_match(base_gray, res_gray, r.bbox)
    delta = _appearance_difference(base_gray, res_gray, r.bbox)
    advantage = shifted_score - in_place
    extra["shift_score"] = round(shifted_score, 4)
    extra["in_place_score"] = round(in_place, 4)
    extra["shift_advantage"] = round(advantage, 4)
    extra["appearance_delta"] = round(delta, 3)

    # A genuine translation needs a real displacement whose shifted match is
    # clearly better than the in-place match. Requiring both is what stops an
    # in-place change being reported as "moved" on the strength of a
    # coincidental correlation somewhere in the search window.
    if (
        disp > motion_tolerance_px
        and shifted_score >= cfg.shift_match_min
        and advantage >= cfg.shift_advantage_min
    ):
        conf = float(np.clip(
            0.30 * r.confidence
            + 0.30 * min(1.0, disp / (motion_tolerance_px * 3))
            + 0.25 * shifted_score
            + 0.15 * _gap_term(advantage),
            0.0, 1.0,
        ))
        return (
            "moved", conf, disp, delta,
            f"baseline content matches the rescan {disp:.0f}px from its recorded "
            f"position, better than in place ({in_place:.2f} -> {shifted_score:.2f})",
            extra,
        )

    # --- In-place change ---------------------------------------------------
    # The object is where it was, but it does not look the same. Confidence
    # tracks how far the appearance difference rises above what a re-visit
    # alone would produce; a weak difference is reported honestly as uncertain
    # rather than inflated into a definite claim.
    delta_term = float(np.clip(delta / 55.0, 0.0, 1.0))
    conf = float(np.clip(0.40 * r.confidence + 0.60 * delta_term, 0.0, 1.0))
    if delta_term < 0.12:
        return (
            "uncertain", min(conf, 0.35), disp, delta,
            "structure present in both captures at this location and no clear "
            "direction of change; appearance difference is within the re-visit "
            "floor",
            extra,
        )
    return (
        "state_changed", conf, disp, delta,
        "structure present in both captures at this location; appearance "
        "differs beyond the re-visit floor",
        extra,
    )


def _gap_term(gap: float) -> float:
    """Map a presence gap into a 0-1 confidence contribution."""
    return float(np.clip(gap / 0.30, 0.0, 1.0))


def _in_place_match(
    base_gray: np.ndarray, res_gray: np.ndarray, bbox: tuple[int, int, int, int]
) -> float:
    """Normalised correlation of the baseline box against the rescan *in place*.

    This is the baseline the shifted match must beat. Because both patches are
    taken at identical coordinates, any residual misregistration depresses this
    value, which is why the moved/state_changed decision uses the *difference*
    between the two scores rather than either one alone.
    """
    x, y, w, h = bbox
    H, W = base_gray.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return 0.0
    a = base_gray[y0:y1, x0:x1].astype(np.float32)
    b = res_gray[y0:y1, x0:x1].astype(np.float32)
    if a.std() < 1e-6 or b.std() < 1e-6:
        return 0.0
    return float(np.clip(np.corrcoef(a.ravel(), b.ravel())[0, 1], -1.0, 1.0))


def _structure_gap(high: float, low: float) -> float:
    if high <= 0:
        return 0.0
    return float(np.clip(1.0 - (low / max(1e-6, high)), 0.0, 1.0))


def _best_shift(
    base_gray: np.ndarray, res_gray: np.ndarray, bbox: tuple[int, int, int, int], search: int
) -> tuple[float, float]:
    """Template-match the baseline patch within +/- ``search`` px of its place.

    Returns ``(displacement_px, normalised_correlation)``. A high correlation at
    a non-zero offset means the object moved rather than changed in place.
    """
    x, y, w, h = bbox
    H, W = base_gray.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return 0.0, 0.0
    tmpl = base_gray[y0:y1, x0:x1]

    s = int(round(search))
    sx0, sy0 = max(0, x0 - s), max(0, y0 - s)
    sx1, sy1 = min(W, x1 + s), min(H, y1 + s)
    window = res_gray[sy0:sy1, sx0:sx1]
    if window.shape[0] < tmpl.shape[0] or window.shape[1] < tmpl.shape[1]:
        return 0.0, 0.0

    try:
        res = cv2.matchTemplate(window, tmpl, cv2.TM_CCOEFF_NORMED)
    except cv2.error:
        return 0.0, 0.0
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    found_x, found_y = sx0 + max_loc[0], sy0 + max_loc[1]
    disp = float(np.hypot(found_x - x0, found_y - y0))
    return disp, float(max_val)


def _appearance_difference(
    base_gray: np.ndarray, res_gray: np.ndarray, bbox: tuple[int, int, int, int]
) -> float:
    """Mean absolute intensity difference inside a box, 0-255 scale."""
    x, y, w, h = bbox
    H, W = base_gray.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return 0.0
    a = base_gray[y0:y1, x0:x1].astype(np.float32)
    b = res_gray[y0:y1, x0:x1].astype(np.float32)
    return float(np.mean(np.abs(a - b)))


def _summarise(kind: EventKind, r: ChangeRegion, disp: float, delta: float) -> str:
    area = r.area_px
    if kind == "missing":
        return f"Object removed: {area}px of structure present in the baseline is gone"
    if kind == "added":
        return f"Object added: {area}px of new structure present in the rescan"
    if kind == "moved":
        return f"Object moved {disp:.0f}px from its recorded position"
    if kind == "state_changed":
        return f"Object changed in place (mean intensity delta {delta:.1f})"
    return f"Uncertain change over {area}px (registration residual {r.registration_residual:.1f}px)"


def event_totals(events: list[ChangeEvent]) -> dict[str, int]:
    """Count events by kind - the headline number a human reads first."""
    totals: dict[str, int] = {
        k: 0 for k in ("added", "missing", "moved", "state_changed", "uncertain")
    }
    for e in events:
        totals[e.kind] = totals.get(e.kind, 0) + 1
    totals["total"] = len(events)
    return totals


def render_events(
    canvas: np.ndarray, events: list[ChangeEvent], *, min_confidence: float = 0.0
) -> np.ndarray:
    """Annotate a frame with typed event boxes and their labels."""
    out = canvas.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    palette = {
        "added": (80, 220, 100),
        "missing": (90, 90, 240),
        "moved": (255, 190, 60),
        "state_changed": (200, 120, 255),
        "uncertain": (170, 170, 170),
    }
    for ev in events:
        if ev.confidence < min_confidence:
            continue
        colour = palette.get(ev.kind, (200, 200, 200))
        x, y, w, h = ev.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), colour, 2)
        label = f"{ev.kind} {ev.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x, max(0, y - th - 8)), (x + tw + 8, y), colour, -1)
        cv2.putText(
            out, label, (x + 4, max(13, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25, 25, 25), 1, cv2.LINE_AA,
        )
    return out
