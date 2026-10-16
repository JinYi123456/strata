"""Synthetic scene generator with ground truth, for tests and evaluation.

Real evaluation needs known answers. This module renders a structured scene -
a shelf-like backdrop with discrete object tiles - then produces a second view
of the same scene from a *different camera pose* with a **known homography**,
optionally after applying a set of known mutations.

That gives the pipeline a test with a ground-truth answer for three things at
once:

* **registration accuracy** - how close the recovered homography is to the
  known one, in pixels;
* **change detection accuracy** - precision/recall of the reported change
  regions against the mutated object boxes;
* **event typing accuracy** - whether a removal was reported as ``missing``, an
  insertion as ``added``, a translation as ``moved``.

The generator is deliberately configurable and seeded, so a benchmark run is
reproducible down to the pixel.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

import cv2
import numpy as np

from .config import CANONICAL_LONG_EDGE

MutationKind = Literal["remove", "add", "move", "recolour"]


@dataclass
class ObjectTile:
    """One discrete object placed in the scene, in scene coordinates.

    ``render_seed`` is the object's *stable identity*: its internal appearance
    (label marks, band, stripe) is derived from it and from nothing else. Two
    placements of the same object render identically, and one object's presence
    or absence never perturbs another's appearance. Without this, the two
    visits would differ on objects that never changed.
    """

    object_id: str
    x: int
    y: int
    w: int
    h: int
    colour: tuple[int, int, int]
    render_seed: int = 0

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    @property
    def centroid(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)


@dataclass
class Mutation:
    """A known change applied to produce the rescan view."""

    kind: MutationKind
    object_id: str
    dx: int = 0
    dy: int = 0
    new_colour: tuple[int, int, int] | None = None

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "object_id": self.object_id,
            "dx": int(self.dx),
            "dy": int(self.dy),
            "new_colour": list(self.new_colour) if self.new_colour else None,
        }


@dataclass
class SyntheticScene:
    """A generated scene with its ground truth."""

    baseline: np.ndarray                    # (H, W, 3) BGR, canonical-ish view
    rescan: np.ndarray                      # (H, W, 3) BGR, different pose
    homography: np.ndarray                  # true H mapping rescan -> baseline
    baseline_objects: list[ObjectTile]
    rescan_objects: list[ObjectTile]
    mutations: list[Mutation]
    seed: int
    size: tuple[int, int] = (960, 720)      # (w, h)

    def ground_truth_events(self) -> list[dict]:
        """The expected event set, one dict per mutation.

        Every mutation produces at least one expected change *location*, and a
        translation produces **two** - the position the object vacated and the
        position it arrived at. Both are genuine changes on the ground, and a
        detector that reports them is correct, so both are enumerated here.
        Scoring only the origin would count a correctly detected destination as
        a false positive, which is how a detector can be penalised for being
        right.

        Insertions have no baseline location, so the expected box is the
        object's position in the *rescan*.
        """
        out = []
        for m in self.mutations:
            if m.kind == "add":
                dst = next((o for o in self.rescan_objects if o.object_id == m.object_id), None)
                if dst is None:
                    continue
                out.append({
                    "kind": "added",
                    "object_id": m.object_id,
                    "baseline_bbox": list(dst.bbox),
                    "bbox_space": "rescan",
                    "mutation": m.to_dict(),
                })
                continue

            src = next((o for o in self.baseline_objects if o.object_id == m.object_id), None)
            if src is None:
                continue
            out.append({
                "kind": {"remove": "missing", "add": "added",
                         "move": "moved", "recolour": "state_changed"}[m.kind],
                "object_id": m.object_id,
                "baseline_bbox": list(src.bbox),
                "bbox_space": "baseline",
                "mutation": m.to_dict(),
            })
            if m.kind == "move":
                dst = next((o for o in self.rescan_objects if o.object_id == m.object_id), None)
                if dst is not None:
                    # The vacated-then-reoccupied pair: the object's new home is
                    # a change in its own right.
                    out.append({
                        "kind": "moved_destination",
                        "object_id": m.object_id,
                        "baseline_bbox": list(dst.bbox),
                        "bbox_space": "rescan",
                        "mutation": m.to_dict(),
                    })
        return out

    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / "baseline.png"), self.baseline)
        cv2.imwrite(str(d / "rescan.png"), self.rescan)
        meta = {
            "seed": self.seed,
            "size": list(self.size),
            "homography": self.homography.tolist(),
            "baseline_objects": [
                {"object_id": o.object_id, "bbox": list(o.bbox),
                 "colour": list(o.colour)} for o in self.baseline_objects
            ],
            "rescan_objects": [
                {"object_id": o.object_id, "bbox": list(o.bbox),
                 "colour": list(o.colour)} for o in self.rescan_objects
            ],
            "mutations": [m.to_dict() for m in self.mutations],
            "ground_truth_events": self.ground_truth_events(),
        }
        (d / "ground_truth.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return d


def _random_colour(
    rng: np.random.Generator, dim: bool = False, *, ref: tuple[int, int, int] | None = None
) -> tuple[int, int, int]:
    """A random object colour, optionally far enough from ``ref`` to be visible.

    When ``ref`` is given the colour is resampled until it differs from it by a
    margin in at least one channel. A "recolour" mutation that lands on a
    near-identical colour is not a change anyone could observe, and a benchmark
    built from one would score a correct detector as having missed something
    that never happened.
    """
    for _ in range(24):
        if dim:
            v = int(rng.integers(70, 170))
            colour = (v, v, v)
        else:
            colour = (
                int(rng.integers(60, 255)),
                int(rng.integers(60, 255)),
                int(rng.integers(60, 255)),
            )
        if ref is None:
            return colour
        if max(abs(colour[i] - ref[i]) for i in range(3)) >= 70:
            return colour
    return colour


def _draw_backdrop(canvas: np.ndarray, rng: np.random.Generator) -> None:
    """A smooth, well-lit surface - the thing objects sit *on*.

    Real interiors - a shelf back, a counter, a wall, a fridge interior - are
    predominantly *smooth*, with a gentle lighting gradient and at most a few
    soft seams. That is the property the change detector depends on: it must be
    able to tell "an object is here" from "bare surface is here", and it can
    only do that if bare surface actually looks like bare surface.

    An earlier revision drew high-contrast banding and hard shadow lines across
    the whole surface, which made every patch of background statistically
    indistinguishable from an object. The result was a benchmark that no
    classifier could satisfy. Background structure is now low-amplitude and
    low-frequency, as it is in real imagery.
    """
    h, w = canvas.shape[:2]
    base = 132.0
    canvas[:] = int(base)

    # Gentle vertical lighting falloff: brighter where a light source would be,
    # never more than a few percent of the range.
    vert = np.linspace(18.0, -22.0, h, dtype=np.float32)
    canvas[:] = np.clip(
        canvas.astype(np.float32) + vert[:, None, None], 0, 255
    ).astype(np.uint8)

    # A faint horizontal gradient, plus very low-amplitude grain so the surface
    # is not perfectly synthetic (real sensors see texture everywhere).
    horiz = np.linspace(-10.0, 10.0, w, dtype=np.float32)
    img = canvas.astype(np.float32) + horiz[None, :, None]
    grain = rng.normal(0.0, 2.0, (h, w, 1)).astype(np.float32)
    canvas[:] = np.clip(img + grain, 0, 255).astype(np.uint8)

    # One or two soft structural seams, at low contrast - a shelf divider seen
    # out of focus, say. Visible to a human, subtle to a threshold.
    for frac in (0.34, 0.67):
        x = int(w * frac)
        cv2.line(canvas, (x, 0), (x, h), (base - 34, base - 34, base - 32), 1)


def _object_patch(
    colour: tuple[int, int, int], w: int, h: int, *, seed: int
) -> np.ndarray:
    """Render one object tile, with internal detail, at its final size.

    A real object is not a flat rectangle of colour: it has a label, an edge
    highlight, a band, a cap, a barcode. Those internal features are what make
    it distinct from the surface it rests on, and they are what the presence
    measure keys on. A flat rectangle on a smooth surface is genuinely
    ambiguous; a detailed object is not.

    Detail is drawn at the tile's *final* size rather than at a fixed working
    size, because downscaling a 96x96 pattern to a 50x50 tile averages the
    internal features away - which would defeat the entire point.

    The randomness is seeded **per object**, not taken from a shared stream.
    That is essential: if tiles drew from one running generator, then removing
    or inserting an object would shift every later tile's internal detail, and
    the two visits would differ on objects that never moved. The generator
    would then be producing a scene that contradicts its own ground truth, and
    any detector tested against it would be penalised for being right.
    """
    rng = np.random.default_rng(seed)
    tile = np.zeros((h, w, 3), np.uint8)
    tile[:] = colour

    b, g, r = int(colour[0]), int(colour[1]), int(colour[2])
    dark = (max(0, b - 80), max(0, g - 80), max(0, r - 80))
    light = (min(255, b + 60), min(255, g + 60), min(255, r + 60))
    t = max(2, min(w, h) // 24)  # stroke width holds up at small tile sizes

    # Bevel: a bright top-left edge and a dark bottom-right, giving the tile a
    # 3-D read and a strong closed contour.
    cv2.rectangle(tile, (t, t), (w - 1 - t, h - 1 - t), light, t)
    cv2.line(tile, (t, h - 1 - t), (w - 1 - t, h - 1 - t), dark, t)
    cv2.line(tile, (w - 1 - t, t), (w - 1 - t, h - 1 - t), dark, t)

    # An internal label block: a contrasting band with glyph-like marks. Sized
    # as a fraction of the tile so it reads at any resolution.
    band_y = int(h * rng.integers(28, 44) / 100)
    band_h = max(3, int(h * rng.integers(18, 26) / 100))
    band_y = min(band_y, h - band_h - t - 2)
    x0, x1 = int(w * 0.12), int(w * 0.88)
    if x1 - x0 > 4:
        cv2.rectangle(tile, (x0, band_y), (x1, band_y + band_h), (245, 245, 245), -1)
        n_marks = int(rng.integers(3, 6))
        mark_w = max(2, int((x1 - x0) * 0.09))
        for k in range(n_marks):
            mx = int(x0 + (x1 - x0) * (k + 0.5) / n_marks) - mark_w // 2
            mh = max(2, int(band_h * rng.integers(50, 85) / 100))
            my = band_y + (band_h - mh) // 2
            cv2.rectangle(tile, (mx, my), (mx + mark_w, my + mh), dark, -1)

    # A second, smaller element lower down - a cap, a stripe, a seam.
    if rng.random() < 0.75:
        sy = int(h * rng.integers(62, 78) / 100)
        sh = max(2, int(h * rng.integers(7, 13) / 100))
        sy = min(sy, h - sh - t - 2)
        sx0, sx1 = int(w * 0.20), int(w * 0.80)
        if sx1 - sx0 > 4:
            cv2.rectangle(tile, (sx0, sy), (sx1, sy + sh), dark, -1)

    return tile


def _place_objects(
    n: int, size: tuple[int, int], rng: np.random.Generator, avoid: list[ObjectTile] | None = None
) -> list[ObjectTile]:
    w, h = size
    avoid = avoid or []
    tiles: list[ObjectTile] = []
    for i in range(n):
        for _ in range(200):
            tw = int(rng.integers(46, 96))
            th = int(rng.integers(46, 96))
            x = int(rng.integers(10, max(11, w - tw - 10)))
            y = int(rng.integers(10, max(11, h - th - 10)))
            cand = ObjectTile(
                f"obj{i+1:02d}", x, y, tw, th, _random_colour(rng),
                render_seed=int(rng.integers(1, 2**31 - 1)),
            )
            if any(_overlaps(cand, o, pad=12) for o in tiles + avoid):
                continue
            tiles.append(cand)
            break
    return tiles


def _overlaps(a: ObjectTile, b: ObjectTile, pad: int = 0) -> bool:
    ax1, ay1, ax2, ay2 = a.x - pad, a.y - pad, a.x + a.w + pad, a.y + a.h + pad
    bx1, by1, bx2, by2 = b.x - pad, b.y - pad, b.x + b.w + pad, b.y + b.h + pad
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)


def _render(objects: Iterable[ObjectTile], size: tuple[int, int]) -> np.ndarray:
    w, h = size
    img = np.zeros((h, w, 3), np.uint8)
    rng = np.random.default_rng(11)
    _draw_backdrop(img, rng)
    for o in objects:
        patch = _object_patch(o.colour, o.w, o.h, seed=o.render_seed)
        # Guard the write to the frame. A camera re-visit is a projective map,
        # and under a projective map the scene is not a rectangle - places near
        # the frame edge can be pushed outside it. numpy clips a slice silently,
        # which would spread the tile over the wrong number of pixels and make
        # the two visits differ on an object that never changed. Clipping the
        # patch to the visible region keeps every tile the same object in both
        # renders.
        x0, y0 = max(0, o.x), max(0, o.y)
        x1, y1 = min(w, o.x + o.w), min(h, o.y + o.h)
        if x1 <= x0 or y1 <= y0:
            continue
        img[y0:y1, x0:x1] = patch[y0 - o.y:y1 - o.y, x0 - o.x:x1 - o.x]
    return img


def _camera_homography(
    rng: np.random.Generator, size: tuple[int, int], strength: float
) -> np.ndarray:
    """A plausible hand-held re-visit: small similarity + mild projective tilt."""
    w, h = size
    scale = 1.0 + rng.normal(0, 0.030 * strength)
    angle = rng.normal(0, 2.2 * strength) * np.pi / 180.0
    tx = rng.normal(0, 26 * strength)
    ty = rng.normal(0, 20 * strength)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    cx, cy = w / 2.0, h / 2.0
    S = np.array([
        [scale * cos_a, -scale * sin_a, cx - scale * cos_a * cx + scale * sin_a * cy + tx],
        [scale * sin_a, scale * cos_a, cy - scale * sin_a * cx - scale * cos_a * cy + ty],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    # Mild projective component (out-of-plane tilt).
    P = np.eye(3, dtype=np.float64)
    P[2, 0] = rng.normal(0, 1.1e-4 * strength)
    P[2, 1] = rng.normal(0, 1.1e-4 * strength)
    return P @ S


def canonical_size_for_scene(
    scene: "SyntheticScene", long_edge: int = CANONICAL_LONG_EDGE
) -> tuple[int, int]:
    """The canonical frame a scan of this scene produces, as ``(width, height)``."""
    w, h = scene.size
    scale = long_edge / float(max(w, h))
    return int(round(w * scale)), int(round(h * scale))


def baseline_box_to_canonical(
    bbox: Iterable[float],
    src_size: tuple[int, int],
    canonical_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    """Scale a box from baseline-frame pixels into canonical pixels.

    Ground truth is expressed in the scene's own coordinates because that is
    where the mutations were applied; detections are reported in canonical
    coordinates because that is where the comparison happens. Comparing one
    against the other without converting compares two different units and yields
    a score near zero no matter how good the detector is.
    """
    x, y, w, h = (float(v) for v in bbox)
    sw, sh = src_size
    cw, ch = canonical_size
    sx, sy = cw / float(sw), ch / float(sh)
    return (x * sx, y * sy, w * sx, h * sy)


def ground_truth_in_canonical(
    scene: "SyntheticScene",
    *,
    long_edge: int = CANONICAL_LONG_EDGE,
    include_destinations: bool = False,
) -> list[dict]:
    """Every true change, with its box expressed in canonical pixels.

    Lives in the core rather than beside any one consumer, because two callers
    need it - the evaluation harness and the service's answer-key panel - and the
    first version of the panel simply forgot to convert. Both now share this, so
    they cannot disagree about where a change was.

    Each entry keeps the space its box was recorded in, since that is the reason
    the conversion is needed at all:

    * most changes are recorded where the *baseline* saw them;
    * an insertion has no baseline presence, so its only location is in the
      *rescan*;
    * a translation's destination is a real change in its own right, but this
      system reports a translation as one event, so it is excluded by default.
      Keeping it would ask a single event to satisfy two expected boxes and would
      cap recall below 1.0 by construction. ``scripts/oracle.py`` measures that
      cap.
    """
    canonical = canonical_size_for_scene(scene, long_edge)
    baseline_size = (int(scene.size[0]), int(scene.size[1]))

    out: list[dict] = []
    for entry in scene.ground_truth_events():
        kind = entry["kind"]
        if kind == "moved_destination" and not include_destinations:
            continue
        space = entry.get("bbox_space", "baseline")
        # Canonical space is the baseline's frame scaled up, and both captures
        # are rendered at the same size, so one uniform scale covers both spaces.
        box = baseline_box_to_canonical(entry["baseline_bbox"], baseline_size, canonical)
        out.append(
            {
                "kind": kind,
                "object_id": entry["object_id"],
                "bbox": [float(v) for v in box],
                "bbox_space": space,
                "mutation": entry.get("mutation", {}),
            }
        )
    return out


def make_scene(
    *,
    seed: int = 20261026,
    size: tuple[int, int] = (960, 720),
    n_objects: int = 18,
    n_mutations: int = 4,
    pose_strength: float = 1.0,
    noise_sigma: float = 4.0,
    lighting_shift: float = 12.0,
) -> SyntheticScene:
    """Generate a paired baseline/rescan with known changes.

    Args:
        seed: RNG seed; the same seed reproduces the scene exactly.
        size: ``(width, height)`` of the rendered scene.
        n_objects: discrete tiles placed in the baseline.
        n_mutations: how many known changes to apply.
        pose_strength: scales the camera-pose difference between views.
        noise_sigma: sensor noise standard deviation added to both views.
        lighting_shift: global brightness shift applied to the rescan, to test
            that photometric normalisation absorbs exposure drift.
    """
    rng = np.random.default_rng(seed)
    baseline_objects = _place_objects(n_objects, size, rng)
    # The rescan starts from the *same* objects with the *same* identities -
    # only the mutations below are allowed to differ.
    rescan_objects = [
        ObjectTile(o.object_id, o.x, o.y, o.w, o.h, o.colour, o.render_seed)
        for o in baseline_objects
    ]

    kinds: list[MutationKind] = ["remove", "add", "move", "recolour"]
    mutations: list[Mutation] = []
    w, h = size

    for i in range(n_mutations):
        kind = kinds[i % len(kinds)]
        if kind == "add":
            new_tiles = _place_objects(1, size, rng, avoid=rescan_objects)
            if new_tiles:
                t = new_tiles[0]
                t.object_id = f"new{i+1:02d}"
                rescan_objects.append(t)
                mutations.append(Mutation("add", t.object_id))
            continue

        # Pick a target that still exists and has not already been mutated.
        used = {m.object_id for m in mutations}
        candidates = [o for o in rescan_objects if o.object_id in {b.object_id for b in baseline_objects}
                      and o.object_id not in used]
        if not candidates:
            continue
        target = candidates[int(rng.integers(0, len(candidates)))]

        if kind == "remove":
            rescan_objects = [o for o in rescan_objects if o.object_id != target.object_id]
            mutations.append(Mutation("remove", target.object_id))
        elif kind == "move":
            dx = int(rng.choice([-1, 1]) * rng.integers(70, 150))
            dy = int(rng.choice([-1, 1]) * rng.integers(30, 90))
            target.x = int(np.clip(target.x + dx, 4, w - target.w - 4))
            target.y = int(np.clip(target.y + dy, 4, h - target.h - 4))
            mutations.append(Mutation("move", target.object_id, dx=dx, dy=dy))
        elif kind == "recolour":
            new_colour = _random_colour(rng, ref=target.colour)
            target.colour = new_colour
            mutations.append(Mutation("recolour", target.object_id, new_colour=new_colour))

    baseline = _render(baseline_objects, size)
    rescan = _render(rescan_objects, size)

    # ``H`` maps scene (baseline-frame) coordinates into the rescan's sensor
    # frame, so applying it produces the rescan view. The ground-truth matrix
    # exposed on the scene is its inverse - rescan -> baseline - which is the
    # direction the pipeline recovers and therefore the only comparable one.
    H_scene_to_rescan = _camera_homography(rng, size, pose_strength)
    rescan = cv2.warpPerspective(
        rescan, H_scene_to_rescan, size, flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    H_true = np.linalg.inv(H_scene_to_rescan)

    if noise_sigma > 0:
        baseline = np.clip(
            baseline.astype(np.float32) + rng.normal(0, noise_sigma, baseline.shape), 0, 255
        ).astype(np.uint8)
        rescan = np.clip(
            rescan.astype(np.float32) + rng.normal(0, noise_sigma, rescan.shape), 0, 255
        ).astype(np.uint8)

    if lighting_shift:
        rescan = np.clip(rescan.astype(np.float32) + lighting_shift, 0, 255).astype(np.uint8)

    return SyntheticScene(
        baseline=baseline,
        rescan=rescan,
        homography=H_true,
        baseline_objects=baseline_objects,
        rescan_objects=rescan_objects,
        mutations=mutations,
        seed=seed,
        size=size,
    )


def make_still_pair(
    *, seed: int = 1, size: tuple[int, int] = (960, 720), strength: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A pair with **no** mutations - used to measure false-positive rate."""
    s = make_scene(seed=seed, size=size, n_objects=18, n_mutations=0,
                   pose_strength=strength, noise_sigma=3.0, lighting_shift=10.0)
    return s.baseline, s.rescan, s.homography
