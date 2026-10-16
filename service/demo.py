"""Demo sites: real pipeline input that needs no upload.

The live app has to be usable the moment it loads, so the service ships with
sites whose frames are generated deterministically. Each one is a genuine
pipeline input - the same code path as an uploaded capture - so what a visitor
sees is real system output, not a canned screenshot.

The sites are also the honest failure cases. ``known_hard`` marks a site whose
scene is deliberately difficult and whose detection is known to be imperfect;
the UI labels it as such. A demo that only ever shows successes is a demo that
misrepresents the system, and a reviewer will find the weak spot anyway - far
better to point at it first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from strata_core.synth import SyntheticScene, ground_truth_in_canonical, make_scene

# Human names for the mutation kinds, in the language of the people who would
# actually use this: someone checking a store room, not someone reading a paper.
_KIND_PHRASE = {
    "remove": "an item was taken away",
    "add": "an item appeared",
    "move": "an item was moved",
    "recolour": "an item changed colour",
}


@dataclass(frozen=True)
class DemoSite:
    """A named, reproducible scene the operator can scan."""

    site_id: str
    name: str
    description: str
    seed: int
    n_objects: int = 18
    n_mutations: int = 4
    pose_strength: float = 1.0
    noise_sigma: float = 4.0
    lighting_shift: float = 12.0
    known_hard: bool = False
    tags: tuple[str, ...] = field(default_factory=tuple)

    def build(self) -> SyntheticScene:
        """Generate the scene. Deterministic for a given seed."""
        return make_scene(
            seed=self.seed,
            n_objects=self.n_objects,
            n_mutations=self.n_mutations,
            pose_strength=self.pose_strength,
            noise_sigma=self.noise_sigma,
            lighting_shift=self.lighting_shift,
        )

    def change_summary(self) -> list[str]:
        """What actually changed, phrased for a human, from the ground truth."""
        scene = self.build()
        out: list[str] = []
        for m in scene.mutations:
            phrase = _KIND_PHRASE.get(m.kind, f"change ({m.kind})")
            out.append(f"{m.object_id}: {phrase}")
        return out


DEMO_SITES: tuple[DemoSite, ...] = (
    DemoSite(
        site_id="bay-4",
        name="Loading Bay 4",
        description=(
            "A pallet racking bay photographed from the aisle. Two items were "
            "picked and one was relocated to the far end of the shelf between "
            "visits."
        ),
        seed=1021,
        tags=("racking", "picking"),
    ),
    DemoSite(
        site_id="aisle-2",
        name="Cold Store Aisle 2",
        description=(
            "Chilled storage, re-photographed a week later under different "
            "lighting. The exposure drift between visits is real, not simulated "
            "away."
        ),
        seed=1042,
        lighting_shift=18.0,
        tags=("chilled", "lighting drift"),
    ),
    DemoSite(
        site_id="bench-c",
        name="Assembly Bench C",
        description=(
            "A workbench with hand tools. Useful for checking the system against "
            "a scene where objects are small relative to the frame."
        ),
        seed=1105,
        n_objects=22,
        tags=("small objects",),
    ),
    DemoSite(
        site_id="yard-north",
        name="Yard North",
        description=(
            "An outdoor storage yard with a wider baseline-to-rescan viewpoint "
            "change than the indoor sites. Registered with a homography because "
            "the scene is close to planar."
        ),
        seed=1140,
        pose_strength=1.6,
        noise_sigma=6.0,
        tags=("outdoor", "wide baseline"),
    ),
    DemoSite(
        site_id="archive-row-7",
        name="Archive Row 7",
        description=(
            "Shelved archive boxes where several changes sit close together. "
            "Included as a difficult case: adjacent changes can fall into one "
            "connected region, so the system sometimes reports fewer events than "
            "were made. The true changes are shown alongside the result."
        ),
        seed=1056,
        n_mutations=5,
        known_hard=True,
        tags=("cluttered", "known limitation"),
    ),
)

_BY_ID = {s.site_id: s for s in DEMO_SITES}


def demo_site_ids() -> list[str]:
    return [s.site_id for s in DEMO_SITES]


def get_demo_site(site_id: str) -> DemoSite | None:
    return _BY_ID.get(site_id)


def ground_truth_payload(site: DemoSite) -> list[dict[str, Any]]:
    """The true changes, in the same shape and the same units as a detection.

    The boxes come back in **canonical pixels**, converted by the core rather
    than here. That conversion is load-bearing: ground truth is recorded in the
    scene's own pixels and detections are reported in canonical pixels, so
    serving the raw boxes put the answer key in a different coordinate system
    from the result it was meant to check - and the panel then reported that a
    system which had found every change had found none.

    Destinations of a translation are excluded, matching how the system is
    scored. A translation is one change and is reported as one event, so
    enumerating its destination as a separate true change would make the answer
    key look like it has a change the system never had a chance to report.
    """
    scene = site.build()
    entries = ground_truth_in_canonical(scene)
    return [
        {
            "kind": entry["kind"],
            "object_id": entry["object_id"],
            "bbox": [round(float(v), 1) for v in entry["bbox"]],
            "bbox_space": entry["bbox_space"],
            "mutation": entry["mutation"],
        }
        for entry in entries
    ]


def thumbnail(scene: SyntheticScene, long_edge: int = 320) -> np.ndarray:
    """A small baseline view for the site picker."""
    import cv2

    img = scene.baseline
    h, w = img.shape[:2]
    scale = long_edge / float(max(h, w))
    if scale < 1.0:
        img = cv2.resize(
            img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA
        )
    return img
