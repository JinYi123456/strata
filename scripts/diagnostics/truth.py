"""Ground truth in canonical pixels, shared by the diagnostic sweeps.

Delegates to the core's own conversion in ``strata_core.synth``. The sweeps used
to import two local helpers that no longer exist, which made them fail at import
time - the quiet fate of every tuning script nobody runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _candidate in (ROOT / "vision", ROOT / "scripts"):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from strata_core.synth import ground_truth_in_canonical  # noqa: E402


def canonical_boxes(scene, canonical_size, *, include_destinations: bool = False):
    """Every expected change location, in canonical pixels."""
    return [
        list(g["bbox"])
        for g in ground_truth_in_canonical(
            scene, long_edge=canonical_size[0], include_destinations=include_destinations
        )
    ]


def canonical_primary(scene, canonical_size):
    """The expected set for kind scoring, with object identity kept.

    Destinations are excluded: a translation is one event, and asking a single
    event to satisfy two expected locations caps recall below 1.0 by construction.
    """
    return [
        {**g, "baseline_bbox": list(g["bbox"])}
        for g in ground_truth_in_canonical(scene, long_edge=canonical_size[0])
    ]
