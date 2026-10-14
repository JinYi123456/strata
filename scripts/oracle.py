"""Upper-bound probe: what does the harness score a *perfect* detector?

The change-detection score is capped by the harness's own bookkeeping, not by
the detector, if the expected set contains more locations than a correct
detector would ever report. A translation is exactly that case: the system
collapses it into **one** ``moved`` event, while the generator enumerates the
vacated position *and* the destination as two expected locations.

This script removes the detector from the question. It synthesises the reported
events straight from the scene's own ground truth - correct boxes, correct
kinds, nothing else - and hands them to the same scoring functions
``scripts/evaluate.py`` uses. Whatever number comes back is the score the
harness would award a detector that is exactly right, so any real result can be
read against it.

Run:

    python scripts/oracle.py --cases 6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vision"))
sys.path.insert(0, str(ROOT / "scripts"))

from strata_core.evaluate import score_event_kinds, score_regions  # noqa: E402
from strata_core.synth import make_scene  # noqa: E402

from evaluate import _expected_boxes, _scene_to_canonical_box  # noqa: E402


def oracle_case(seed: int, *, size=(960, 720), canonical=(1600, 1200)):
    scene = make_scene(seed=seed, size=size)

    # A perfect detector reports one event per mutation, at the location the
    # mutation happened, typed correctly. It does NOT report a second event for
    # a translation's destination - the destination is where the *same* object
    # now is, not a separate change.
    primary = [g for g in scene.ground_truth_events() if g["kind"] != "moved_destination"]
    reported = []
    for g in primary:
        space = g.get("bbox_space", "baseline")
        if space == "rescan":
            box = g["baseline_bbox"]  # already in scene (rescan) coordinates
            box = _scene_to_canonical_box(box, size, canonical)
        else:
            box = _scene_to_canonical_box(g["baseline_bbox"], size, canonical)
        reported.append({"kind": g["kind"], "bbox": list(box), "confidence": 1.0})

    truth_boxes = _expected_boxes(scene, size, canonical)
    metrics = score_regions([r["bbox"] for r in reported], truth_boxes, iou_threshold=0.25)

    scaled_primary = [
        {**g, "baseline_bbox": list(_scene_to_canonical_box(g["baseline_bbox"], size, canonical))}
        for g in primary
    ]
    kind_acc, _ = score_event_kinds(
        reported, scaled_primary, iou_threshold=0.25,
        accepted_kinds={"added": ("moved",)},
    )
    return metrics, kind_acc, len(reported), len(truth_boxes), len(primary)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=6)
    a = ap.parse_args()

    print(f"{'seed':>6} {'P':>6} {'R':>6} {'F1':>6} {'kind':>6}  "
          f"{'reported':>8} {'truth_boxes':>11} {'primary':>7}")
    tot_p = tot_r = tot_k = 0.0
    for i in range(a.cases):
        seed = 1000 + i * 7
        m, k, nrep, ntruth, nprim = oracle_case(seed)
        print(f"{seed:>6} {m.precision:>6.3f} {m.recall:>6.3f} {m.f1:>6.3f} {k:>6.3f}  "
              f"{nrep:>8} {ntruth:>11} {nprim:>7}")
        tot_p += m.precision
        tot_r += m.recall
        tot_k += k
    n = max(1, a.cases)
    f1 = (2 * (tot_p / n) * (tot_r / n) / ((tot_p / n) + (tot_r / n))) if (tot_p or tot_r) else 0.0
    print("-" * 62)
    print(f"{'mean':>6} {tot_p / n:>6.3f} {tot_r / n:>6.3f} {f1:>6.3f} {tot_k / n:>6.3f}")
    print()
    print("Read this as the harness's ceiling: a detector that is exactly right")
    print("cannot score above these numbers under the current expected set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
