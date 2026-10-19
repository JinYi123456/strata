"""Upper-bound probe: what does the harness score a *perfect* detector?

The change-detection score is capped by the harness's own bookkeeping, not by
the detector, if the expected set contains more locations than a correct
detector would ever report. A translation is exactly that case: the system
collapses it into **one** ``moved`` event, while the generator can enumerate the
vacated position *and* the destination as two expected locations.

This script removes the detector from the question. It synthesises the reported
events straight from the scene's own ground truth - correct boxes, correct
kinds, nothing else - and hands them to the same scoring functions
``scripts/evaluate.py`` uses. Whatever number comes back is the score the
harness would award a detector that is exactly right, so any real result can be
read against it.

Two modes, because both halves of the lesson are worth being able to show:

    python scripts/oracle.py --cases 6
    python scripts/oracle.py --cases 6 --include-destinations

The second reproduces the bug that was found and fixed: with destinations left
in the expected set, a *perfect* detector cannot score above 0.80 recall, and
the pipeline's own 0.800 was therefore already at the ceiling rather than short
of a target.

It shares the coordinate conversion with the harness rather than re-deriving it.
An earlier copy of this file imported a helper that did not exist and failed at
import time, which is the failure mode of a probe nobody runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vision"))
sys.path.insert(0, str(ROOT / "scripts"))

from strata_core.evaluate import score_event_kinds, score_regions  # noqa: E402
from strata_core.synth import (  # noqa: E402
    canonical_size_for_scene,
    ground_truth_in_canonical,
    make_scene,
)

# The same generosity the harness applies: an arrival that a translation also
# explains is scored as correct either way. Kept identical so the ceiling is
# measured under the same rules as the real result.
ACCEPTED_KINDS = {"added": ("moved",)}
IOU_THRESHOLD = 0.25


def oracle_case(
    seed: int,
    *,
    size: tuple[int, int] = (960, 720),
    include_destinations: bool = False,
    long_edge: int = 1600,
):
    """Score a detector that reports exactly the truth, once per mutation."""
    scene = make_scene(seed=seed, size=size)
    canonical = canonical_size_for_scene(scene, long_edge)

    # A perfect detector reports one event per mutation, at the location the
    # mutation happened, typed correctly. It does NOT report a second event for
    # a translation's destination - the destination is where the *same* object
    # now is, not a separate change.
    reported_truth = ground_truth_in_canonical(
        scene, long_edge=canonical[0], include_destinations=False
    )
    reported = [
        {"kind": g["kind"], "bbox": list(g["bbox"]), "confidence": 1.0}
        for g in reported_truth
    ]

    expected_truth = ground_truth_in_canonical(
        scene, long_edge=canonical[0], include_destinations=include_destinations
    )
    expected = [
        {**g, "baseline_bbox": list(g["bbox"])} for g in expected_truth
    ]

    metrics = score_regions(
        [r["bbox"] for r in reported], [e["bbox"] for e in expected],
        iou_threshold=IOU_THRESHOLD,
    )
    kind_acc, _details = score_event_kinds(
        reported, expected, iou_threshold=IOU_THRESHOLD, accepted_kinds=ACCEPTED_KINDS,
    )
    return metrics, kind_acc, len(reported), len(expected), canonical


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", type=int, default=6)
    ap.add_argument("--include-destinations", action="store_true",
                    help="score against the pre-fix expected set, which counted a "
                         "translation's origin and destination as two changes")
    args = ap.parse_args()

    print(f"{'seed':>6} {'P':>6} {'R':>6} {'F1':>6} {'kind':>6}  "
          f"{'reported':>8} {'expected':>8}")
    tot_p = tot_r = tot_k = 0.0
    rows = []
    for i in range(args.cases):
        seed = 1000 + i * 7
        m, k, nrep, nexp, _canonical = oracle_case(
            seed, include_destinations=args.include_destinations
        )
        print(f"{seed:>6} {m.precision:>6.3f} {m.recall:>6.3f} {m.f1:>6.3f} {k:>6.3f}  "
              f"{nrep:>8} {nexp:>8}")
        tot_p += m.precision
        tot_r += m.recall
        tot_k += k
        rows.append(m)

    n = max(1, args.cases)
    mean_p, mean_r = tot_p / n, tot_r / n
    f1 = (2 * mean_p * mean_r / (mean_p + mean_r)) if (mean_p or mean_r) else 0.0
    print("-" * 62)
    print(f"{'mean':>6} {mean_p:>6.3f} {mean_r:>6.3f} {f1:>6.3f} {tot_k / n:>6.3f}")
    print()
    if args.include_destinations:
        print("This is the harness BEFORE the fix: a translation was enumerated as")
        print("two expected locations while a correct system reports one event, so")
        print("recall was capped at 4/5 by construction. The pipeline's own 0.800")
        print("was already at this ceiling.")
    else:
        print("Read this as the harness's ceiling under the current expected set:")
        print("a detector that is exactly right scores 1.000 on every metric, so")
        print("any shortfall in the real evaluation belongs to the detector.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
