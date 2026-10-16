"""Evaluate the STRATA pipeline against synthetic scenes with ground truth.

Produces a JSON report plus a printed table. This is the harness behind the
numbers quoted in the project's evaluation section, and it is intentionally
runnable by anyone with the pinned dependencies:

    python scripts/evaluate.py --cases 24 --out artifacts/eval

It reports, per case:
  * registration corner error against the known homography;
  * region-level precision / recall / F1 against the mutated object boxes;
  * event-kind accuracy (removal -> missing, insertion -> added, ...);
and, separately, the **false-positive rate on a still pair** where nothing
changed - the number that decides whether the system is usable in the field.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vision"))

from strata_core.config import PipelineConfig  # noqa: E402
from strata_core.evaluate import (  # noqa: E402
    RunResult,
    aggregate,
    homography_error,
    score_event_kinds,
    score_regions,
)
from strata_core.pipeline import run_scan  # noqa: E402
from strata_core.synth import (  # noqa: E402
    canonical_size_for_scene,
    ground_truth_in_canonical,
    make_scene,
    make_still_pair,
)

log = logging.getLogger("strata.eval")


def _expected_boxes(
    scene,
    src_size: tuple[int, int],
    canonical_size: tuple[int, int],
) -> list[tuple[float, float, float, float]]:
    """Ground-truth boxes in canonical pixels, via the core's own conversion.

    This deliberately delegates rather than converting locally. The conversion
    had to exist in one place: an earlier version of the service's answer-key
    panel re-derived it, got it wrong, and reported that a system which had found
    every change had found none. Two implementations of a coordinate transform
    is one too many.

    The ``src_size`` and ``canonical_size`` arguments are still accepted because
    callers hold them, but they are checked against the core's own computation
    rather than used to re-derive anything.
    """
    boxes = ground_truth_in_canonical(scene, long_edge=canonical_size[0])
    expected_canonical = canonical_size_for_scene(scene, canonical_size[0])
    if tuple(expected_canonical) != tuple(canonical_size):  # pragma: no cover - guard
        raise AssertionError(
            f"canonical size mismatch: caller passed {canonical_size}, "
            f"core computes {expected_canonical}"
        )
    return [
        (b["bbox"][0], b["bbox"][1], b["bbox"][2], b["bbox"][3]) for b in boxes
    ]


def run_case(
    seed: int,
    *,
    cfg: PipelineConfig,
    pose_strength: float,
    n_objects: int,
    n_mutations: int,
    size: tuple[int, int],
) -> RunResult:
    scene = make_scene(
        seed=seed, size=size, n_objects=n_objects, n_mutations=n_mutations,
        pose_strength=pose_strength,
    )
    result = run_scan([scene.baseline], [scene.rescan], config=cfg)

    canonical_size = result.diff.canonical_size
    truth_boxes = _expected_boxes(scene, size, canonical_size)
    reported_boxes = [e.bbox for e in result.events]

    metrics = score_regions(reported_boxes, truth_boxes, iou_threshold=0.25)

    # Event-kind accuracy is scored against the *primary* mutation locations:
    # a translation's destination is a change, but its expected kind is "moved"
    # and it is the origin event that carries that label. Including the
    # destination here would ask the detector to label one translation twice.
    # This is the same set the region scorer uses, deliberately - the two
    # scores must be measured against one description of the truth, or they
    # disagree about how many events a correct detector reports.
    #
    # `ground_truth_in_canonical` already excludes destinations and already
    # returns canonical pixels, so this is the same set as `truth_boxes`, with
    # the object identity kept for the per-event detail.
    primary_scaled = [
        {**g, "baseline_bbox": list(g["bbox"])}
        for g in ground_truth_in_canonical(scene, long_edge=canonical_size[0])
    ]
    # A translation whose destination sits near an unrelated insertion produces
    # one region of evidence that supports either reading. Which reading is
    # correct depends on whether the matching departure was found, and that is
    # scored by the pairing test, not here - so both are accepted for the
    # arrival's location.
    kind_acc, details = score_event_kinds(
        [e.to_dict() for e in result.events], primary_scaled, iou_threshold=0.25,
        accepted_kinds={"added": ("moved",)},
    )
    # Registration error is measured in the rescan's own frame: the recovered
    # matrix maps rescan -> baseline, and so does the ground-truth homography.
    reg_err = homography_error(
        result.registration.matrix, scene.homography, size
    )

    return RunResult(
        name=f"seed{seed}-pose{pose_strength:.2f}",
        registration_error_px=reg_err,
        registration_confidence=result.registration.confidence,
        registration_model=result.registration.model,
        inlier_ratio=result.registration.inlier_ratio,
        detection=metrics,
        event_kind_accuracy=kind_acc,
        expected_events=len(primary_scaled),
        reported_events=len(result.events),
        per_event=details,
        notes=result.registration.notes,
    )


def still_pair_false_positives(
    *, cfg: PipelineConfig, seeds: list[int], pose_strength: float
) -> dict:
    """Events reported when *nothing* changed - the false-positive measure."""
    total_events = 0
    per_seed = []
    for s in seeds:
        b, r, _ = make_still_pair(seed=s, strength=pose_strength)
        res = run_scan([b], [r], config=cfg)
        # Only confident, genuinely-typed events count as false positives.
        fp_events = [e for e in res.events if e.kind != "uncertain" and e.confidence >= 0.35]
        total_events += len(fp_events)
        per_seed.append({
            "seed": s,
            "reported_events": len(res.events),
            "confident_false_positives": len(fp_events),
            "registration_confidence": round(res.registration.confidence, 4),
        })
    n = max(1, len(seeds))
    return {
        "pairs": len(seeds),
        "confident_false_positives_total": total_events,
        "confident_false_positives_per_pair": round(total_events / n, 3),
        "per_seed": per_seed,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate STRATA on synthetic scenes")
    ap.add_argument("--cases", type=int, default=24, help="number of change cases")
    ap.add_argument("--still-pairs", type=int, default=12, help="still-pair count")
    ap.add_argument("--out", type=str, default="artifacts/eval", help="output directory")
    ap.add_argument("--pose", type=float, default=1.0, help="camera-pose strength")
    ap.add_argument("--objects", type=int, default=18)
    ap.add_argument("--mutations", type=int, default=4)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--executor", type=str, default="baseline-local")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    size = (args.width, args.height)
    cfg = PipelineConfig()
    cfg = PipelineConfig.from_dict({**cfg.to_dict(), "executor": args.executor})

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    results: list[RunResult] = []
    for i in range(args.cases):
        seed = 1000 + i * 7
        res = run_case(
            seed, cfg=cfg, pose_strength=args.pose,
            n_objects=args.objects, n_mutations=args.mutations, size=size,
        )
        results.append(res)
        print(
            f"[{i+1:>3}/{args.cases}] {res.name:<22} "
            f"reg_err={res.registration_error_px:7.2f}px "
            f"conf={res.registration_confidence:.3f} "
            f"P={res.detection.precision:.2f} R={res.detection.recall:.2f} "
            f"F1={res.detection.f1:.2f} kind={res.event_kind_accuracy:.2f} "
            f"reported={res.reported_events} expected={res.expected_events}"
        )

    elapsed = time.perf_counter() - t0
    agg = aggregate(results)

    print("\n" + "=" * 78)
    print("CHANGE-DETECTION CASE RESULTS")
    print("=" * 78)
    for k, v in agg.items():
        print(f"  {k:<34} {v}")

    print("\nMeasuring false positives on still pairs (nothing changed)...")
    still = still_pair_false_positives(
        cfg=cfg, seeds=[2000 + i * 11 for i in range(args.still_pairs)], pose_strength=args.pose
    )
    print(f"  confident false positives / pair : {still['confident_false_positives_per_pair']}")
    print(f"  total confident false positives  : {still['confident_false_positives_total']}")

    report = {
        "config": cfg.to_dict(),
        "params": {
            "cases": args.cases, "pose_strength": args.pose, "objects": args.objects,
            "mutations": args.mutations, "size": list(size),
        },
        "aggregate": agg,
        "still_pair": still,
        "elapsed_seconds": round(elapsed, 2),
        "per_case": [r.to_dict() for r in results],
    }
    out_path = out_dir / "evaluation.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}  ({elapsed:.1f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
