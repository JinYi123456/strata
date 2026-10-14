"""Sweep STRATA detection parameters against a fixed case set.

Hand-tuning thresholds one at a time against single examples is how you get a
system that works on the three scenes you looked at. This script evaluates a
grid of settings over many cases at once and reports precision / recall / F1
plus the false-positive rate on still pairs, so the choice of operating point is
based on measured trade-offs rather than intuition.

Usage:
    python scripts/tune.py                       # default sweep
    python scripts/tune.py --quick               # smaller grid
    python scripts/tune.py --out artifacts/tune
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vision"))

from strata_core.config import DiffConfig, EventConfig, PipelineConfig  # noqa: E402
from strata_core.evaluate import score_regions  # noqa: E402
from strata_core.pipeline import run_scan  # noqa: E402
from strata_core.synth import make_scene, make_still_pair  # noqa: E402

log = logging.getLogger("strata.tune")


def evaluate_setting(
    cfg: PipelineConfig,
    case_seeds: list[int],
    still_seeds: list[int],
    *,
    size: tuple[int, int],
) -> dict:
    tp = fp = fn = 0
    for s in case_seeds:
        sc = make_scene(seed=s, size=size, n_objects=18, n_mutations=3)
        res = run_scan([sc.baseline], [sc.rescan], config=cfg)
        cw, ch = res.diff.canonical_size
        sx, sy = cw / float(size[0]), ch / float(size[1])
        # Ground truth lives in scene pixels; detections live in canonical
        # pixels. Convert before comparing, or the score is meaningless.
        truth = [
            (g["baseline_bbox"][0] * sx, g["baseline_bbox"][1] * sy,
             g["baseline_bbox"][2] * sx, g["baseline_bbox"][3] * sy)
            for g in sc.ground_truth_events()
        ]
        # Only definite events count towards precision; uncertainties are
        # explicitly the system declining to commit, not a claim.
        det = [
            e.bbox for e in res.events
            if e.kind != "uncertain" and e.confidence >= cfg.events.min_confidence
        ]
        m = score_regions(det, truth, iou_threshold=0.25)
        tp += m.true_positives
        fp += m.false_positives
        fn += m.false_negatives

    still_fp = 0
    for s in still_seeds:
        b, r, _ = make_still_pair(seed=s)
        res = run_scan([b], [r], config=cfg)
        still_fp += len([
            e for e in res.events
            if e.kind != "uncertain" and e.confidence >= cfg.events.min_confidence
        ])

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "still_fp": still_fp,
        "still_fp_per_pair": round(still_fp / max(1, len(still_seeds)), 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Tune STRATA detection parameters")
    ap.add_argument("--cases", type=int, default=8)
    ap.add_argument("--still-pairs", type=int, default=6)
    ap.add_argument("--out", type=str, default="artifacts/tune")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)
    size = (args.width, args.height)
    case_seeds = [1000 + i * 7 for i in range(args.cases)]
    still_seeds = [2000 + i * 11 for i in range(args.still_pairs)]

    if args.quick:
        min_areas = [320, 520]
        min_deltas = [22.0, 30.0]
        obj_floors = [0.26, 0.34]
    else:
        min_areas = [260, 380, 520, 700]
        min_deltas = [18.0, 24.0, 30.0, 38.0]
        obj_floors = [0.22, 0.28, 0.34, 0.42]

    grid = list(itertools.product(min_areas, min_deltas, obj_floors))
    print(f"Sweeping {len(grid)} settings over {len(case_seeds)} change cases "
          f"and {len(still_seeds)} still pairs\n")

    base = PipelineConfig()
    results = []
    t0 = time.perf_counter()
    for i, (area, delta, floor) in enumerate(grid, 1):
        cfg = PipelineConfig.from_dict({
            **base.to_dict(),
            "diff": {**base.diff.__dict__, "min_region_area_px": area, "min_abs_delta": delta},
            "events": {**base.events.__dict__},
        })
        # The objectness floor is a module constant; patch it for the sweep.
        import strata_core.events as EV
        original_floor = EV.OBJECTNESS_FLOOR
        EV.OBJECTNESS_FLOOR = floor
        try:
            metrics = evaluate_setting(cfg, case_seeds, still_seeds, size=size)
        finally:
            EV.OBJECTNESS_FLOOR = original_floor
        metrics.update({"min_area": area, "min_abs_delta": delta, "objectness_floor": floor})
        results.append(metrics)
        print(
            f"[{i:>3}/{len(grid)}] area={area:<4} delta={delta:<5} obj={floor:<5} "
            f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f} "
            f"stillFP/pair={metrics['still_fp_per_pair']:.2f}"
        )

    elapsed = time.perf_counter() - t0

    # Rank by F1, with the still-pair false-positive rate as a hard qualifier:
    # a setting that cries wolf is not usable however good its F1 looks.
    def rank_key(m: dict) -> tuple:
        usable = 1 if m["still_fp_per_pair"] <= 0.5 else 0
        return (usable, m["f1"], -m["still_fp_per_pair"])

    ranked = sorted(results, key=rank_key, reverse=True)
    print("\n" + "=" * 78)
    print("TOP 8 SETTINGS (ranked by usability, then F1)")
    print("=" * 78)
    print(f"{'area':>6} {'delta':>7} {'objfloor':>9} {'P':>7} {'R':>7} {'F1':>7} {'stillFP':>8}")
    for m in ranked[:8]:
        print(
            f"{m['min_area']:>6} {m['min_abs_delta']:>7} {m['objectness_floor']:>9} "
            f"{m['precision']:>7.3f} {m['recall']:>7.3f} {m['f1']:>7.3f} "
            f"{m['still_fp_per_pair']:>8.2f}"
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tuning.json").write_text(
        json.dumps({
            "params": {"cases": len(case_seeds), "still_pairs": len(still_seeds), "size": list(size)},
            "best": ranked[0] if ranked else None,
            "ranked": ranked,
            "all": results,
            "elapsed_seconds": round(elapsed, 2),
        }, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {out/'tuning.json'}  ({elapsed:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
