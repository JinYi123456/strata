"""Deep diagnostic for one synthetic case.

A tuning sweep that returns identical zeros for every axis means the problem is
not the thresholds - it is upstream of them. This script answers the questions
that a parameter sweep cannot:

1. Is the rescan actually a *rendering of the same scene* after mutation, or was
   the mutation applied and then the whole thing warped in a way that also
   scales it?
2. Where do the ground-truth boxes land once mapped into canonical space, versus
   where the detected regions are?
3. Does the diff mask actually contain pixels inside the ground-truth boxes, or
   is the mask simply empty / elsewhere?

Usage:
    python scripts/diagnose.py --seed 1000 --out artifacts/diag
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vision"))

from strata_core.config import PipelineConfig  # noqa: E402
from strata_core.pipeline import run_scan  # noqa: E402
from strata_core.synth import make_scene  # noqa: E402


def box_in_canonical(
    bbox: tuple[int, int, int, int],
    H_scene_to_canonical: np.ndarray,
) -> tuple[int, int, int, int]:
    """Map a scene-space box into canonical space via its four corners."""
    x, y, w, h = bbox
    corners = np.array(
        [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.float64
    ).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(corners, H_scene_to_canonical).reshape(-1, 2)
    x0, y0 = out.min(axis=0)
    x1, y1 = out.max(axis=0)
    return (int(round(x0)), int(round(y0)), int(round(x1 - x0)), int(round(y1 - y0)))


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out", type=str, default="artifacts/diag")
    ap.add_argument("--objects", type=int, default=18)
    ap.add_argument("--mutations", type=int, default=3)
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    scene = make_scene(
        seed=args.seed, size=(960, 720),
        n_objects=args.objects, n_mutations=args.mutations,
    )
    print(f"scene seed={args.seed} size={scene.size} "
          f"objects={len(scene.baseline_objects)} mutations={len(scene.mutations)}")
    for m in scene.mutations:
        print(f"  mutation {m.kind:<9} {m.object_id:<8} dx={m.dx} dy={m.dy}")

    cfg = PipelineConfig()
    res = run_scan([scene.baseline], [scene.rescan], config=cfg)

    print(f"\nbackend={res.feature_backend}")
    print(f"registration model={res.registration.model} "
          f"inliers={res.registration.inlier_count} "
          f"ratio={res.registration.inlier_ratio:.3f} "
          f"conf={res.registration.confidence:.3f} "
          f"reproj={res.registration.reproj_error_px:.3f}px")
    print("h_true (rescan->baseline, scene px):\n", np.round(scene.homography, 4))
    print("h_est  (rescan->baseline, scene px):\n", np.round(res.registration.matrix, 4))

    # --- registration accuracy in *scene* pixels -------------------------
    w, h = scene.size
    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float64).reshape(-1, 1, 2)
    a = cv2.perspectiveTransform(corners, scene.homography).reshape(-1, 2)
    b = cv2.perspectiveTransform(corners, res.registration.matrix).reshape(-1, 2)
    print(f"corner error (scene px): {np.mean(np.linalg.norm(a - b, axis=1)):.3f}")

    # --- canonical geometry ----------------------------------------------
    cw, ch = res.diff.canonical_size
    print(f"\ncanonical_size={cw}x{ch}")
    print(f"diff stats: {json.dumps(res.diff.stats, indent=2)}")
    print(f"change_fraction={res.diff.change_fraction:.5f}  "
          f"mask_px={int(np.count_nonzero(res.diff.mask))}")
    print(f"detected regions={len(res.diff.regions)}")

    # --- where should ground truth land? ----------------------------------
    # scene (baseline) px -> canonical px  ==  scale of canonical frame
    S = np.diag([cw / float(w), ch / float(h), 1.0])
    H_scene_to_canonical = S

    print("\n--- ground truth vs detection (canonical space) ---")
    gt = scene.ground_truth_events()
    gt_boxes = []
    for g in gt:
        bcan = box_in_canonical(tuple(g["baseline_bbox"]), H_scene_to_canonical)
        gt_boxes.append((g["kind"], g["object_id"], bcan))
        # how many diff-mask pixels fall inside this ground-truth box?
        x, y, bw, bh = bcan
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(cw, x + bw), min(ch, y + bh)
        inside = int(np.count_nonzero(res.diff.mask[y0:y1, x0:x1])) if x1 > x0 and y1 > y0 else 0
        area = max(1, (x1 - x0) * (y1 - y0))
        print(f"  GT {g['kind']:<13} {g['object_id']:<8} canonical_bbox={bcan} "
              f"mask_px_inside={inside}/{area} ({100.0*inside/area:.1f}%)")

    print("\ndetected regions:")
    for r in res.diff.regions:
        best = max((iou(r.bbox, b[2]) for b in gt_boxes), default=0.0)
        print(f"  #{r.region_id:<3} bbox={r.bbox} area={r.area_px:<7} "
              f"conf={r.confidence:.3f} low_conf={r.low_confidence} "
              f"mean={r.mean_abs_diff:.2f} best_gt_iou={best:.3f}")

    print("\nevents:")
    for e in res.events:
        print(f"  {e.event_id} {e.kind:<14} conf={e.confidence:.3f} bbox={e.bbox} "
              f"obj={e.evidence.get('baseline_objectness')}->"
              f"{e.evidence.get('rescan_objectness')}")

    # --- artefact dump ----------------------------------------------------
    canvas = res.baseline.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    for kind, oid, (x, y, bw, bh) in gt_boxes:
        cv2.rectangle(canvas, (x, y), (x + bw, y + bh), (0, 255, 0), 3)
        cv2.putText(canvas, f"GT {kind}", (x, max(16, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite(str(out / f"diag_gt_{args.seed}.png"), canvas)
    cv2.imwrite(str(out / f"diag_overlay_{args.seed}.png"), res.overlay())
    cv2.imwrite(str(out / f"diag_mask_{args.seed}.png"), res.diff.mask)
    cv2.imwrite(str(out / f"diag_side_{args.seed}.png"), res.side_by_side())
    print(f"\nwrote artefacts to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
