"""Explain each case: which GT boxes were missed and which detections were spurious."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vision"))

from strata_core.config import PipelineConfig  # noqa: E402
from strata_core.pipeline import run_scan  # noqa: E402
from strata_core import synth  # noqa: E402


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / float(aw * ah + bw * bh - inter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1000, 1014, 1021])
    a = ap.parse_args()
    cfg = PipelineConfig()

    for seed in a.seeds:
        scene = synth.make_scene(seed=seed)
        result = run_scan([scene.baseline], [scene.rescan], config=cfg)
        src = scene.baseline.shape[1], scene.baseline.shape[0]
        canon = result.registration.canonical_size
        sx, sy = canon[0] / src[0], canon[1] / src[1]

        gt = []
        for e in scene.ground_truth_events():
            if e.get("bbox_space") == "rescan" or e["kind"] == "moved_destination":
                continue
            x, y, w, h = e["baseline_bbox"]
            gt.append((e["kind"], [x * sx, y * sy, w * sx, h * sy]))

        dets = list(result.events)
        print(f"\n=== seed {seed}  mutations={[m.kind for m in scene.mutations]} ===")
        for kind, box in gt:
            best = max((iou(box, d.bbox) for d in dets), default=0.0)
            tag = "OK " if best >= 0.35 else "MISS"
            print(f"  {tag} GT {kind:16s} iou_best={best:.2f} box=[{box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}]")
        for d in dets:
            best = max((iou(d.bbox, b) for _, b in gt), default=0.0)
            tag = "OK " if best >= 0.35 else "FP "
            print(f"  {tag} DET {d.kind:16s} conf={d.confidence:.2f} box=[{d.bbox[0]:.0f},{d.bbox[1]:.0f},{d.bbox[2]:.0f},{d.bbox[3]:.0f}] best_gt_iou={best:.2f}")


if __name__ == "__main__":
    main()
