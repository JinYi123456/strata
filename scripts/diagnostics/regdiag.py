"""Why is the recovered homography off by tens of pixels?"""
from __future__ import annotations
import sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vision"))

from strata_core.config import PipelineConfig
from strata_core import synth
from strata_core.features import build_extractor, resolve_backend
from strata_core.features import Matcher, to_gray
from strata_core.geometry import register


def main():
    cfg = PipelineConfig()
    for seed in (1000, 1014, 1021, 1028):
        scene = synth.make_scene(seed=seed)
        extractor = build_extractor(cfg.features)
        backend = resolve_backend(cfg.features)
        matcher = Matcher(cfg.features, backend=backend)

        f_b = extractor.detect(to_gray(scene.baseline))
        f_r = extractor.detect(to_gray(scene.rescan))
        matches = matcher.match(f_r, f_b)

        print(f"\n=== seed {seed} ===")
        print(f"  kp baseline={len(f_b.keypoints)} rescan={len(f_r.keypoints)} matches={len(matches)}")

        # Where do the matched points sit? Count how many are on objects vs backdrop.
        obj_mask = np.zeros(scene.baseline.shape[:2], np.uint8)
        for o in scene.baseline_objects:
            obj_mask[o.y:o.y + o.h, o.x:o.x + o.w] = 255
        interior = cv2.erode(obj_mask, np.ones((9, 9), np.uint8))
        on_obj = 0
        for pt in matches.src_points:
            x, y = int(round(pt[0])), int(round(pt[1]))
            if 0 <= y < obj_mask.shape[0] and 0 <= x < obj_mask.shape[1] and interior[y, x]:
                on_obj += 1
        print(f"  matches on object interiors: {on_obj}/{len(matches)} "
              f"({on_obj/max(1,len(matches)):.0%})  <-- backdrop matches carry no pose info")

        reg = register(matches, src_shape=scene.rescan.shape[:2],
                       dst_shape=scene.baseline.shape[:2], cfg=cfg.geometry)
        print(f"  model={reg.model} inliers={reg.inlier_count}/{reg.match_count} ratio={reg.inlier_ratio:.3f} conf={reg.confidence:.3f} reproj={reg.reproj_error_px:.2f}px")

        # Evaluate both transforms in canonical space, in canonical pixels.
        cw, ch = reg.canonical_size
        S = np.diag([cw / scene.size[0], ch / scene.size[1], 1.0])
        H_est = S @ reg.matrix @ np.linalg.inv(S)
        H_tru = S @ scene.homography @ np.linalg.inv(S)
        pts = np.float32([[0, 0], [cw, 0], [cw, ch], [0, ch]]).reshape(-1, 1, 2)
        pe = cv2.perspectiveTransform(pts, H_est).reshape(-1, 2)
        pt = cv2.perspectiveTransform(pts, H_tru).reshape(-1, 2)
        err = np.linalg.norm(pe - pt, axis=1)
        print(f"  canonical corner error: mean={err.mean():6.2f}px max={err.max():6.2f}px")


if __name__ == "__main__":
    main()
