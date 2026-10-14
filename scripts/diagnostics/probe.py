"""Probe why certain mutations produce no region, and what the slivers are."""
from __future__ import annotations
import sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vision"))

from strata_core.config import PipelineConfig
from strata_core.pipeline import run_scan
from strata_core import synth


def main():
    cfg = PipelineConfig()
    for seed in (1000, 1014, 1021, 1028):
        scene = synth.make_scene(seed=seed)
        res = run_scan([scene.baseline], [scene.rescan], config=cfg)
        src = scene.baseline.shape[1], scene.baseline.shape[0]
        cw, ch = res.registration.canonical_size
        sx, sy = cw / src[0], ch / src[1]
        base = cv2.cvtColor(res.baseline, cv2.COLOR_BGR2GRAY)
        rsc = cv2.cvtColor(res.rescan_canonical, cv2.COLOR_BGR2GRAY)
        mag = res.diff.diff_gray.astype(np.float32)

        print(f"\n=== seed {seed} ===")
        print(f"  stats={ {k: v for k, v in res.diff.stats.items() if 'thresh' in k or 'region' in k} } regions={len(res.diff.regions)}")
        for m in scene.mutations:
            src_obj = next((o for o in scene.baseline_objects if o.object_id == m.object_id), None)
            if src_obj is None:
                continue
            x, y, w, h = [int(round(v)) for v in (src_obj.x * sx, src_obj.y * sy,
                                                  src_obj.w * sx, src_obj.h * sy)]
            x, y = max(0, x), max(0, y)
            patch = mag[y:y + h, x:x + w]
            if patch.size == 0:
                print(f"  {m.kind:9s} {m.object_id}  (box off-frame)")
                continue
            bd = np.abs(base[y:y + h, x:x + w].astype(np.float32) - rsc[y:y + h, x:x + w].astype(np.float32))
            # how much of the box is above threshold
            thr = float(res.diff.stats.get('threshold_used', 50.0))
            frac = float((patch > thr).mean())
            bstats = (base[y:y + h, x:x + w].astype(np.float32))
            rstats = (rsc[y:y + h, x:x + w].astype(np.float32))
            print(
                f"  {m.kind:9s} {m.object_id}  "
                f"diff mean={bd.mean():6.1f} p95={np.percentile(bd,95):6.1f} frac>thr={frac:5.1%}  "
                f"base(std={bstats.std():5.1f} mean={bstats.mean():6.1f}) "
                f"res(std={rstats.std():5.1f} mean={rstats.mean():6.1f})"
            )


if __name__ == "__main__":
    main()
