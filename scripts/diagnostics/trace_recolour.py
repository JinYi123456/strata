"""Trace where a large in-scene signal disappears in the pipeline."""
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


def gray(a):
    return cv2.cvtColor(a, cv2.COLOR_BGR2GRAY) if a.ndim == 3 else a


def main():
    cfg = PipelineConfig()
    scene = synth.make_scene(seed=1000)
    res = run_scan([scene.baseline], [scene.rescan], config=cfg)
    src = scene.baseline.shape[1], scene.baseline.shape[0]
    cw, ch = res.registration.canonical_size
    sx, sy = cw / src[0], ch / src[1]
    print(f"canonical {cw}x{ch} scale {sx:.3f},{sy:.3f}")

    o = next(o for o in scene.baseline_objects if o.object_id == "obj15")
    print(f"obj15 scene box {o.bbox}")

    base_c = gray(res.baseline)
    resc_c = gray(res.rescan_canonical)

    # canonical box
    cb = (int(o.x * sx), int(o.y * sy), int(o.w * sx), int(o.h * sy))
    print(f"canonical box {cb}")

    def delta_on(img_a, img_b, box, tag):
        x, y, w, h = box
        H, W = img_a.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        a = img_a[y0:y1, x0:x1].astype(np.float32)
        b = img_b[y0:y1, x0:x1].astype(np.float32)
        d = np.abs(a - b).mean()
        print(f"  {tag:34s} raw-delta={d:6.2f}  a.std={a.std():5.1f} b.std={b.std():5.1f}")
        return d

    print("\n-- progressive stages --")
    delta_on(base_c, resc_c, cb, "A: canonical baseline vs rescan")

    dg = res.diff.diff_gray.astype(np.float32)
    delta_on(np.zeros_like(dg), dg, cb, "B: normalised diff_gray")

    # Stage: what does the pipeline's own normalisation do?
    from strata_core.diff import _normalise
    nb = _normalise(base_c, cfg.diff)
    nr = _normalise(resc_c, cfg.diff)
    delta_on(nb, nr, cb, "C: after _normalise on both")

    x, y, w, h = cb
    a = nb[y:y + h, x:x + w]
    b = nr[y:y + h, x:x + w]
    print(f"\n  raw abs delta in box (normalised) = {np.abs(a-b).mean():.2f}")
    print(f"  local mean a={a.mean():.1f} b={b.mean():.1f}  (difference of means {abs(a.mean()-b.mean()):.1f})")

    # The gating step
    print("\n-- the local-gate step in _normalise --")
    import inspect
    print(inspect.getsource(_normalise))


if __name__ == "__main__":
    main()
