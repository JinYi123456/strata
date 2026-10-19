"""Sweep presence_gap_min against the corrected benchmark."""
from __future__ import annotations
import sys, time, json
from pathlib import Path
from dataclasses import replace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vision"))
sys.path.insert(0, str(ROOT / "scripts"))

from strata_core.config import PipelineConfig
from strata_core.pipeline import run_scan
from strata_core.synth import make_scene
from strata_core.evaluate import score_regions, score_event_kinds
from truth import canonical_boxes, canonical_primary

SEEDS = [1000, 1007, 1014, 1021, 1028, 1035, 1042, 1049]
SIZE = (960, 720)


def boxes(scene, canonical_size):
    return canonical_boxes(scene, canonical_size)


def run(gap_min: float, accept_displacement: float | None = None):
    base = PipelineConfig()
    ev = replace(base.events, presence_gap_min=gap_min)
    if accept_displacement is not None:
        ev = replace(ev, shift_match_min=accept_displacement)
    cfg = replace(base, events=ev)
    P = R = K = 0.0
    n = 0
    for seed in SEEDS:
        scene = make_scene(seed=seed, size=SIZE, n_objects=18, n_mutations=4)
        res = run_scan([scene.baseline], [scene.rescan], config=cfg)
        cs = res.registration.canonical_size
        truth = boxes(scene, cs)
        m = score_regions([e.bbox for e in res.events], truth, iou_threshold=0.25)
        ps = canonical_primary(scene, cs)
        k, _ = score_event_kinds([e.to_dict() for e in res.events], ps,
                                 iou_threshold=0.25, accepted_kinds={"added": ("moved",)})
        P += m.precision; R += m.recall; K += k; n += 1
    return P / n, R / n, K / n


if __name__ == "__main__":
    print(f"{'gap_min':>8} {'P':>7} {'R':>7} {'F1':>7} {'kind':>7}")
    for g in (0.04, 0.06, 0.08, 0.10, 0.13):
        t0 = time.time()
        p, r, k = run(g)
        f1 = 2 * p * r / max(1e-9, p + r)
        print(f"{g:>8.2f} {p:>7.3f} {r:>7.3f} {f1:>7.3f} {k:>7.3f}   ({time.time()-t0:.0f}s)")
