"""Compare translation-pairing policies: which arrival kinds may be paired."""
from __future__ import annotations
import sys, time
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

SEEDS = [1000, 1007, 1014, 1021, 1028, 1035, 1042, 1049, 1056, 1063]
SIZE = (960, 720)


def evaluate(pair_kinds: tuple[str, ...], min_px: float = 30.0):
    import strata_core.events as EV
    orig = EV._resolve_translations
    base = PipelineConfig()

    def patched(events, cfg, base_gray=None, res_gray=None):
        deps = [e for e in events if e.kind == "missing" and e.evidence.get("area_px", 0) > 0]
        arrs = [e for e in events
                if e.kind in pair_kinds and e.evidence.get("area_px", 0) > 0]
        if not deps or not arrs:
            return events
        import numpy as np
        merged, consumed = [], set()
        for dep in deps:
            best, best_cost = None, float("inf")
            for arr in arrs:
                if id(arr) in consumed:
                    continue
                gx = max(dep.bbox[0] - (arr.bbox[0] + arr.bbox[2]),
                         arr.bbox[0] - (dep.bbox[0] + dep.bbox[2]), 0.0)
                gy = max(dep.bbox[1] - (arr.bbox[1] + arr.bbox[3]),
                         arr.bbox[1] - (dep.bbox[1] + dep.bbox[3]), 0.0)
                sep = float(np.hypot(gx, gy))
                cd = float(np.hypot(dep.centroid[0] - arr.centroid[0],
                                    dep.centroid[1] - arr.centroid[1]))
                if cd < min_px or sep > cfg.translation_max_px:
                    continue
                cost = sep * (2.0 - EV._box_similarity(dep.bbox, arr.bbox))
                if cost < best_cost:
                    best_cost, best = cost, arr
            if best is None:
                continue
            consumed.add(id(best))
            merged.append(EV.ChangeEvent(
                event_id=dep.event_id, kind="moved", confidence=0.8,
                regions=dep.regions + best.regions, bbox=dep.bbox,
                centroid=dep.centroid,
                displacement_px=float(np.hypot(dep.centroid[0] - best.centroid[0],
                                               dep.centroid[1] - best.centroid[1])),
                appearance_delta=dep.appearance_delta, summary="moved",
                evidence={"area_px": dep.evidence.get("area_px"),
                          "presence_gap": dep.evidence.get("presence_gap", 0.0)},
            ))
        out = [e for e in events if id(e) not in consumed]
        out.extend(merged)
        return out

    EV._resolve_translations = patched
    try:
        P = R = K = 0.0
        n = 0
        for seed in SEEDS:
            scene = make_scene(seed=seed, size=SIZE, n_objects=18, n_mutations=4)
            res = run_scan([scene.baseline], [scene.rescan], config=base)
            cs = res.registration.canonical_size
            truth = canonical_boxes(scene, cs)
            m = score_regions([e.bbox for e in res.events], truth, iou_threshold=0.25)
            ps = canonical_primary(scene, cs)
            k, _ = score_event_kinds([e.to_dict() for e in res.events], ps,
                                     iou_threshold=0.25, accepted_kinds={"added": ("moved",)})
            P += m.precision; R += m.recall; K += k; n += 1
        return P / n, R / n, K / n
    finally:
        EV._resolve_translations = orig


if __name__ == "__main__":
    print(f"{'arrival kinds':>28} {'P':>7} {'R':>7} {'F1':>7} {'kind':>7}")
    for kinds in [("added",), ("added", "moved"), ("added", "state_changed", "moved")]:
        t0 = time.time()
        p, r, k = evaluate(kinds)
        f1 = 2 * p * r / max(1e-9, p + r)
        print(f"{str(kinds):>28} {p:>7.3f} {r:>7.3f} {f1:>7.3f} {k:>7.3f}   ({time.time()-t0:.0f}s)")
