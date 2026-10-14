"""Diagnose the translation-pairing pass: was a correct arrival available?

The pairing pass in ``events._resolve_translations`` promotes a ``missing``
departure plus a nearby ``added`` arrival into a single ``moved`` event. Two
failures are possible and they look identical from the outside - a move that
came back unpaired - but they need opposite fixes:

  * **no arrival available**: the region at the destination was typed something
    other than ``added`` (commonly ``moved`` or ``state_changed``), so the pass
    never saw it. The fix is in the *classifier*, not the pairing policy.
  * **arrival available but not chosen**: the pass saw a partner and picked a
    worse one, or rejected the right one on a gate. The fix is in the pairing.

For every ground-truth move this script prints the departure event, the
destination box mapped into canonical space, and every candidate arrival with
its kind, separation, and whether it overlaps that destination. That is enough
to tell which of the two failures is happening.

Run:

    python scripts/probe_pairing.py --seeds 1000 1007 1028 1035
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vision"))

from strata_core.config import PipelineConfig  # noqa: E402
from strata_core.evaluate import bbox_iou  # noqa: E402
from strata_core.pipeline import run_scan  # noqa: E402
from strata_core import synth  # noqa: E402


def box_sep(a, b) -> float:
    """Edge-to-edge separation of two boxes; 0 when they overlap."""
    gx = max(a[0] - (b[0] + b[2]), b[0] - (a[0] + a[2]), 0.0)
    gy = max(a[1] - (b[1] + b[3]), b[1] - (a[1] + a[3]), 0.0)
    return float(np.hypot(gx, gy))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1000, 1007, 1028, 1035])
    a = ap.parse_args()
    cfg = PipelineConfig()

    for seed in a.seeds:
        scene = synth.make_scene(seed=seed)
        res = run_scan([scene.baseline], [scene.rescan], config=cfg)
        src = (scene.baseline.shape[1], scene.baseline.shape[0])
        canon = res.diff.canonical_size
        sx, sy = canon[0] / src[0], canon[1] / src[1]

        gts = scene.ground_truth_events()
        moves = [g for g in gts if g["kind"] == "moved"]
        dests = [g for g in gts if g["kind"] == "moved_destination"]

        print(f"\n=== seed {seed}  ({len(res.events)} events reported)")
        for e in res.events:
            ev = e.evidence
            print(f"    REP {e.kind:14s} conf={e.confidence:.2f} "
                  f"gap={ev.get('presence_gap')} "
                  f"box=[{e.bbox[0]:.0f},{e.bbox[1]:.0f},{e.bbox[2]:.0f},{e.bbox[3]:.0f}]")

        for mv, dz in zip(moves, dests):
            dep_box = [mv["baseline_bbox"][0] * sx, mv["baseline_bbox"][1] * sy,
                       mv["baseline_bbox"][2] * sx, mv["baseline_bbox"][3] * sy]
            dst_box = [dz["baseline_bbox"][0] * sx, dz["baseline_bbox"][1] * sy,
                       dz["baseline_bbox"][2] * sx, dz["baseline_bbox"][3] * sy]
            print(f"    GT MOVE  departure={[round(v) for v in dep_box]}")
            print(f"             destination={[round(v) for v in dst_box]}")

            # Which reported event sits at the departure, and which at the destination?
            dep_hit = max(res.events, key=lambda e: bbox_iou(e.bbox, dep_box), default=None)
            dst_hit = max(res.events, key=lambda e: bbox_iou(e.bbox, dst_box), default=None)
            if dep_hit is not None:
                print(f"             departure reported as {dep_hit.kind:14s} "
                      f"iou={bbox_iou(dep_hit.bbox, dep_box):.3f}")
            if dst_hit is not None:
                print(f"             destination reported as {dst_hit.kind:14s} "
                      f"iou={bbox_iou(dst_hit.bbox, dst_box):.3f}  "
                      f"sep_from_dep={box_sep(dep_box, dst_hit.bbox):.1f}px")

            # Would the destination have been pairable, had it been `added`?
            print("             candidate arrivals (kind / sep-to-departure / iou-with-destination):")
            for e in res.events:
                sep = box_sep(dep_box, e.bbox)
                if sep > cfg.events.translation_max_px:
                    continue
                print(f"               {e.kind:14s} sep={sep:6.1f} "
                      f"iou_dst={bbox_iou(e.bbox, dst_box):.3f} "
                      f"conf={e.confidence:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
