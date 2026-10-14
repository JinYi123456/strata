"""Is a translation's destination ever typed ``added``?

The pairing pass promotes a ``missing`` departure plus a nearby ``added``
arrival into one ``moved`` event. That design assumes a moved object's
destination is classified ``added`` - new structure where there was none.

It is not obvious that it should be. ``added`` is emitted when the presence gap
clears ``+presence_gap_min``, and presence is comparative: a box reads as an
object when its interior is more uniform than its surround. An object that has
just moved *onto a location that already looked like background* satisfies
that, so the gap is large and the label is ``added``. But an object that moved
onto a location where something already stood, or that visually resembles its
new surround, has a near-zero gap and is labelled ``state_changed`` - the label
for "changed in place", which is precisely the wrong reading.

This script measures the distribution directly: for every ground-truth
translation it reports the presence gap and the assigned kind at both the
vacated location and the destination. If destinations routinely come back as
``state_changed`` rather than ``added``, then keying the pairing pass on the
arrival's kind discards most true pairings, and the kind filter - not the
thresholds - is the thing to change.

Run:

    python scripts/probe_arrival_kind.py --seeds 1000 1007 1014 1021 1028 1035
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vision"))

from strata_core.config import PipelineConfig  # noqa: E402
from strata_core.evaluate import bbox_iou  # noqa: E402
from strata_core.pipeline import run_scan  # noqa: E402
from strata_core import synth  # noqa: E402

MATCH_IOU = 0.30


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[1000, 1007, 1014, 1021, 1028, 1035])
    a = ap.parse_args()
    cfg = PipelineConfig()

    dep_kinds = collections.Counter()
    dst_kinds = collections.Counter()
    rows = []

    for seed in a.seeds:
        scene = synth.make_scene(seed=seed)
        res = run_scan([scene.baseline], [scene.rescan], config=cfg)
        src = (scene.baseline.shape[1], scene.baseline.shape[0])
        canon = res.diff.canonical_size
        sx, sy = canon[0] / src[0], canon[1] / src[1]

        gts = scene.ground_truth_events()
        moves = [g for g in gts if g["kind"] == "moved"]
        dests = [g for g in gts if g["kind"] == "moved_destination"]

        for mv, dz in zip(moves, dests):
            dep_box = (mv["baseline_bbox"][0] * sx, mv["baseline_bbox"][1] * sy,
                       mv["baseline_bbox"][2] * sx, mv["baseline_bbox"][3] * sy)
            dst_box = (dz["baseline_bbox"][0] * sx, dz["baseline_bbox"][1] * sy,
                       dz["baseline_bbox"][2] * sx, dz["baseline_bbox"][3] * sy)

            dep_hit = max(res.events, key=lambda e: bbox_iou(e.bbox, dep_box), default=None)
            dst_hit = max(res.events, key=lambda e: bbox_iou(e.bbox, dst_box), default=None)

            dep_iou = bbox_iou(dep_hit.bbox, dep_box) if dep_hit else 0.0
            dst_iou = bbox_iou(dst_hit.bbox, dst_box) if dst_hit else 0.0

            dep_ok = dep_hit.kind if dep_iou >= MATCH_IOU else None
            dst_ok = dst_hit.kind if dst_iou >= MATCH_IOU else None
            dep_gap = dep_hit.evidence.get("presence_gap") if dep_iou >= MATCH_IOU else None
            dst_gap = dst_hit.evidence.get("presence_gap") if dst_iou >= MATCH_IOU else None

            if dep_ok:
                dep_kinds[dep_ok] += 1
            if dst_ok:
                dst_kinds[dst_ok] += 1
            rows.append((seed, dep_ok, dep_gap, dst_ok, dst_gap))

    print(f"{'seed':>6}  {'departure':>14} {'gap':>8}   {'destination':>16} {'gap':>8}")
    print("-" * 62)
    for seed, dk, dg, sk, sg in rows:
        print(f"{seed:>6}  {str(dk):>14} {str(dg):>8}   {str(sk):>16} {str(sg):>8}")

    print()
    print("departure kinds  :", dict(dep_kinds))
    print("destination kinds:", dict(dst_kinds))
    n = len(rows)
    added = dst_kinds.get("added", 0)
    print()
    print(f"destinations typed 'added'      : {added}/{n}")
    print(f"destinations typed 'state_changed': {dst_kinds.get('state_changed', 0)}/{n}")
    print(f"destinations not localised at all : "
          f"{sum(1 for _, _, _, sk, _ in rows if sk is None)}/{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
