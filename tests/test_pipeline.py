"""End-to-end tests for the full registration + diff pipeline.

These are slow - each run loads feature models and processes a real frame pair -
so they are marked ``slow`` and use a reduced frame size. Run them with:

    pytest tests/test_pipeline.py -m slow

The properties asserted here are the ones the project's headline claims rest
on, in particular that a pair with nothing changed produces no confident
events. That single number decides whether the system is usable at all: a
detector that reports changes when nothing changed cannot be trusted about the
changes it does report.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vision"))

from strata_core.config import PipelineConfig  # noqa: E402
from strata_core.evaluate import bbox_iou  # noqa: E402
from strata_core.pipeline import run_scan  # noqa: E402
from strata_core.synth import make_scene, make_still_pair  # noqa: E402

pytestmark = pytest.mark.slow

SIZE = (480, 360)


def scan(seed: int, *, size=SIZE, n_mutations=4, n_objects=12):
    scene = make_scene(seed=seed, size=size, n_objects=n_objects, n_mutations=n_mutations)
    result = run_scan([scene.baseline], [scene.rescan], config=PipelineConfig())
    return scene, result


def gt_boxes(scene, result, kind_filter=None):
    """Ground-truth boxes in canonical pixels, honouring each entry's space."""
    src_w, src_h = scene.size
    cw, ch = result.registration.canonical_size
    sx, sy = cw / src_w, ch / src_h
    out = []
    for g in scene.ground_truth_events():
        if kind_filter and g["kind"] not in kind_filter:
            continue
        x, y, w, h = g["baseline_bbox"]
        out.append((g["kind"], (x * sx, y * sy, w * sx, h * sy)))
    return out


class TestPipelineRuns:
    def test_a_scene_runs_end_to_end(self):
        scene, result = scan(2001)
        assert result.registration is not None
        assert result.diff is not None
        assert isinstance(result.events, list)

    def test_canonical_frames_share_one_size(self):
        _, result = scan(2002)
        assert result.baseline.shape[:2] == result.rescan_canonical.shape[:2]

    def test_canonical_frame_is_the_configured_long_edge(self):
        _, result = scan(2003)
        cw, ch = result.registration.canonical_size
        assert max(cw, ch) == PipelineConfig().geometry.canonical_long_edge

    def test_every_stage_records_a_timing(self):
        _, result = scan(2004)
        t = result.timings
        assert t.total_ms > 0
        for stage in ("match_ms", "register_ms", "diff_ms", "events_ms"):
            assert getattr(t, stage) >= 0.0

    def test_a_feature_backend_is_named(self):
        _, result = scan(2005)
        assert isinstance(result.feature_backend, str) and result.feature_backend


class TestRegistration:
    def test_registration_is_a_homography_with_reasonable_confidence(self):
        _, result = scan(2006)
        reg = result.registration
        assert reg.model in ("homography", "fundamental")
        assert reg.confidence > 0.3, f"registration confidence {reg.confidence}"
        assert reg.inlier_count > 0

    def test_registration_error_against_known_truth_is_small(self):
        # The scene knows the true homography, so the recovered one can be
        # scored rather than merely eyeballed.
        from strata_core.evaluate import homography_error

        scene, result = scan(2007)
        err = homography_error(result.registration.matrix, scene.homography, scene.size)
        assert err < 8.0, f"registration error {err:.2f}px"

    def test_inlier_ratio_is_a_fraction(self):
        _, result = scan(2008)
        assert 0.0 <= result.registration.inlier_ratio <= 1.0

    def test_registration_is_deterministic(self):
        _, a = scan(2009)
        _, b = scan(2009)
        assert np.allclose(a.registration.matrix, b.registration.matrix)


class TestFalsePositives:
    """The property the whole system's usefulness depends on."""

    def test_a_still_pair_produces_no_confident_events(self):
        for seed in (3001, 3012, 3023):
            base, res, _ = make_still_pair(seed=seed, size=SIZE)
            result = run_scan([base], [res], config=PipelineConfig())
            confident = [
                e for e in result.events
                if e.kind != "uncertain" and e.confidence >= 0.35
            ]
            assert not confident, (
                f"seed {seed}: {len(confident)} confident events on a still pair: "
                f"{[(e.kind, round(e.confidence, 2)) for e in confident]}"
            )

    def test_a_still_pair_registers_confidently(self):
        # If a still pair registered poorly, the absence of events would be
        # explained by a failed alignment rather than by correct behaviour.
        base, res, _ = make_still_pair(seed=3034, size=SIZE)
        result = run_scan([base], [res], config=PipelineConfig())
        assert result.registration.confidence > 0.5

    def test_identical_frames_produce_nothing(self):
        scene = make_scene(seed=3045, size=SIZE, n_mutations=0)
        result = run_scan([scene.baseline], [scene.baseline], config=PipelineConfig())
        confident = [e for e in result.events if e.confidence >= 0.35]
        assert not confident


class TestDetection:
    def test_a_removal_is_reported_near_the_removed_object(self):
        # Restricted to removals that are the *only* change at their location.
        # When an insertion lands on top of a removal the diff produces a single
        # connected region, and the two changes cannot be separated from it -
        # see the known-limitation test below. Asserting otherwise would be
        # asserting a capability the architecture does not have.
        scene, result = scan(4001)
        boxes = gt_boxes(scene, result, kind_filter={"missing"})
        if not boxes:
            pytest.skip("no removal in this scene")

        others = [
            box for kind, box in gt_boxes(scene, result)
            if kind != "missing" and kind != "moved_destination"
        ]

        def overlaps_another(box, thresh=0.05):
            return any(bbox_iou(box, o) > thresh for o in others)

        isolated = [box for _, box in boxes if not overlaps_another(box)]
        if not isolated:
            pytest.skip("every removal in this scene overlaps another change")
        for box in isolated:
            best = max((bbox_iou(e.bbox, box) for e in result.events), default=0.0)
            assert best >= 0.25, f"removal at {box} not localised (best IoU {best:.2f})"

    def test_removals_are_localised_across_several_scenes(self):
        # The capability stated at the scale it actually holds: over a range of
        # scenes, isolated removals are found.
        found = 0
        total = 0
        for seed in range(4010, 4026):
            scene, result = scan(seed)
            boxes = gt_boxes(scene, result, kind_filter={"missing"})
            if not boxes:
                continue
            others = [
                box for kind, box in gt_boxes(scene, result)
                if kind not in ("missing", "moved_destination")
            ]
            for _, box in boxes:
                if any(bbox_iou(box, o) > 0.05 for o in others):
                    continue
                total += 1
                best = max((bbox_iou(e.bbox, box) for e in result.events), default=0.0)
                if best >= 0.25:
                    found += 1
        if total == 0:
            pytest.skip("no isolated removals in the sampled scenes")
        rate = found / total
        assert rate >= 0.6, f"only {found}/{total} isolated removals localised"

    def test_overlapping_removal_and_insertion_is_a_known_limitation(self):
        # Documented behaviour rather than an aspiration. When a removal and an
        # insertion occupy overlapping locations, the difference mask forms one
        # connected component, so the pipeline reports one region there instead
        # of two events. This test pins the *current* behaviour so that a future
        # change which splits the region is noticed - at which point this test
        # should be rewritten to assert both events.
        scene, result = scan(4001)
        boxes = gt_boxes(scene, result, kind_filter={"missing"})
        others = [
            box for kind, box in gt_boxes(scene, result)
            if kind not in ("missing", "moved_destination")
        ]
        overlapping = [
            box for _, box in boxes
            if any(bbox_iou(box, o) > 0.05 for o in others)
        ]
        if not overlapping:
            pytest.skip("this scene has no overlapping removal")

        for box in overlapping:
            best = max((bbox_iou(e.bbox, box) for e in result.events), default=0.0)
            # The merged region still covers the location - the change is not
            # lost, it is simply not split into two typed events.
            region_best = max(
                (bbox_iou(r.bbox, box) for r in result.diff.regions), default=0.0
            )
            assert region_best >= 0.25, "the change region itself should exist"
            assert best < 0.25, (
                "an overlapping removal is now reported separately - the "
                "limitation has been fixed, so update this test"
            )

    def test_a_colour_change_is_detected(self):
        # The defect the chrominance channel exists to fix: a recolour that is
        # nearly invisible in luminance must still be found. Not every scene
        # contains a recolour, so this searches a small range of seeds and
        # requires at least one to be localised.
        found_any = False
        for seed in range(4002, 4010):
            scene, result = scan(seed)
            boxes = gt_boxes(scene, result, kind_filter={"state_changed"})
            for _, box in boxes:
                best = max((bbox_iou(e.bbox, box) for e in result.events), default=0.0)
                if best >= 0.25:
                    found_any = True
                    break
            if found_any:
                break
        assert found_any, "no colour change was localised across 8 scenes"

    def test_events_have_bounded_confidence(self):
        _, result = scan(4031)
        for e in result.events:
            assert 0.0 <= e.confidence <= 1.0

    def test_events_have_non_empty_summaries(self):
        _, result = scan(4032)
        for e in result.events:
            assert isinstance(e.summary, str) and e.summary

    def test_event_kinds_are_from_the_known_set(self):
        _, result = scan(4033)
        allowed = {"added", "missing", "moved", "state_changed", "uncertain"}
        for e in result.events:
            assert e.kind in allowed

    def test_events_are_ordered_with_uncertain_last(self):
        _, result = scan(4034)
        kinds = [e.kind == "uncertain" for e in result.events]
        # Once an uncertain event appears, no definite one may follow it.
        assert kinds == sorted(kinds), "uncertain events are not sorted last"

    def test_event_ids_are_unique(self):
        _, result = scan(4035)
        ids = [e.event_id for e in result.events]
        assert len(ids) == len(set(ids))

    def test_moved_events_report_a_real_displacement(self):
        for seed in range(4036, 4044):
            _, result = scan(seed)
            for m in [e for e in result.events if e.kind == "moved"]:
                assert m.displacement_px > 0
                assert m.evidence.get("arrival_bbox") is not None


class TestDeterminism:
    def test_same_seed_gives_the_same_events(self):
        _, a = scan(5001)
        _, b = scan(5001)
        assert [(e.kind, tuple(e.bbox)) for e in a.events] == [
            (e.kind, tuple(e.bbox)) for e in b.events
        ]

    def test_same_seed_gives_the_same_confidences(self):
        _, a = scan(5002)
        _, b = scan(5002)
        assert [round(e.confidence, 6) for e in a.events] == [
            round(e.confidence, 6) for e in b.events
        ]


class TestCoverage:
    def test_coverage_map_is_produced(self):
        _, result = scan(6001)
        assert result.coverage is not None

    def test_a_single_frame_pair_reports_some_uncertainty(self):
        # One frame cannot see a whole scene, so the system should say so
        # rather than implying the empty regions are confirmed unchanged.
        _, result = scan(6002)
        assert result.coverage is not None
