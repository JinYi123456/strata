"""Unit tests for the pure geometry and scoring helpers.

These are deliberately dependency-light: no models, no pipeline, no images
larger than a few pixels. Everything here is arithmetic that the rest of the
system trusts, so a failure means a real defect rather than a flaky model load.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vision"))

from strata_core.evaluate import (  # noqa: E402
    ChangeMetrics,
    bbox_iou,
    homography_error,
    score_event_kinds,
    score_regions,
)


class TestBboxIou:
    def test_identical_boxes_score_one(self):
        assert bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)

    def test_disjoint_boxes_score_zero(self):
        assert bbox_iou([0, 0, 10, 10], [20, 20, 10, 10]) == 0.0

    def test_touching_boxes_score_zero(self):
        # Sharing an edge is not overlapping: the intersection has zero area.
        assert bbox_iou([0, 0, 10, 10], [10, 0, 10, 10]) == 0.0

    def test_half_overlap_matches_hand_computation(self):
        # 10x10 at origin and 10x10 shifted 5 right: intersection 5x10 = 50,
        # union 100 + 100 - 50 = 150.
        assert bbox_iou([0, 0, 10, 10], [5, 0, 10, 10]) == pytest.approx(50 / 150)

    def test_is_symmetric(self):
        a, b = [0, 0, 10, 10], [5, 5, 10, 10]
        assert bbox_iou(a, b) == pytest.approx(bbox_iou(b, a))

    def test_contained_box_scores_by_area_ratio(self):
        # A 5x5 inside a 10x10: intersection 25, union 100.
        assert bbox_iou([0, 0, 10, 10], [2, 2, 5, 5]) == pytest.approx(0.25)


class TestChangeMetrics:
    def test_precision_recall_f1_on_a_clean_result(self):
        m = ChangeMetrics(true_positives=4, false_positives=0, false_negatives=0)
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1 == 1.0

    def test_empty_result_does_not_divide_by_zero(self):
        m = ChangeMetrics()
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 == 0.0

    def test_perfect_precision_with_misses(self):
        # This is the shape the corrected harness produces for a detector that
        # reports only true changes: no false positives, some misses.
        m = ChangeMetrics(true_positives=3, false_positives=0, false_negatives=1)
        assert m.precision == 1.0
        assert m.recall == pytest.approx(0.75)
        assert m.f1 == pytest.approx(2 * 1.0 * 0.75 / 1.75)


class TestScoreRegions:
    def test_exact_match_is_a_true_positive(self):
        m = score_regions([[0, 0, 10, 10]], [[0, 0, 10, 10]], iou_threshold=0.5)
        assert m.true_positives == 1
        assert m.false_positives == 0
        assert m.false_negatives == 0

    def test_extra_detection_is_a_false_positive(self):
        m = score_regions(
            [[0, 0, 10, 10], [50, 50, 10, 10]], [[0, 0, 10, 10]], iou_threshold=0.5
        )
        assert m.true_positives == 1
        assert m.false_positives == 1

    def test_missed_truth_is_a_false_negative(self):
        m = score_regions([], [[0, 0, 10, 10]], iou_threshold=0.5)
        assert m.false_negatives == 1

    def test_each_truth_consumes_at_most_one_detection(self):
        # Two detections both overlapping one truth must not both count as
        # true positives - that would let a detector inflate recall by
        # reporting the same change repeatedly.
        m = score_regions(
            [[0, 0, 10, 10], [1, 1, 10, 10]], [[0, 0, 10, 10]], iou_threshold=0.5
        )
        assert m.true_positives == 1
        assert m.false_positives == 1

    def test_threshold_is_respected(self):
        # IoU 50/150 = 0.333, below a 0.5 threshold but above 0.25.
        det, tru = [[0, 0, 10, 10]], [[5, 0, 10, 10]]
        assert score_regions(det, tru, iou_threshold=0.5).true_positives == 0
        assert score_regions(det, tru, iou_threshold=0.25).true_positives == 1


class TestScoreEventKinds:
    def _rep(self, kind, bbox):
        return {"kind": kind, "bbox": bbox}

    def _exp(self, kind, bbox, oid="x"):
        return {"kind": kind, "baseline_bbox": bbox, "object_id": oid}

    def test_correct_kind_scores_one(self):
        acc, details = score_event_kinds(
            [self._rep("missing", [0, 0, 10, 10])],
            [self._exp("missing", [0, 0, 10, 10])],
            iou_threshold=0.5,
        )
        assert acc == 1.0
        assert details[0]["correct"] is True

    def test_wrong_kind_scores_zero(self):
        acc, _ = score_event_kinds(
            [self._rep("added", [0, 0, 10, 10])],
            [self._exp("missing", [0, 0, 10, 10])],
            iou_threshold=0.5,
        )
        assert acc == 0.0

    def test_unmatched_expected_event_is_not_correct(self):
        acc, details = score_event_kinds(
            [], [self._exp("missing", [0, 0, 10, 10])], iou_threshold=0.5
        )
        assert acc == 0.0
        assert details[0]["reported_kind"] is None

    def test_accepted_kinds_widens_the_match(self):
        acc, details = score_event_kinds(
            [self._rep("moved", [0, 0, 10, 10])],
            [self._exp("added", [0, 0, 10, 10])],
            iou_threshold=0.5,
            accepted_kinds={"added": ("moved",)},
        )
        assert acc == 1.0
        assert details[0]["accepted"] == ["added", "moved"]

    def test_accepted_kinds_is_opt_in_per_kind(self):
        # Widening "added" must not accidentally widen "missing".
        acc, _ = score_event_kinds(
            [self._rep("added", [0, 0, 10, 10])],
            [self._exp("missing", [0, 0, 10, 10])],
            iou_threshold=0.5,
            accepted_kinds={"added": ("moved",)},
        )
        assert acc == 0.0

    def test_extra_reported_events_do_not_lower_accuracy(self):
        # Kind accuracy is measured over expected events; surplus detections
        # are already charged as false positives by the region scorer.
        acc, _ = score_event_kinds(
            [self._rep("missing", [0, 0, 10, 10]), self._rep("added", [90, 90, 10, 10])],
            [self._exp("missing", [0, 0, 10, 10])],
            iou_threshold=0.5,
        )
        assert acc == 1.0

    def test_two_expected_events_do_not_share_one_detection(self):
        # Two expected events over the same location, one detection: only one
        # can be satisfied. Without a used-set this would score 1.0 twice.
        acc, _ = score_event_kinds(
            [self._rep("missing", [0, 0, 10, 10])],
            [self._exp("missing", [0, 0, 10, 10], "a"),
             self._exp("missing", [0, 0, 10, 10], "b")],
            iou_threshold=0.5,
        )
        assert acc == pytest.approx(0.5)

    def test_empty_expected_set_is_vacuously_perfect(self):
        acc, details = score_event_kinds([], [], iou_threshold=0.5)
        assert acc == 1.0
        assert details == []


class TestHomographyError:
    def test_identity_has_zero_error(self):
        H = np.eye(3)
        assert homography_error(H, H, (100, 100)) == pytest.approx(0.0, abs=1e-9)

    def test_pure_translation_error_equals_the_shift(self):
        # A translation of 5px in x moves every corner by 5px.
        H_true = np.eye(3)
        H_est = np.eye(3)
        H_est[0, 2] = 5.0
        assert homography_error(H_est, H_true, (100, 100)) == pytest.approx(5.0, abs=1e-6)

    def test_pure_translation_is_size_invariant(self):
        # The third column of the matrix is a raw pixel translation, so the
        # same value moves every corner by the same number of pixels whatever
        # the image size. This is the property that makes the metric directly
        # readable as "how far off, in pixels".
        H_true = np.eye(3)
        H_est = np.eye(3)
        H_est[0, 2] = 0.01
        assert homography_error(H_est, H_true, (100, 100)) == pytest.approx(0.01)
        assert homography_error(H_est, H_true, (1000, 1000)) == pytest.approx(0.01)

    def test_projective_term_does_scale_with_image_size(self):
        # A perspective term acts on normalised coordinates internally, so the
        # same coefficient displaces corners further in a larger image. A
        # metric that missed this would under-report error on wide inputs.
        H_true = np.eye(3)
        H_est = np.eye(3)
        H_est[2, 0] = 1e-4
        small = homography_error(H_est, H_true, (100, 100))
        large = homography_error(H_est, H_true, (1000, 1000))
        assert large > small

    def test_is_symmetric_in_its_arguments(self):
        A = np.eye(3)
        A[0, 2] = 3.0
        B = np.eye(3)
        assert homography_error(A, B, (200, 150)) == pytest.approx(
            homography_error(B, A, (200, 150))
        )


class TestAggregate:
    def test_means_are_computed_across_cases(self):
        from strata_core.evaluate import RunResult, aggregate

        def mk(reg_err, conf, tp, fp, fn, kind):
            return RunResult(
                name="x",
                registration_error_px=reg_err,
                registration_confidence=conf,
                registration_model="homography",
                inlier_ratio=0.9,
                detection=ChangeMetrics(
                    true_positives=tp, false_positives=fp, false_negatives=fn
                ),
                event_kind_accuracy=kind,
                expected_events=tp + fn,
                reported_events=tp + fp,
            )

        agg = aggregate([mk(1.0, 0.9, 4, 0, 0, 1.0), mk(2.0, 0.8, 3, 0, 1, 0.5)])
        assert agg["cases"] == 2
        assert agg["mean_registration_error_px"] == pytest.approx(1.5)
        assert agg["mean_registration_confidence"] == pytest.approx(0.85)
        assert agg["mean_event_kind_accuracy"] == pytest.approx(0.75)
        assert agg["total_true_positives"] == 7

    def test_empty_input_returns_empty_dict(self):
        from strata_core.evaluate import aggregate

        assert aggregate([]) == {}
