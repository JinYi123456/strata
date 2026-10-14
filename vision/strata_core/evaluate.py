"""Evaluation metrics and harness.

Measurement is the difference between a demo and an engineering result. This
module scores a STRATA run against known ground truth:

* **Registration error** - mean corner error between the recovered homography
  and the true one, in canonical pixels.
* **Change detection** - region-level precision / recall / F1 against the
  ground-truth mutated boxes, at a configurable IoU threshold.
* **Event typing** - accuracy of the reported event kind against the expected
  kind, for changes that were detected at all.
* **False positives on a still pair** - how many events fire when nothing
  changed, which is the number that decides whether the system is trustworthy
  in the field.

Everything is returned as plain dicts so the same numbers can be written to
JSON, rendered in a report table, and asserted in tests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np

log = logging.getLogger("strata.evaluate")


@dataclass
class ChangeMetrics:
    """Precision/recall of detected regions against ground-truth boxes."""

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    matched_pairs: list[tuple[int, int, float]] = field(default_factory=list)

    @property
    def precision(self) -> float:
        d = self.true_positives + self.false_positives
        return float(self.true_positives) / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.true_positives + self.false_negatives
        return float(self.true_positives) / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def to_dict(self) -> dict:
        return {
            "true_positives": int(self.true_positives),
            "false_positives": int(self.false_positives),
            "false_negatives": int(self.false_negatives),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def score_regions(
    detected_boxes: Sequence[Sequence[float]],
    truth_boxes: Sequence[Sequence[float]],
    *,
    iou_threshold: float = 0.30,
) -> ChangeMetrics:
    """Greedy IoU matching of detected regions to ground-truth boxes."""
    m = ChangeMetrics()
    truth_remaining = list(range(len(truth_boxes)))
    for di, db in enumerate(detected_boxes):
        best_j, best_iou = -1, 0.0
        for j in truth_remaining:
            iou = bbox_iou(db, truth_boxes[j])
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0 and best_iou >= iou_threshold:
            m.true_positives += 1
            m.matched_pairs.append((di, best_j, best_iou))
            truth_remaining.remove(best_j)
        else:
            m.false_positives += 1
    m.false_negatives = len(truth_remaining)
    return m


def homography_error(H_est: np.ndarray, H_true: np.ndarray, size: tuple[int, int]) -> float:
    """Mean corner displacement between two homographies, in pixels."""
    if H_est is None or H_true is None:
        return float("inf")
    w, h = size
    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float64).reshape(-1, 1, 2)
    a = cv2.perspectiveTransform(corners, H_est).reshape(-1, 2)
    b = cv2.perspectiveTransform(corners, H_true).reshape(-1, 2)
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


@dataclass
class RunResult:
    """The full scored outcome of one synthetic evaluation case."""

    name: str
    registration_error_px: float
    registration_confidence: float
    registration_model: str
    inlier_ratio: float
    detection: ChangeMetrics
    event_kind_accuracy: float
    expected_events: int
    reported_events: int
    per_event: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "registration_error_px": round(float(self.registration_error_px), 3),
            "registration_confidence": round(float(self.registration_confidence), 4),
            "registration_model": self.registration_model,
            "inlier_ratio": round(float(self.inlier_ratio), 4),
            "detection": self.detection.to_dict(),
            "event_kind_accuracy": round(float(self.event_kind_accuracy), 4),
            "expected_events": int(self.expected_events),
            "reported_events": int(self.reported_events),
            "per_event": self.per_event,
            "notes": self.notes,
        }


def score_event_kinds(
    reported: Sequence[dict],
    expected: Sequence[dict],
    *,
    iou_threshold: float = 0.30,
    accepted_kinds: dict[str, tuple[str, ...]] | None = None,
) -> tuple[float, list[dict]]:
    """Match reported events to expected ones spatially and compare kinds.

    Returns ``(accuracy, details)`` where accuracy is the fraction of *expected*
    events whose kind was reported correctly. Extra reported events do not
    lower accuracy here - they are already counted as false positives by the
    region scorer, and double-counting them would make the number unreadable.

    ``accepted_kinds`` maps an expected kind to the set of reported kinds that
    also count as correct. This exists for genuinely ambiguous cases rather
    than to inflate the score. The clearest one is a translation whose
    destination lands close to an unrelated insertion: the arrival is a single
    region of evidence, and whether the system explains it as "this object
    moved here" or "a new object appeared here" depends on whether it found the
    matching departure. Both readings are defensible from the pixels, and the
    distinction between them is scored separately, by the translation-pairing
    test, where it belongs.
    """
    details: list[dict] = []
    if not expected:
        return 1.0, details

    used = set()
    correct = 0
    for exp in expected:
        exp_box = exp["baseline_bbox"]
        best_i, best_iou = -1, 0.0
        for i, rep in enumerate(reported):
            if i in used:
                continue
            iou = bbox_iou(rep["bbox"], exp_box)
            if iou > best_iou:
                best_iou, best_i = iou, i
        if best_i >= 0 and best_iou >= iou_threshold:
            used.add(best_i)
            reported_kind = reported[best_i]["kind"]
            allowed = (exp["kind"],) + (accepted_kinds or {}).get(exp["kind"], ())
            ok = reported_kind in allowed
            correct += int(ok)
            details.append({
                "object_id": exp["object_id"],
                "expected_kind": exp["kind"],
                "reported_kind": reported_kind,
                "iou": round(best_iou, 3),
                "correct": ok,
                "accepted": list(allowed),
            })
        else:
            details.append({
                "object_id": exp["object_id"],
                "expected_kind": exp["kind"],
                "reported_kind": None,
                "iou": 0.0,
                "correct": False,
            })
    return correct / float(len(expected)), details


def aggregate(results: Sequence[RunResult]) -> dict:
    """Roll several cases into headline numbers for a report table."""
    if not results:
        return {}
    n = len(results)
    return {
        "cases": n,
        "mean_registration_error_px": round(
            float(np.mean([r.registration_error_px for r in results])), 3
        ),
        "mean_registration_confidence": round(
            float(np.mean([r.registration_confidence for r in results])), 4
        ),
        "mean_precision": round(float(np.mean([r.detection.precision for r in results])), 4),
        "mean_recall": round(float(np.mean([r.detection.recall for r in results])), 4),
        "mean_f1": round(float(np.mean([r.detection.f1 for r in results])), 4),
        "mean_event_kind_accuracy": round(
            float(np.mean([r.event_kind_accuracy for r in results])), 4
        ),
        "total_false_positives": int(sum(r.detection.false_positives for r in results)),
        "total_false_negatives": int(sum(r.detection.false_negatives for r in results)),
        "total_true_positives": int(sum(r.detection.true_positives for r in results)),
    }
