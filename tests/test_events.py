"""Unit tests for event typing, the presence measure, and translation pairing.

The pairing rules under test were each derived from a measured failure on the
benchmark, so they are worth pinning down:

* an arrival is *not* required to be labelled ``added`` - no move destination
  in the benchmark ever was, so requiring it meant the pairing pass never ran;
* displacement must be bounded by the *centre-to-centre* distance as well as
  the box-edge separation, because a bloated arrival region keeps its edge near
  the departure while its centre is most of a scene away;
* a departure is allowed to remain ``missing``. A scene with one removal and
  one translation produces two departures and only one partner, so pairing
  must not fire merely because some arrival is in range.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vision"))

from strata_core.config import EventConfig  # noqa: E402
from strata_core.events import (  # noqa: E402
    ChangeEvent,
    _bbox_iou,
    _box_similarity,
    _pair_beats_alternatives,
    _presence,
    _resolve_translations,
    _size_compatible,
)


def make_event(
    kind,
    bbox,
    *,
    area=None,
    gap=0.0,
    eid="e1",
    confidence=0.9,
):
    """Build a ChangeEvent with the evidence fields the pairing pass reads."""
    x, y, w, h = bbox
    area = w * h if area is None else area
    return ChangeEvent(
        event_id=eid,
        kind=kind,
        confidence=confidence,
        regions=[0],
        bbox=bbox,
        centroid=(x + w / 2.0, y + h / 2.0),
        displacement_px=0.0,
        appearance_delta=0.0,
        summary="",
        evidence={"area_px": area, "presence_gap": gap},
    )


def textured_gray(h=200, w=200, seed=0):
    """A noise-textured grey image, so structure energy is non-zero."""
    rng = np.random.default_rng(seed)
    img = rng.integers(40, 210, size=(h, w), dtype=np.uint8)
    return img


class TestBboxIou:
    def test_identical(self):
        assert _bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)

    def test_disjoint(self):
        assert _bbox_iou((0, 0, 10, 10), (50, 50, 10, 10)) == 0.0

    def test_symmetric(self):
        a, b = (0, 0, 10, 10), (5, 5, 10, 10)
        assert _bbox_iou(a, b) == pytest.approx(_bbox_iou(b, a))


class TestBoxSimilarity:
    def test_identical_shape_scores_one(self):
        assert _box_similarity((0, 0, 40, 20), (100, 100, 40, 20)) == pytest.approx(1.0)

    def test_scaled_copy_still_scores_one(self):
        # Aspect ratio, not area: the same object seen closer is still the same
        # object.
        assert _box_similarity((0, 0, 40, 20), (0, 0, 80, 40)) == pytest.approx(1.0)

    def test_transposed_aspect_scores_low(self):
        assert _box_similarity((0, 0, 100, 10), (0, 0, 10, 100)) < 0.2

    def test_is_symmetric(self):
        a, b = (0, 0, 30, 10), (0, 0, 12, 30)
        assert _box_similarity(a, b) == pytest.approx(_box_similarity(b, a))

    def test_result_is_within_unit_interval(self):
        for wh in [(1, 100), (100, 1), (7, 7), (13, 29)]:
            s = _box_similarity((0, 0, wh[0], wh[1]), (0, 0, 50, 50))
            assert 0.0 <= s <= 1.0


class TestSizeCompatible:
    def test_equal_areas_pass(self):
        a = make_event("missing", (0, 0, 100, 100))
        b = make_event("added", (200, 0, 100, 100))
        assert _size_compatible(a, b, 0.35) is True

    def test_order_of_magnitude_difference_fails(self):
        a = make_event("missing", (0, 0, 100, 100), area=10000)
        b = make_event("added", (200, 0, 10, 10), area=100)
        assert _size_compatible(a, b, 0.35) is False

    def test_missing_area_is_treated_as_acceptable(self):
        # Absent evidence should not silently veto a pair; the other guards
        # still apply.
        a = make_event("missing", (0, 0, 100, 100), area=0)
        b = make_event("added", (200, 0, 100, 100), area=10000)
        assert _size_compatible(a, b, 0.35) is True


class TestPairBeatsAlternatives:
    """Pairing must explain the evidence better than 'a removal and an insertion'."""

    def test_comparable_object_sized_pair_is_preferred(self):
        dep = make_event("missing", (0, 0, 100, 100), area=10000, gap=-0.4)
        arr = make_event("added", (300, 0, 100, 100), area=10000, gap=0.3)
        assert _pair_beats_alternatives(dep, arr, 200.0, EventConfig()) is True

    def test_fragment_arrival_is_rejected(self):
        # A few hundred pixels is not an object; pairing with it would assert a
        # translation on the strength of noise.
        dep = make_event("missing", (0, 0, 100, 100), area=10000, gap=-0.4)
        arr = make_event("added", (300, 0, 20, 20), area=400, gap=0.3)
        assert _pair_beats_alternatives(dep, arr, 200.0, EventConfig()) is False

    def test_mismatched_scale_is_rejected(self):
        dep = make_event("missing", (0, 0, 100, 100), area=100000, gap=-0.4)
        arr = make_event("added", (300, 0, 30, 30), area=2000, gap=0.3)
        assert _pair_beats_alternatives(dep, arr, 200.0, EventConfig()) is False

    def test_separation_beyond_the_limit_is_rejected(self):
        dep = make_event("missing", (0, 0, 100, 100), area=10000, gap=-0.4)
        arr = make_event("added", (3000, 0, 100, 100), area=10000, gap=0.3)
        assert _pair_beats_alternatives(dep, arr, 5000.0, EventConfig()) is False


class TestPresence:
    def test_uniform_image_scores_near_the_pivot(self):
        flat = np.full((120, 120), 128, np.uint8)
        out = _presence(flat, (30, 30, 40, 40))
        # No structure anywhere: the ratio is undefined and reports the pivot
        # rather than inventing a direction.
        assert out["presence"] == pytest.approx(0.5)

    def test_tiny_box_reports_the_pivot(self):
        img = textured_gray()
        out = _presence(img, (10, 10, 2, 2))
        assert out["presence"] == pytest.approx(0.5)

    def test_a_highly_textured_box_inside_smooth_surround_scores_above_half(self):
        smooth = np.full((200, 200), 128, np.uint8)
        rng = np.random.default_rng(3)
        smooth[80:120, 80:120] = rng.integers(0, 255, size=(40, 40), dtype=np.uint8)
        out = _presence(smooth, (80, 80, 40, 40))
        assert out["presence"] > 0.5

    def test_a_smooth_box_inside_textured_surround_scores_below_half(self):
        rng = np.random.default_rng(4)
        busy = rng.integers(0, 255, size=(200, 200), dtype=np.uint8)
        busy[80:120, 80:120] = 128
        out = _presence(busy, (80, 80, 40, 40))
        assert out["presence"] < 0.5

    def test_result_is_bounded(self):
        img = textured_gray()
        out = _presence(img, (40, 40, 50, 50))
        assert 0.0 <= out["presence"] <= 1.0
        assert -1.0 <= out["contrast"] <= 1.0

    def test_change_mask_removes_pixels_from_the_surround_only(self):
        # A box surrounded by the change itself. Excluding the change must
        # raise the surround's smoothness (its reference becomes bare surface)
        # and therefore change the ratio, while the interior is untouched.
        rng = np.random.default_rng(5)
        img = np.full((200, 200), 128, np.uint8)
        # Busy band all around the box.
        img[40:160, 40:160] = rng.integers(0, 255, size=(120, 120), dtype=np.uint8)
        img[90:110, 90:110] = 128  # the box itself is smooth

        plain = _presence(img, (90, 90, 20, 20))
        mask = np.zeros((200, 200), np.uint8)
        mask[40:160, 40:160] = 255  # the whole busy band is "change"

        masked = _presence(img, (90, 90, 20, 20), change_mask=mask)
        # Interior is measured identically in both cases.
        assert masked["inside"] == pytest.approx(plain["inside"])
        # The reference differs, so the ratio must differ.
        assert masked["presence"] != pytest.approx(plain["presence"])

    def test_returns_the_documented_keys(self):
        img = textured_gray()
        out = _presence(img, (40, 40, 50, 50))
        assert set(out) == {"presence", "inside", "surround", "contrast"}


class TestResolveTranslations:
    """The pairing pass: promote a departure plus an arrival into one `moved`."""

    def test_a_lone_departure_stays_missing(self):
        # No arrival anywhere near: this is a removal, and must be reported as
        # one. Pairing is not obligatory.
        dep = make_event("missing", (0, 0, 100, 100), area=10000, gap=-0.4, eid="d1")
        out = _resolve_translations([dep], EventConfig())
        assert len(out) == 1
        assert out[0].kind == "missing"

    def test_departure_plus_object_sized_arrival_becomes_moved(self):
        dep = make_event("missing", (100, 100, 100, 100), area=10000, gap=-0.4, eid="d1")
        arr = make_event("added", (400, 100, 100, 100), area=10000, gap=0.3, eid="a1")
        out = _resolve_translations([dep, arr], EventConfig())
        kinds = sorted(e.kind for e in out)
        assert kinds == ["moved"]
        assert out[0].displacement_px == pytest.approx(300.0)

    def test_arrival_need_not_be_labelled_added(self):
        # The measured defect: no move destination in the benchmark was ever
        # classified `added`, so a kind filter here meant the pass never ran.
        dep = make_event("missing", (100, 100, 100, 100), area=10000, gap=-0.37, eid="d1")
        arr = make_event("state_changed", (400, 100, 100, 100), area=10000, gap=0.07, eid="a1")
        out = _resolve_translations([dep, arr], EventConfig())
        assert len(out) == 1
        assert out[0].kind == "moved"
        assert out[0].evidence["arrival_initial_kind"] == "state_changed"

    def test_arrival_with_negligible_evidence_does_not_pair(self):
        # Measured: mis-paired arrivals showed gaps of 0.013-0.022 while real
        # ones showed 0.065-0.45. A gap this small is not an arrival.
        dep = make_event("missing", (100, 100, 100, 100), area=10000, gap=-0.37, eid="d1")
        arr = make_event("added", (400, 100, 100, 100), area=10000, gap=0.01, eid="a1")
        out = _resolve_translations([dep, arr], EventConfig())
        assert sorted(e.kind for e in out) == ["added", "missing"]

    def test_a_missing_event_is_never_used_as_an_arrival(self):
        dep1 = make_event("missing", (100, 100, 100, 100), area=10000, gap=-0.4, eid="d1")
        dep2 = make_event("missing", (400, 100, 100, 100), area=10000, gap=-0.4, eid="d2")
        out = _resolve_translations([dep1, dep2], EventConfig())
        assert all(e.kind == "missing" for e in out)
        assert len(out) == 2

    def test_centroid_distance_bounds_a_bloated_arrival(self):
        # Box edges are close but the centres are far apart, which is what an
        # oversized merged arrival looks like. Measured: every spurious pair in
        # the benchmark had a centroid distance of 451-520px.
        dep = make_event("missing", (0, 0, 120, 120), area=14400, gap=-0.4, eid="d1")
        arr = make_event("added", (500, 0, 200, 200), area=40000, gap=0.3, eid="a1")
        # Edge separation is 380 (within the 420 limit); centroid distance is
        # ~541 (beyond it).
        out = _resolve_translations([dep, arr], EventConfig())
        assert sorted(e.kind for e in out) == ["added", "missing"]

    def test_one_arrival_serves_only_the_nearest_departure(self):
        # Two departures compete for a single arrival. Exactly one pair may be
        # formed; the other departure is a removal and must survive as such.
        dep_a = make_event("missing", (0, 0, 100, 100), area=10000, gap=-0.4, eid="d1")
        dep_b = make_event("missing", (300, 0, 100, 100), area=10000, gap=-0.4, eid="d2")
        arr = make_event("added", (170, 0, 100, 100), area=10000, gap=0.3, eid="a1")
        out = _resolve_translations([dep_a, dep_b, arr], EventConfig())
        kinds = sorted(e.kind for e in out)
        assert kinds == ["missing", "moved"]
        moved = next(e for e in out if e.kind == "moved")
        # The nearer departure (dep_b) is the one that pairs.
        assert moved.event_id == "d2"

    def test_no_events_returns_no_events(self):
        assert _resolve_translations([], EventConfig()) == []

    def test_no_departures_leaves_events_untouched(self):
        arr = make_event("added", (100, 100, 100, 100), area=10000, gap=0.3, eid="a1")
        out = _resolve_translations([arr], EventConfig())
        assert len(out) == 1
        assert out[0].kind == "added"

    def test_moved_event_records_both_halves_as_evidence(self):
        dep = make_event("missing", (100, 100, 100, 100), area=10000, gap=-0.4, eid="d1")
        arr = make_event("added", (400, 100, 100, 100), area=10000, gap=0.3, eid="a1")
        out = _resolve_translations([dep, arr], EventConfig())
        ev = out[0].evidence
        assert ev["departure_presence_gap"] == pytest.approx(0.4)
        assert ev["arrival_presence_gap"] == pytest.approx(0.3)
        assert "arrival_bbox" in ev
        assert 0.0 <= out[0].confidence <= 1.0

    def test_consumed_arrival_is_not_reported_separately(self):
        dep = make_event("missing", (100, 100, 100, 100), area=10000, gap=-0.4, eid="d1")
        arr = make_event("added", (400, 100, 100, 100), area=10000, gap=0.3, eid="a1")
        out = _resolve_translations([dep, arr], EventConfig())
        assert not any(e.kind == "added" for e in out)


class TestChangeEventSerialisation:
    def test_to_dict_has_the_expected_shape(self):
        e = make_event("moved", (0, 0, 10, 20), gap=0.3)
        d = e.to_dict()
        assert d["kind"] == "moved"
        assert d["bbox"] == [0, 0, 10, 20]
        assert isinstance(d["centroid"], list) and len(d["centroid"]) == 2
        assert isinstance(d["evidence"], dict)

    def test_to_dict_is_json_safe(self):
        import json

        e = make_event("added", (0, 0, 10, 20), gap=0.3)
        # Must not raise: ints and floats only.
        json.dumps(e.to_dict())
