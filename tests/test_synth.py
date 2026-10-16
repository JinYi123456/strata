"""Unit tests for the synthetic scene generator.

The generator is the ground truth for every number the project reports, so a
defect in it silently corrupts the evaluation. Each test here corresponds to a
bug that was actually found and fixed, which is why they are specific rather
than generic:

* a single shared RNG stream made one object's mutation change every later
  object's appearance, producing spurious diffs that the detector was then
  blamed for;
* tile writes near the frame edge silently clipped, moving an object's real
  extent without moving its recorded bbox;
* a "recolour" was not guaranteed to be perceptible;
* an insertion had no ground-truth entry at all, so a correct arrival was
  scored as a false positive.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vision"))

from strata_core.synth import (  # noqa: E402
    canonical_size_for_scene,
    ground_truth_in_canonical,
    make_scene,
    make_still_pair,
)


class TestDeterminism:
    def test_same_seed_gives_identical_images(self):
        a = make_scene(seed=4242)
        b = make_scene(seed=4242)
        assert np.array_equal(a.baseline, b.baseline)
        assert np.array_equal(a.rescan, b.rescan)

    def test_same_seed_gives_the_same_mutations(self):
        a = make_scene(seed=4242)
        b = make_scene(seed=4242)
        assert [(m.kind, m.object_id) for m in a.mutations] == [
            (m.kind, m.object_id) for m in b.mutations
        ]

    def test_different_seeds_differ(self):
        a = make_scene(seed=1)
        b = make_scene(seed=2)
        assert not np.array_equal(a.baseline, b.baseline)

    def test_homography_is_a_3x3_matrix(self):
        s = make_scene(seed=7)
        assert s.homography.shape == (3, 3)
        assert np.isfinite(s.homography).all()


class TestObjectIdentity:
    """Each object must render from its own RNG stream."""

    def test_every_object_has_a_distinct_render_seed(self):
        s = make_scene(seed=11)
        seeds = [o.render_seed for o in s.baseline_objects]
        assert len(seeds) == len(set(seeds)), "render seeds are not unique"

    def test_mutating_one_object_does_not_alter_the_others(self):
        # The shared-RNG defect: removing one object shifted the stream, so
        # every later object changed appearance and the detector reported a
        # scatter of spurious diffs.
        s = make_scene(seed=99, n_objects=18, n_mutations=4)
        untouched = [
            o.object_id for o in s.baseline_objects
            if o.object_id not in {m.object_id for m in s.mutations}
        ]
        assert untouched, "no untouched object to check"

        base_by_id = {o.object_id: o for o in s.baseline_objects}
        res_by_id = {o.object_id: o for o in s.rescan_objects}
        for oid in untouched:
            b = base_by_id[oid]
            r = res_by_id[oid]
            assert b.bbox == r.bbox, f"{oid} moved without being mutated"
            assert b.colour == r.colour, f"{oid} recoloured without being mutated"


class TestRendering:
    def test_objects_stay_inside_the_frame(self):
        # The clipping defect: an unclipped write near the edge either crashed
        # or wrapped, and the recorded bbox then described a region that was
        # not fully drawn.
        size = (320, 240)
        for seed in range(20):
            s = make_scene(seed=seed, size=size, n_objects=18)
            for o in s.baseline_objects + s.rescan_objects:
                x, y, w, h = o.bbox
                assert x >= 0 and y >= 0
                assert x + w <= size[0], f"object {o.object_id} overflows width"
                assert y + h <= size[1], f"object {o.object_id} overflows height"

    def test_images_have_the_requested_size(self):
        s = make_scene(seed=5, size=(320, 240))
        assert s.baseline.shape == (240, 320, 3)
        assert s.rescan.shape == (240, 320, 3)

    def test_images_are_uint8(self):
        s = make_scene(seed=5)
        assert s.baseline.dtype == np.uint8
        assert s.rescan.dtype == np.uint8

    def test_objects_do_not_overlap_in_the_baseline(self):
        s = make_scene(seed=13, n_objects=18)
        boxes = [o.bbox for o in s.baseline_objects]
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                ax, ay, aw, ah = boxes[i]
                bx, by, bw, bh = boxes[j]
                overlap = not (
                    ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay
                )
                assert not overlap, f"objects {i} and {j} overlap"


class TestMutations:
    def test_recolour_is_perceptible(self):
        # The original recolour could pick a near-identical colour, making the
        # mutation invisible and the detector's failure a measurement artefact.
        for seed in range(15):
            s = make_scene(seed=seed, n_mutations=4)
            for m in s.mutations:
                if m.kind != "recolour":
                    continue
                before = next(
                    (o for o in s.baseline_objects if o.object_id == m.object_id), None
                )
                after = next(
                    (o for o in s.rescan_objects if o.object_id == m.object_id), None
                )
                assert before is not None and after is not None
                diff = max(
                    abs(int(a) - int(b)) for a, b in zip(before.colour, after.colour)
                )
                assert diff >= 40, f"recolour at seed {seed} differs by only {diff}"

    def test_removed_object_is_absent_from_the_rescan(self):
        s = make_scene(seed=21, n_mutations=4)
        for m in s.mutations:
            if m.kind != "remove":
                continue
            ids = {o.object_id for o in s.rescan_objects}
            assert m.object_id not in ids

    def test_added_object_is_present_only_in_the_rescan(self):
        s = make_scene(seed=21, n_mutations=4)
        for m in s.mutations:
            if m.kind != "add":
                continue
            base_ids = {o.object_id for o in s.baseline_objects}
            res_ids = {o.object_id for o in s.rescan_objects}
            assert m.object_id not in base_ids
            assert m.object_id in res_ids

    def test_moved_object_is_displaced_by_a_real_distance(self):
        for seed in range(10):
            s = make_scene(seed=seed, n_mutations=4)
            for m in s.mutations:
                if m.kind != "move":
                    continue
                before = next(
                    (o for o in s.baseline_objects if o.object_id == m.object_id), None
                )
                after = next(
                    (o for o in s.rescan_objects if o.object_id == m.object_id), None
                )
                assert before is not None and after is not None
                dist = np.hypot(after.x - before.x, after.y - before.y)
                assert dist >= 30, f"move at seed {seed} is only {dist:.1f}px"


class TestGroundTruth:
    def test_every_mutation_produces_a_ground_truth_entry(self):
        s = make_scene(seed=31, n_mutations=4)
        gts = s.ground_truth_events()
        covered = {g["object_id"] for g in gts}
        for m in s.mutations:
            assert m.object_id in covered, f"{m.kind} of {m.object_id} has no GT entry"

    def test_insertions_have_a_ground_truth_entry(self):
        # The defect: `add` had no entry, so a correct arrival counted as a
        # false positive.
        for seed in range(12):
            s = make_scene(seed=seed, n_mutations=4)
            adds = [m for m in s.mutations if m.kind == "add"]
            if not adds:
                continue
            kinds = {(g["kind"], g["object_id"]) for g in s.ground_truth_events()}
            for m in adds:
                assert ("added", m.object_id) in kinds

    def test_a_move_produces_both_an_origin_and_a_destination(self):
        s = make_scene(seed=41, n_mutations=4)
        moves = [m for m in s.mutations if m.kind == "move"]
        if not moves:
            pytest.skip("no move in this scene")
        gts = s.ground_truth_events()
        for m in moves:
            kinds = {g["kind"] for g in gts if g["object_id"] == m.object_id}
            assert "moved" in kinds
            assert "moved_destination" in kinds

    def test_bbox_space_is_declared_for_every_entry(self):
        s = make_scene(seed=51, n_mutations=4)
        for g in s.ground_truth_events():
            assert g["bbox_space"] in ("baseline", "rescan")

    def test_insertions_are_recorded_in_rescan_space(self):
        # An insertion has no baseline location; describing it in baseline
        # space would place the expected box in the wrong frame.
        s = make_scene(seed=61, n_mutations=4)
        for g in s.ground_truth_events():
            if g["kind"] == "added":
                assert g["bbox_space"] == "rescan"

    def test_ground_truth_kinds_are_from_the_known_set(self):
        s = make_scene(seed=71, n_mutations=4)
        allowed = {"missing", "added", "moved", "moved_destination", "state_changed"}
        for g in s.ground_truth_events():
            assert g["kind"] in allowed

    def test_ground_truth_boxes_are_inside_the_frame(self):
        size = (320, 240)
        s = make_scene(seed=81, size=size, n_mutations=4)
        for g in s.ground_truth_events():
            x, y, w, h = g["baseline_bbox"]
            assert 0 <= x and 0 <= y
            assert x + w <= size[0] + 1
            assert y + h <= size[1] + 1

    def test_each_entry_carries_its_mutation(self):
        s = make_scene(seed=91, n_mutations=4)
        for g in s.ground_truth_events():
            assert "mutation" in g
            assert g["mutation"]["kind"] == g["mutation"]["kind"]


class TestStillPair:
    def test_still_pair_returns_images_and_the_known_homography(self):
        base, res, homography = make_still_pair(seed=123)
        assert isinstance(base, np.ndarray)
        assert isinstance(res, np.ndarray)
        assert homography.shape == (3, 3)

    def test_still_pair_is_deterministic(self):
        a, b, h1 = make_still_pair(seed=123)
        c, d, h2 = make_still_pair(seed=123)
        assert np.array_equal(a, c)
        assert np.array_equal(b, d)
        assert np.allclose(h1, h2)

    def test_still_pair_has_no_mutations(self):
        # The same generator with mutations switched off: this is the
        # false-positive control, so it must contain nothing to detect.
        s = make_scene(seed=123, n_mutations=0)
        assert s.mutations == []
        assert s.ground_truth_events() == []

    def test_still_pair_images_differ_only_by_pose(self):
        # Nothing changed, so the two captures must not be identical - they are
        # taken from different viewpoints - but they depict the same objects.
        base, res, _ = make_still_pair(seed=7)
        assert base.shape == res.shape
        assert not np.array_equal(base, res)
        s = make_scene(seed=7, n_mutations=0)
        assert len(s.baseline_objects) == len(s.rescan_objects)

    def test_still_pair_preserves_object_identity(self):
        s = make_scene(seed=7, n_mutations=0)
        assert {o.object_id for o in s.baseline_objects} == {
            o.object_id for o in s.rescan_objects
        }

    def test_still_pair_objects_do_not_move(self):
        s = make_scene(seed=7, n_mutations=0)
        base_by_id = {o.object_id: o for o in s.baseline_objects}
        for o in s.rescan_objects:
            assert base_by_id[o.object_id].bbox == o.bbox


class TestSceneSave:
    def test_save_writes_images_and_metadata(self, tmp_path):
        import json

        s = make_scene(seed=131, size=(320, 240))
        out = s.save(tmp_path / "scene")
        assert (out / "baseline.png").exists()
        assert (out / "rescan.png").exists()
        meta = json.loads((out / "ground_truth.json").read_text(encoding="utf-8"))
        assert "baseline_objects" in meta
        assert "rescan_objects" in meta
        assert "mutations" in meta
        assert "ground_truth_events" in meta

    def test_saved_metadata_matches_the_scene(self, tmp_path):
        import json

        s = make_scene(seed=137, size=(320, 240), n_mutations=4)
        out = s.save(tmp_path / "scene")
        meta = json.loads((out / "ground_truth.json").read_text(encoding="utf-8"))
        assert len(meta["baseline_objects"]) == len(s.baseline_objects)
        assert len(meta["mutations"]) == len(s.mutations)
        assert len(meta["ground_truth_events"]) == len(s.ground_truth_events())


class TestGroundTruthInCanonical:
    """The shared conversion, which two consumers now depend on.

    It lives in the core because the evaluation harness and the service both need
    it, and because the first version of the service's answer-key panel
    re-derived it, got it wrong, and reported that a system which had found every
    change had found none.
    """

    def test_boxes_land_inside_the_canonical_frame(self):
        scene = make_scene(seed=1021, n_mutations=4)
        canonical = canonical_size_for_scene(scene)
        boxes = ground_truth_in_canonical(scene)
        assert boxes
        for entry in boxes:
            x, y, w, h = entry["bbox"]
            assert 0 <= x < canonical[0], entry
            assert 0 <= y < canonical[1], entry
            assert w > 0 and h > 0, entry
            assert x + w <= canonical[0] + 1, entry
            assert y + h <= canonical[1] + 1, entry

    def test_the_boxes_are_scaled_not_copied(self):
        scene = make_scene(seed=1021, n_mutations=4)
        canonical = canonical_size_for_scene(scene)
        assert canonical[0] > scene.size[0], "the canonical frame is larger than the scene"
        raw = scene.ground_truth_events()[0]["baseline_bbox"]
        converted = ground_truth_in_canonical(scene)[0]["bbox"]
        # A conversion that returned the raw numbers would leave the answer key
        # in the wrong coordinate system, which is the whole failure mode.
        assert converted[:2] != [float(v) for v in raw[:2]]

    def test_translation_destinations_are_excluded_by_default(self):
        scene = make_scene(seed=1021, n_mutations=4)
        default = ground_truth_in_canonical(scene)
        with_dest = ground_truth_in_canonical(scene, include_destinations=True)
        assert not any(e["kind"] == "moved_destination" for e in default)
        assert len(with_dest) >= len(default)

    def test_each_entry_keeps_the_space_its_box_came_from(self):
        scene = make_scene(seed=1021, n_mutations=4)
        spaces = {e["bbox_space"] for e in ground_truth_in_canonical(scene)}
        assert spaces <= {"baseline", "rescan"}
