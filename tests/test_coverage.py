"""Tests for the coverage map and the capture guidance.

Coverage is the module behind the product's most distinctive behaviour: knowing
where it has not looked, and asking for a specific extra capture instead of
quietly reporting "nothing changed" over ground it never saw. It had no tests
before these, which is the wrong place to have a blind spot - a coverage bug
produces confident silence, and confident silence is the worst output this
system can give.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vision"))

from strata_core.config import CoverageConfig  # noqa: E402
from strata_core.coverage import (  # noqa: E402
    CoverageAccumulator,
    coverage_summary,
    direction_to_cell,
    view_requests,
)
from strata_core.features import build_extractor  # noqa: E402
from strata_core.config import FeatureConfig  # noqa: E402

CANONICAL = (1600, 1200)
CFG = CoverageConfig()


def _textured(h: int = 600, w: int = 800, seed: int = 0) -> np.ndarray:
    """A frame with real structure: noise plus hard edges."""
    rng = np.random.default_rng(seed)
    img = np.clip(rng.normal(120, 40, (h, w)), 0, 255).astype(np.uint8)
    for y in range(0, h, 40):
        cv2.line(img, (0, y), (w, y), 240, 3)
    for x in range(0, w, 60):
        cv2.line(img, (x, 0), (x, h), 20, 3)
    return img


def _accumulate(frames: list[np.ndarray]) -> object:
    acc = CoverageAccumulator(CFG, CANONICAL)
    extractor = build_extractor(FeatureConfig(backend="classic"))
    for f in frames:
        canonical = cv2.resize(f, CANONICAL, interpolation=cv2.INTER_AREA)
        acc.add_frame(canonical, extractor.detect(canonical))
    return acc.build()


class TestAccumulator:
    def test_no_frames_gives_an_empty_map(self):
        cmap = CoverageAccumulator(CFG, CANONICAL).build()
        assert cmap.frame_count == 0
        assert cmap.observed_fraction == 0.0
        assert cmap.status == "unobserved"

    def test_the_grid_shape_is_honoured(self):
        cmap = _accumulate([_textured()])
        assert cmap.rows == CFG.grid_rows
        assert cmap.cols == CFG.grid_cols
        assert cmap.keypoint_counts.shape == (CFG.grid_rows, CFG.grid_cols)

    def test_a_textured_frame_observes_something(self):
        cmap = _accumulate([_textured()])
        assert cmap.frame_count == 1
        assert cmap.observed_fraction > 0.0

    def test_a_blank_frame_observes_nothing(self):
        blank = np.full((600, 800), 127, np.uint8)
        cmap = _accumulate([blank])
        assert cmap.observed_fraction == 0.0
        assert cmap.status == "unobserved"

    def test_the_canonical_size_is_carried_not_assumed(self):
        # The grid is resolution-independent but the coordinates fed into it are
        # not, so a consumer mapping a canonical point to a cell needs this.
        cmap = CoverageAccumulator(CFG, (800, 600)).build()
        assert cmap.canonical_size == (800, 600)
        assert cmap.to_dict()["canonical_size"] == [800, 600]

    def test_more_frames_never_reduce_coverage(self):
        one = _accumulate([_textured(seed=1)])
        two = _accumulate([_textured(seed=1), _textured(seed=2)])
        assert two.observed_fraction >= one.observed_fraction

    def test_the_frame_cap_is_respected(self):
        cfg = CoverageConfig(max_frames=2)
        acc = CoverageAccumulator(cfg, CANONICAL)
        for i in range(5):
            acc.add_frame(np.zeros((120, 160), np.uint8))
        assert acc.frame_count == 2

    def test_the_imaged_mask_is_more_permissive_than_the_observed_mask(self):
        # They answer different questions and the difference is load-bearing:
        # imaged gates the diff, observed drives guidance. Using the demanding
        # one as the gate discards exactly the smooth regions where a new object
        # would show up.
        cmap = _accumulate([_textured()])
        imaged = cmap.imaged_mask(CANONICAL)
        observed = cmap.mask(CANONICAL)
        assert np.count_nonzero(imaged) >= np.count_nonzero(observed)


class TestViewRequests:
    def test_nothing_to_ask_for_when_coverage_is_good(self):
        cmap = _accumulate([_textured(seed=3)])
        # Force a well-covered map by hand: every cell above the bar.
        cmap.keypoint_counts[:] = 50
        cmap.texture_energy[:] = 1.0
        assert view_requests(cmap, CFG) == []

    def test_a_thin_map_produces_requests(self):
        cmap = _accumulate([np.full((600, 800), 127, np.uint8)])
        requests = view_requests(cmap, CFG)
        assert requests, "a scene nothing was observed in should produce guidance"
        for r in requests:
            assert r.direction
            assert r.reason
            assert 0 <= r.target_cell[0] < cmap.rows
            assert 0 <= r.target_cell[1] < cmap.cols
            assert r.deficit > 0

    def test_requests_are_capped(self):
        cmap = _accumulate([np.full((600, 800), 127, np.uint8)])
        assert len(view_requests(cmap, CFG, max_requests=2)) <= 2

    def test_one_request_per_direction(self):
        """The defect this pins.

        A single unobserved frame produces a whole set of adjacent thin cells,
        and they nearly all imply the same instruction. Reporting them
        individually gave an operator three separate "pan left" entries for
        three cells down the same edge - which reads as three jobs rather than
        the one movement that fixes all of them.
        """
        for seed in (0, 1, 2):
            cmap = _accumulate([_textured(seed=seed)])
            directions = [r.direction for r in view_requests(cmap, CFG, max_requests=4)]
            assert len(directions) == len(set(directions)), directions

    def test_distinct_directions_are_still_all_reported(self):
        # Merging must not collapse a scene with genuinely different problems
        # into a single instruction.
        cmap = _accumulate([np.full((600, 800), 127, np.uint8)])
        directions = {r.direction for r in view_requests(cmap, CFG, max_requests=4)}
        assert len(directions) >= 2, directions

    def test_a_merged_request_says_how_many_cells_it_covers(self):
        cmap = _accumulate([np.full((600, 800), 127, np.uint8)])
        merged = [r for r in view_requests(cmap, CFG, max_requests=4) if "cells in this direction" in r.reason]
        assert merged, "a scene with many thin cells should report a merged instruction"

    def test_requests_are_ordered_by_priority(self):
        cmap = _accumulate([np.full((600, 800), 127, np.uint8)])
        priorities = [r.priority for r in view_requests(cmap, CFG, max_requests=4)]
        assert priorities == sorted(priorities, reverse=True)

    def test_directions_are_words_an_operator_can_follow(self):
        cmap = _accumulate([np.full((600, 800), 127, np.uint8)])
        for r in view_requests(cmap, CFG, max_requests=4):
            assert r.direction.split()[0] in {"pan", "tilt", "step"}


class TestDirectionToCell:
    def test_a_left_cell_is_reached_by_panning_left(self):
        cmap = _accumulate([np.full((600, 800), 127, np.uint8)])
        direction, reason = direction_to_cell(cmap, (cmap.rows // 2, 0))
        assert "left" in direction
        assert reason

    def test_a_right_cell_is_reached_by_panning_right(self):
        cmap = _accumulate([np.full((600, 800), 127, np.uint8)])
        direction, _ = direction_to_cell(cmap, (cmap.rows // 2, cmap.cols - 1))
        assert "right" in direction

    def test_a_high_cell_is_reached_by_tilting(self):
        cmap = _accumulate([np.full((600, 800), 127, np.uint8)])
        direction, _ = direction_to_cell(cmap, (0, cmap.cols // 2))
        assert "tilt" in direction

    def test_an_out_of_range_cell_is_clamped_not_a_crash(self):
        cmap = _accumulate([np.full((600, 800), 127, np.uint8)])
        for cell in ((-3, -3), (cmap.rows + 5, cmap.cols + 5)):
            direction, reason = direction_to_cell(cmap, cell)
            assert direction and reason


class TestSummary:
    def test_the_summary_is_json_serialisable(self):
        import json

        cmap = _accumulate([_textured()])
        summary = coverage_summary(cmap, view_requests(cmap, CFG))
        json.dumps(summary)
        assert summary["status"] in {"observed", "weak", "unobserved"}
        assert 0.0 <= summary["observed_fraction"] <= 1.0
        assert summary["grid"] == [CFG.grid_rows, CFG.grid_cols]
        assert isinstance(summary["view_requests"], list)

    def test_the_status_follows_the_fraction(self):
        cmap = _accumulate([_textured()])
        summary = coverage_summary(cmap, [])
        if summary["observed_fraction"] >= CFG.confident_fraction:
            assert summary["status"] == "observed"
        elif summary["observed_fraction"] >= CFG.weak_fraction:
            assert summary["status"] == "weak"
        else:
            assert summary["status"] == "unobserved"


class TestHeatmapRendering:
    def test_the_heatmap_is_an_image_at_the_requested_size(self):
        cmap = _accumulate([_textured()])
        heat = cmap.heatmap((320, 240))
        assert heat.shape == (240, 320, 3)
        assert heat.dtype == np.uint8

    def test_the_heatmap_defaults_to_grid_resolution(self):
        """One pixel per cell, which is what was actually measured.

        Resizing by default would interpolate smoothness into a per-cell
        measurement and present it as if it were per-pixel detail.
        """
        cmap = _accumulate([_textured()])
        assert cmap.heatmap().shape[:2] == (cmap.rows, cmap.cols)
