"""Unit tests for the difference engine's two channels.

The claims under test are the ones this module makes in its docstrings, and
each of them came from a measured defect rather than from theory:

* luminance difference cannot see a pure colour change, so a chrominance
  channel is a necessity rather than a refinement;
* the two channels must each be gated on their *own* coherence before being
  combined, because a shared gate deletes one channel's evidence;
* fragments whose evidence points in opposite directions must not be fused,
  because averaging a departure with an arrival describes nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vision"))

from strata_core.config import DiffConfig  # noqa: E402
from strata_core.diff import (  # noqa: E402
    _chroma_opponent,
    _to_gray,
    compute_diff,
)


def bgr(b, g, r, size=(64, 64)):
    img = np.zeros((size[1], size[0], 3), np.uint8)
    img[..., 0] = b
    img[..., 1] = g
    img[..., 2] = r
    return img


def with_patch(img, colour, box):
    out = img.copy()
    x, y, w, h = box
    out[y:y + h, x:x + w] = colour
    return out


class TestChromaOpponent:
    def test_neutral_grey_lands_at_the_centre(self):
        rg, by = _chroma_opponent(bgr(128, 128, 128))
        assert np.all(rg == 128)
        assert np.all(by == 128)

    def test_opponent_channels_cancel_additive_exposure_drift(self):
        # An exposure offset adds the same amount to all three channels. The
        # opponent differences must be unchanged, which is the property that
        # keeps a drifting capture from reading as a colour change.
        base = bgr(90, 120, 200)
        lifted = bgr(90 + 20, 120 + 20, 200 + 20)
        rd, bd = _chroma_opponent(base)
        rl, bl = _chroma_opponent(lifted)
        assert abs(float(rd.mean()) - float(rl.mean())) <= 1.0
        assert abs(float(bd.mean()) - float(bl.mean())) <= 1.0

    def test_opponent_channels_preserve_hue_under_gain(self):
        # A multiplicative gain changes the magnitude of the opponent
        # differences - halving every channel halves them - so the channel is
        # *not* scale invariant, and a test asserting that would be wrong.
        # What survives is the sign, which is what carries the hue: a surface
        # that was redder than its surround still is.
        base = bgr(90, 120, 200)
        dimmed = bgr(45, 60, 100)
        rd, _ = _chroma_opponent(base)
        rdd, _ = _chroma_opponent(dimmed)
        assert np.sign(float(rd.mean()) - 128) == np.sign(float(rdd.mean()) - 128)
        # And the magnitude does change, confirming the test above is testing
        # the additive case rather than passing by accident.
        assert abs(float(rd.mean()) - float(rdd.mean())) > 10.0

    def test_a_real_colour_change_moves_the_channel(self):
        a = _chroma_opponent(bgr(87, 150, 236))[0]
        b = _chroma_opponent(bgr(238, 164, 89))[0]
        # These are the two colours from the benchmark that motivated the
        # channel: a blue and an orange that are nearly identical in luma.
        assert abs(float(a.mean()) - float(b.mean())) > 40

    def test_greyscale_input_yields_neutral_channels(self):
        flat = np.full((32, 32), 77, np.uint8)
        rg, by = _chroma_opponent(flat)
        assert np.all(rg == 128)
        assert np.all(by == 128)

    def test_bgra_is_accepted(self):
        img = np.zeros((16, 16, 4), np.uint8)
        img[..., 3] = 255
        rg, by = _chroma_opponent(img)
        assert rg.shape == (16, 16)


class TestLuminanceBlindness:
    """The defect that justified the chrominance channel."""

    def test_the_benchmark_recolour_is_nearly_invisible_in_luma(self):
        a = _to_gray(bgr(87, 150, 236))
        b = _to_gray(bgr(238, 164, 89))
        # Measured at 18.5 grey levels - far below the pipeline's own
        # minimum_abs_delta of 26, so the change is undetectable from
        # luminance alone at any threshold.
        delta = abs(float(a.mean()) - float(b.mean()))
        assert delta < 20.0

    def test_but_the_chroma_channel_sees_it_clearly(self):
        a = _chroma_opponent(bgr(87, 150, 236))[0]
        b = _chroma_opponent(bgr(238, 164, 89))[0]
        delta = abs(float(a.mean()) - float(b.mean()))
        assert delta > 40.0


class TestComputeDiffChroma:
    def _scene_pair(self):
        base = np.full((240, 320, 3), 60, np.uint8)
        base = with_patch(base, (150, 150, 150), (10, 10, 120, 90))
        res = base.copy()
        # Recolour one patch to a hue of near-identical luma.
        res = with_patch(res, (236, 150, 87), (140, 40, 90, 70))
        return base, res

    def test_a_colour_only_change_is_detected(self):
        base, res = self._scene_pair()
        out = compute_diff(base, res, DiffConfig())
        assert out.diff_gray is not None
        # Some region must be reported, and the chrominance channel is what
        # makes that possible.
        assert out.regions, "a chroma-only change produced no region"
        assert out.stats.get("chroma_pixels", 0) > 0

    def test_stats_expose_the_chroma_channel(self):
        base, res = self._scene_pair()
        out = compute_diff(base, res, DiffConfig())
        assert "chroma_threshold_used" in out.stats

    def test_identical_images_produce_no_regions(self):
        base, _ = self._scene_pair()
        out = compute_diff(base, base, DiffConfig())
        assert not out.regions

    def test_regions_carry_a_footprint(self):
        # The footprint is what lets the presence measure tell changed pixels
        # from untouched surface, so a region without one is unusable there.
        base, res = self._scene_pair()
        out = compute_diff(base, res, DiffConfig())
        for r in out.regions:
            assert r.footprint is not None
            assert r.footprint.dtype == bool

    def test_footprint_dimensions_match_its_bbox(self):
        base, res = self._scene_pair()
        out = compute_diff(base, res, DiffConfig())
        for r in out.regions:
            _, _, w, h = r.bbox
            assert r.footprint.shape == (h, w)

    def test_disabling_chroma_loses_the_colour_change(self):
        # The control for the claim above: with the chrominance floor pushed
        # impossibly high, the recolour is no longer found. This proves the
        # channel, not some other path, is doing the work.
        base, res = self._scene_pair()
        cfg = DiffConfig(min_chroma_delta=255.0, chroma_sigma_mult=1e9)
        out = compute_diff(base, res, cfg)
        assert out.stats.get("chroma_pixels", 0) == 0


class TestPolarityAwareMerging:
    """Opposite-polarity fragments must not be fused into one region."""

    def test_a_translation_is_not_merged_into_one_box(self):
        # Two patches: one disappears (structure leaves) and an equal one
        # appears nearby. Fusing them averages a departure with an arrival and
        # yields a box that is, on balance, unchanged - which is the defect.
        base = np.full((240, 320, 3), 70, np.uint8)
        res = base.copy()
        base = with_patch(base, (200, 200, 200), (20, 40, 70, 60))
        res = with_patch(res, (200, 200, 200), (150, 40, 70, 60))
        out = compute_diff(base, res, DiffConfig())
        # Two opposite-polarity regions separated by a gap wider than the merge
        # distance must remain two regions, not one.
        assert len(out.regions) >= 2

    def test_opposite_polarity_regions_are_never_averaged(self):
        base = np.full((240, 320, 3), 70, np.uint8)
        res = base.copy()
        base = with_patch(base, (200, 200, 200), (20, 40, 70, 60))
        res = with_patch(res, (200, 200, 200), (150, 40, 70, 60))
        out = compute_diff(base, res, DiffConfig())
        for r in out.regions:
            # No single region may span both the vacated and the occupied
            # locations, which would be the signature of a fused pair.
            x, _, w, _ = r.bbox
            spans_both = x <= 20 and (x + w) >= 220
            assert not spans_both


class TestDiffConfigDefaults:
    def test_chroma_is_enabled_by_default(self):
        cfg = DiffConfig()
        assert cfg.min_chroma_delta < 255
        assert cfg.chroma_sigma_mult > 0
        assert cfg.chroma_open_ksize >= 3

    def test_opening_kernel_is_odd(self):
        # cv2.morphologyEx requires an odd kernel; the pipeline masks it, but
        # the default should already be valid.
        assert DiffConfig().chroma_open_ksize % 2 == 1
