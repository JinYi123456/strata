"""Unit tests for the canonicalisation geometry.

Canonicalisation is the step that lets two captures taken from different
viewpoints be compared at all, so a defect here is invisible downstream: every
later stage reports confidently on images that are not actually aligned.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vision"))

from strata_core.config import GeometryConfig  # noqa: E402
from strata_core.geometry import (  # noqa: E402
    _mean_symmetric_error,
    _sane_homography,
    canonical_size_for,
    render_canonical_baseline,
    warp_to_canonical,
)


class TestCanonicalSizeFor:
    def test_landscape_is_scaled_by_its_long_edge(self):
        # 960x720 -> long edge 1600, so the scale is 1600/960.
        w, h = canonical_size_for((720, 960), 1600)
        assert w == 1600
        assert h == pytest.approx(1200, abs=1)

    def test_portrait_is_scaled_by_its_long_edge(self):
        w, h = canonical_size_for((960, 720), 1600)
        assert h == 1600
        assert w == pytest.approx(1200, abs=1)

    def test_square_input_is_square_output(self):
        w, h = canonical_size_for((500, 500), 1000)
        assert (w, h) == (1000, 1000)

    def test_aspect_ratio_is_preserved(self):
        w, h = canonical_size_for((720, 1280), 1600)
        assert (w / h) == pytest.approx(1280 / 720, rel=1e-3)

    def test_small_input_is_upscaled(self):
        w, h = canonical_size_for((72, 96), 1600)
        assert w >= 1600


class TestSaneHomography:
    """A degeneracy check, not a magnitude check."""

    def test_identity_is_sane(self):
        assert _sane_homography(np.eye(3)) is True

    def test_nan_is_rejected(self):
        H = np.eye(3)
        H[0, 0] = np.nan
        assert _sane_homography(H) is False

    def test_infinity_is_rejected(self):
        H = np.eye(3)
        H[0, 0] = np.inf
        assert _sane_homography(H) is False

    def test_none_is_rejected(self):
        assert _sane_homography(None) is False

    def test_a_realistic_translation_is_sane(self):
        H = np.eye(3)
        H[0, 2] = 40.0
        H[1, 2] = -25.0
        assert _sane_homography(H) is True

    def test_a_singular_linear_part_is_rejected(self):
        # det of the 2x2 linear part is zero: the warp collapses the frame.
        H = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        assert _sane_homography(H) is False

    def test_a_large_but_non_degenerate_scale_is_accepted(self):
        # The guard exists to reject collapse and explosion, so a consistent
        # 50x scale - which still maps a bounded region to a bounded region -
        # is deliberately allowed through.
        H = np.eye(3)
        H[0, 0] = 50.0
        H[1, 1] = 50.0
        assert _sane_homography(H) is True


class TestMeanSymmetricError:
    def test_identical_points_have_zero_error(self):
        pts = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
        assert _mean_symmetric_error(np.eye(3), pts, pts) == pytest.approx(0.0)

    def test_error_matches_a_known_translation(self):
        src = np.array([[0.0, 0.0], [10.0, 0.0]])
        dst = np.array([[5.0, 0.0], [15.0, 0.0]])
        # The identity maps src to src, so the residual is the 5px gap.
        assert _mean_symmetric_error(np.eye(3), src, dst) == pytest.approx(5.0)

    def test_a_perfect_mapping_scores_zero(self):
        src = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
        H = np.eye(3)
        H[0, 2] = 5.0
        dst = src + np.array([5.0, 0.0])
        assert _mean_symmetric_error(H, src, dst) == pytest.approx(0.0, abs=1e-6)

    def test_empty_input_is_infinite(self):
        assert _mean_symmetric_error(np.eye(3), np.empty((0, 2)), np.empty((0, 2))) == float("inf")


class TestWarpToCanonical:
    def _reg(self, matrix, canonical=(160, 120)):
        from strata_core.geometry import Registration

        return Registration(
            model="homography",
            matrix=matrix,
            inliers_mask=np.ones(10, bool),
            inlier_count=10,
            match_count=10,
            inlier_ratio=1.0,
            reproj_error_px=0.5,
            canonical_size=canonical,
            confidence=0.9,
        )

    def test_returns_an_image_and_a_mask(self):
        img = np.zeros((120, 160, 3), np.uint8)
        out = warp_to_canonical(img, self._reg(np.eye(3)))
        assert isinstance(out, tuple) and len(out) == 2
        warped, mask = out
        assert warped.shape[:2] == mask.shape[:2]

    def test_identity_registration_preserves_content(self):
        img = np.zeros((120, 160, 3), np.uint8)
        img[40:80, 60:100] = 200
        warped, _ = warp_to_canonical(img, self._reg(np.eye(3)))
        assert warped.shape[:2] == (120, 160)
        # A centred patch must survive the identity warp.
        assert warped[40:80, 60:100].mean() > 150

    def test_output_has_the_canonical_size(self):
        img = np.zeros((120, 160, 3), np.uint8)
        warped, _ = warp_to_canonical(img, self._reg(np.eye(3), canonical=(320, 240)))
        assert warped.shape[:2] == (240, 320)

    def test_upscaling_produces_the_larger_frame(self):
        # 960x720 source into a 1600x1200 canonical frame.
        img = np.full((720, 960, 3), 90, np.uint8)
        warped, _ = warp_to_canonical(img, self._reg(np.eye(3), canonical=(1600, 1200)))
        assert warped.shape[:2] == (1200, 1600)

    def test_a_translation_moves_content(self):
        img = np.zeros((120, 160, 3), np.uint8)
        img[40:80, 60:100] = 255
        H = np.eye(3)
        H[0, 2] = 20.0
        warped, _ = warp_to_canonical(img, self._reg(H))
        # The patch should now sit 20px to the right.
        assert warped[40:80, 80:120].mean() > 150

    def test_mask_marks_the_covered_region(self):
        img = np.full((120, 160, 3), 200, np.uint8)
        _, mask = warp_to_canonical(img, self._reg(np.eye(3)))
        # An identity warp covers the whole frame, so nothing is marked as
        # missing evidence.
        assert mask.shape == (120, 160)

    def test_non_homography_model_resizes_instead_of_warping(self):
        img = np.zeros((120, 160, 3), np.uint8)
        img[40:80, 60:100] = 255
        reg = self._reg(np.eye(3))
        reg.model = "fundamental"
        warped, mask = warp_to_canonical(img, reg)
        assert warped.shape[:2] == (120, 160)
        # Everything is flagged: a non-homography model cannot place the frame,
        # so no region may be treated as reliable.
        assert mask.min() > 0


class TestRenderCanonicalBaseline:
    def test_pads_rather_than_distorting(self):
        # The baseline defines the canonical frame, so it must be placed
        # without changing its aspect ratio - padding is acceptable, stretching
        # is not.
        img = np.zeros((120, 160, 3), np.uint8)
        img[40:80, 60:100] = 255
        out = render_canonical_baseline(img, (320, 240))
        assert out.shape[:2] == (240, 320)
        # The patch's aspect ratio must be preserved in the output.
        ys, xs = np.where(out[..., 0] > 128)
        assert len(xs) > 0
        w = xs.max() - xs.min() + 1
        h = ys.max() - ys.min() + 1
        assert (w / h) == pytest.approx(40 / 40, rel=0.2)


class TestGeometryConfigDefaults:
    def test_defaults_are_coherent(self):
        cfg = GeometryConfig()
        assert cfg.min_inliers > 0
        assert 0.0 < cfg.min_inlier_ratio <= 1.0
        assert cfg.homography_reproj_threshold > 0
        assert cfg.canonical_long_edge >= 640

    def test_residual_thresholds_are_ordered(self):
        cfg = GeometryConfig()
        assert cfg.residual_good_px < cfg.residual_poor_px
