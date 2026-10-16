"""Tests for the feature backends and the low-structure guard.

The guard exists because of a hard crash: LightGlue divides by zero on an image
with no structure, and on Windows that surfaces as a *fatal* C-level exception.
Python never gets to run a handler - the interpreter dies and takes the service
process with it. So the protection cannot be a ``try/except`` around the call;
it has to be a decision made before the call. That decision is what these tests
pin down.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vision"))

from strata_core.config import FeatureConfig, GeometryConfig  # noqa: E402
from strata_core.features import (  # noqa: E402
    _MIN_MATCH_STRUCTURE,
    _NO_STRUCTURE_FLOOR,
    FeatureSet,
    Matcher,
    _structure_energy,
    _structure_verdict,
    build_extractor,
    neural_available,
    resolve_backend,
)
from strata_core.geometry import register  # noqa: E402

CFG = FeatureConfig()


def _uniform(value: int = 127) -> np.ndarray:
    return np.full((720, 960, 3), value, np.uint8)


def _textured(seed: int = 0) -> np.ndarray:
    """A frame with real structure: noise plus hard edges."""
    rng = np.random.default_rng(seed)
    img = np.clip(rng.normal(120, 40, (720, 960, 3)), 0, 255).astype(np.uint8)
    for y in range(0, 720, 90):
        cv2.line(img, (0, y), (960, y), (210, 210, 210), 3)
    for x in range(0, 960, 120):
        cv2.line(img, (x, 0), (x, 720), (40, 40, 40), 2)
    return img


def _thin(offset: int = 0) -> np.ndarray:
    """A single dark square on a flat field - structure, but barely."""
    img = np.full((720, 960, 3), 128, np.uint8)
    cv2.rectangle(img, (400 + offset, 300 + offset), (500 + offset, 400 + offset), (30, 30, 30), -1)
    return img


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _fake_features(energy: float) -> FeatureSet:
    return FeatureSet(keypoints=[], descriptors=None, image_shape=(720, 960), structure_energy=energy)


class _Bomb:
    """Stands in for LightGlue; any use of it is a test failure."""

    def __getattr__(self, name):
        raise AssertionError(f"LightGlue was called ({name}) on a low-structure pair")


# --------------------------------------------------------------------------
# The measurement itself
# --------------------------------------------------------------------------


class TestStructureEnergy:
    def test_a_uniform_frame_has_none(self):
        assert _structure_energy(_gray(_uniform())) == 0.0

    def test_a_black_frame_has_none(self):
        assert _structure_energy(np.zeros((240, 320), np.uint8)) == 0.0

    def test_an_empty_array_is_not_a_crash(self):
        assert _structure_energy(np.zeros((0, 0), np.uint8)) == 0.0

    def test_a_real_scene_is_an_order_of_magnitude_higher(self):
        # The thresholds are only defensible if real content is far from them.
        assert _structure_energy(_gray(_textured())) > 10.0

    def test_a_single_small_square_is_barely_above_zero(self):
        energy = _structure_energy(_gray(_thin()))
        assert 0.0 < energy < _MIN_MATCH_STRUCTURE


# --------------------------------------------------------------------------
# The decision
# --------------------------------------------------------------------------


class TestStructureVerdict:
    def test_a_uniform_pair_is_flat(self):
        assert _structure_verdict(_fake_features(0.0), _fake_features(0.0)) == "flat"

    def test_one_flat_frame_makes_the_pair_flat(self):
        # LightGlue normalises over the pair, so the weaker frame governs.
        assert _structure_verdict(_fake_features(0.0), _fake_features(20.0)) == "flat"

    def test_a_thin_pair_is_thin(self):
        assert _structure_verdict(_fake_features(0.4), _fake_features(20.0)) == "thin"

    def test_a_real_pair_is_ok(self):
        assert _structure_verdict(_fake_features(18.0), _fake_features(22.0)) == "ok"

    def test_the_thresholds_are_ordered_and_positive(self):
        assert 0.0 < _NO_STRUCTURE_FLOOR < _MIN_MATCH_STRUCTURE


# --------------------------------------------------------------------------
# The guard actually prevents the call
# --------------------------------------------------------------------------


class TestGuardShortCircuitsBeforeLightGlue:
    """The crash is unfixable from Python once it starts, so the call must not happen."""

    @pytest.fixture
    def guarded(self):
        cfg = FeatureConfig(backend="neural")
        if not neural_available(cfg):
            pytest.skip("neural weights unavailable")
        matcher = Matcher(cfg)
        matcher._lightglue = _Bomb()  # noqa: SLF001 - deliberately armed
        return matcher

    def test_a_flat_pair_never_reaches_lightglue(self, guarded):
        ms = guarded.match(_fake_features(0.0), _fake_features(0.0))
        assert ms.backend == "degenerate-flat"
        assert len(ms) == 0
        assert ms.low_structure is True

    def test_a_thin_pair_never_reaches_lightglue(self, guarded):
        ms = guarded.match(_fake_features(0.4), _fake_features(0.4))
        assert ms.backend == "classic-lowstructure"
        assert ms.low_structure is True

    def test_a_real_pair_still_reaches_lightglue(self):
        cfg = FeatureConfig(backend="neural")
        if not neural_available(cfg):
            pytest.skip("neural weights unavailable")
        extractor = build_extractor(cfg)
        matcher = Matcher(cfg)
        a = extractor.detect(_gray(_textured(1)))
        b = extractor.detect(_gray(_textured(1)))
        assert _structure_verdict(a, b) == "ok"
        ms = matcher.match(a, b)
        assert ms.backend == "neural-lightglue"
        assert ms.low_structure is False


# --------------------------------------------------------------------------
# The consequence downstream: a thin alignment is never reported as confident
# --------------------------------------------------------------------------


class TestLowStructureRegistrationIsNotTrusted:
    def test_the_flag_survives_into_the_registration(self):
        gcfg = GeometryConfig()
        rng = np.random.default_rng(7)
        src = rng.uniform(0, 400, (120, 2)).astype(np.float32)
        from strata_core.features import MatchSet

        ms = MatchSet(
            src_points=src,
            dst_points=src + 5.0,
            backend="classic-lowstructure",
            low_structure=True,
        )
        reg = register(ms, src_shape=(720, 960), dst_shape=(720, 960), cfg=gcfg)
        assert reg.confidence <= gcfg.low_structure_confidence_cap
        assert any("low-structure" in n for n in reg.notes)

    def test_a_flat_pair_is_reported_as_such_rather_than_as_a_plain_shortfall(self):
        gcfg = GeometryConfig()
        from strata_core.features import MatchSet

        ms = MatchSet(
            src_points=np.zeros((0, 2), np.float32),
            dst_points=np.zeros((0, 2), np.float32),
            backend="degenerate-flat",
            low_structure=True,
        )
        reg = register(ms, src_shape=(720, 960), dst_shape=(720, 960), cfg=gcfg)
        assert reg.ok is False
        assert reg.confidence == 0.0
        assert any("no structure" in n for n in reg.notes)


# --------------------------------------------------------------------------
# Measured behaviour worth pinning, including one honest limitation
# --------------------------------------------------------------------------


class TestMeasuredBehaviour:
    def test_a_thin_pair_with_no_real_correspondence_cannot_be_caught(self):
        """Known limitation, measured.

        Two thin images that genuinely have nothing in common still produce a
        geometrically consistent match set: the ratio test cannot discriminate
        when every descriptor is near-identical, so several hundred
        correspondences survive and MAGSAC fits a transform to them. Before the
        cap this was reported at confidence 0.95 - a confident, fabricated
        alignment.

        It cannot be fixed by inspecting the geometry, because the geometry
        *looks* fine. The cap is the honest remedy: the alignment is still
        returned, but never as trustworthy. This test will fail loudly if a
        future change makes the discrimination possible, which is the point.
        """
        cfg = FeatureConfig(backend="neural")
        if not neural_available(cfg):
            pytest.skip("neural weights unavailable")
        extractor = build_extractor(cfg)
        matcher = Matcher(cfg)

        a = extractor.detect(_gray(_thin(offset=0)))
        b = extractor.detect(_gray(_thin(offset=500)))

        ms = matcher.match(a, b)
        assert ms.low_structure is True
        reg = register(
            ms, src_shape=(720, 960), dst_shape=(720, 960), cfg=GeometryConfig()
        )
        # The alignment is fabricated, but it is not presented as confident.
        assert reg.confidence <= GeometryConfig().low_structure_confidence_cap

    def test_a_genuine_thin_pair_still_aligns(self):
        # The guard must not cost a real capability: two views of the same
        # object on a plain background are still comparable.
        cfg = FeatureConfig(backend="neural")
        if not neural_available(cfg):
            pytest.skip("neural weights unavailable")
        extractor = build_extractor(cfg)
        matcher = Matcher(cfg)

        a = extractor.detect(_gray(_thin(offset=0)))
        b = extractor.detect(_gray(_thin(offset=10)))

        ms = matcher.match(a, b)
        assert len(ms) > 0
        reg = register(
            ms, src_shape=(720, 960), dst_shape=(720, 960), cfg=GeometryConfig()
        )
        assert reg.ok is True
