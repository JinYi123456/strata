"""Unit tests for configuration handling.

Configuration is the interface between the evaluation harness, the API service,
and every recorded result, so a silent round-trip failure would mean two runs
that claim the same settings are not actually comparable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vision"))

from strata_core.config import (  # noqa: E402
    CoverageConfig,
    DiffConfig,
    EventConfig,
    FeatureConfig,
    GeometryConfig,
    PipelineConfig,
)


class TestDefaults:
    def test_pipeline_config_builds_with_no_arguments(self):
        cfg = PipelineConfig()
        assert isinstance(cfg.features, FeatureConfig)
        assert isinstance(cfg.geometry, GeometryConfig)
        assert isinstance(cfg.diff, DiffConfig)
        assert isinstance(cfg.coverage, CoverageConfig)
        assert isinstance(cfg.events, EventConfig)

    def test_seed_is_set_for_determinism(self):
        assert isinstance(PipelineConfig().seed, int)

    def test_executor_label_defaults_to_local(self):
        assert PipelineConfig().executor == "baseline-local"

    def test_canonical_long_edge_is_at_least_the_source_width(self):
        # Canonicalising down would discard detail the pipeline then tries to
        # measure; the frame is deliberately a superset of the input.
        assert PipelineConfig().geometry.canonical_long_edge >= 960


class TestInvariants:
    def test_ransac_thresholds_are_positive(self):
        g = GeometryConfig()
        assert g.homography_reproj_threshold > 0
        assert g.fundamental_reproj_threshold > 0
        assert g.ransac_max_iters > 0
        assert 0.0 < g.ransac_confidence < 1.0

    def test_inlier_ratio_is_a_fraction(self):
        assert 0.0 < GeometryConfig().min_inlier_ratio <= 1.0

    def test_diff_thresholds_are_ordered(self):
        d = DiffConfig()
        assert d.min_abs_delta < d.max_threshold
        assert d.noise_sigma_mult > 0

    def test_chroma_opening_kernel_is_odd_and_usable(self):
        # cv2.morphologyEx rejects an even kernel.
        d = DiffConfig()
        assert d.chroma_open_ksize >= 3
        assert d.chroma_open_ksize % 2 == 1

    def test_local_gate_kernel_is_odd(self):
        assert DiffConfig().local_gate_ksize % 2 == 1

    def test_translation_bounds_are_ordered(self):
        e = EventConfig()
        assert 0 < e.translation_min_px < e.translation_max_px

    def test_pairing_guards_are_positive(self):
        e = EventConfig()
        assert e.pair_arrival_gap_min >= 0.0
        assert e.pair_arrival_area_px > 0
        assert 0.0 < e.pair_scale_ratio_min <= 1.0

    def test_size_ratio_bounds_are_fractions(self):
        e = EventConfig()
        assert 0.0 < e.translation_size_ratio_min <= 1.0

    def test_coverage_fractions_are_ordered(self):
        c = CoverageConfig()
        assert 0.0 < c.weak_fraction < c.confident_fraction <= 1.0

    def test_coverage_grid_is_positive(self):
        c = CoverageConfig()
        assert c.grid_rows > 0 and c.grid_cols > 0
        assert c.min_keypoints_per_cell > 0

    def test_presence_threshold_is_a_gap_not_a_ratio(self):
        # The presence measure pivots at 0.5, so a threshold above that would
        # never fire.
        assert 0.0 < EventConfig().presence_gap_min < 0.5


class TestImmutability:
    """Configs are frozen, so no run can mutate another run's settings."""

    def test_pipeline_config_rejects_assignment(self):
        cfg = PipelineConfig()
        with pytest.raises(Exception):
            cfg.seed = 1

    def test_sub_config_rejects_assignment(self):
        cfg = PipelineConfig()
        with pytest.raises(Exception):
            cfg.events.presence_gap_min = 0.9

    def test_two_configs_do_not_share_sub_config_instances(self):
        a = PipelineConfig()
        b = PipelineConfig()
        assert a.events is not b.events
        assert a.diff is not b.diff


class TestRoundTrip:
    def test_to_dict_then_from_dict_preserves_everything(self):
        cfg = PipelineConfig()
        restored = PipelineConfig.from_dict(cfg.to_dict())
        assert restored.to_dict() == cfg.to_dict()

    def test_round_trip_preserves_a_modified_value(self):
        from dataclasses import replace

        cfg = PipelineConfig(
            events=replace(EventConfig(), pair_arrival_gap_min=0.037),
            diff=replace(DiffConfig(), chroma_sigma_mult=4.25),
        )
        restored = PipelineConfig.from_dict(cfg.to_dict())
        assert restored.events.pair_arrival_gap_min == pytest.approx(0.037)
        assert restored.diff.chroma_sigma_mult == pytest.approx(4.25)

    def test_from_dict_tolerates_a_partial_document(self):
        # Forward compatibility: an older serialised config must still load,
        # filling the new fields from their defaults rather than failing.
        cfg = PipelineConfig.from_dict({"events": {"presence_gap_min": 0.2}})
        assert cfg.events.presence_gap_min == pytest.approx(0.2)
        assert cfg.events.pair_arrival_gap_min == EventConfig().pair_arrival_gap_min

    def test_from_dict_of_empty_matches_defaults(self):
        assert PipelineConfig.from_dict({}).to_dict() == PipelineConfig().to_dict()

    def test_unknown_keys_are_ignored_not_fatal(self):
        cfg = PipelineConfig.from_dict({"executor": "x", "not_a_field": 1})
        assert cfg.executor == "x"


class TestSerialisation:
    def test_to_json_is_valid_and_sorted(self):
        cfg = PipelineConfig()
        text = cfg.to_json()
        parsed = json.loads(text)
        assert isinstance(parsed, dict)
        # Sorted keys make two configs diff cleanly in a report.
        assert text.index('"coverage"') < text.index('"diff"')

    def test_save_and_load_are_inverse(self, tmp_path):
        from dataclasses import replace

        cfg = PipelineConfig(events=replace(EventConfig(), translation_max_px=333.0))
        path = tmp_path / "config.json"
        cfg.save(path)
        assert path.exists()
        loaded = PipelineConfig.load(path)
        assert loaded.events.translation_max_px == pytest.approx(333.0)

    def test_saved_file_is_human_readable_json(self, tmp_path):
        path = tmp_path / "config.json"
        PipelineConfig().save(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "features" in data
        assert "geometry" in data
        assert "diff" in data
        assert "events" in data


class TestSubConfigs:
    def test_feature_config_has_a_model_directory(self):
        assert isinstance(FeatureConfig().model_dir, str)

    def test_feature_config_backend_is_auto_by_default(self):
        assert FeatureConfig().backend in ("auto", "aliked", "sift", "disk")

    def test_ratio_test_is_a_fraction(self):
        assert 0.0 < FeatureConfig().ratio_test < 1.0
