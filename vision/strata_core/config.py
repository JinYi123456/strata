"""Configuration objects for the STRATA vision pipeline.

Every knob that changes the geometry, the diff, or the coverage math lives here
so a run is fully reproducible from a single serialisable object.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

FeatureBackend = Literal["auto", "neural", "classic"]

# Canonical frame geometry. Every session is registered into a frame of this
# size so that two captures of the same place end up pixel-comparable.
CANONICAL_LONG_EDGE = 1600


@dataclass(frozen=True)
class FeatureConfig:
    """Local-feature extraction and matching.

    ``backend="auto"`` prefers the neural path (ALIKED + LightGlue) when the
    ONNX weights are present on disk, and falls back to SIFT + FLANN
    otherwise. The fallback is not a degraded mode: it is the guaranteed-works
    path that keeps the pipeline runnable with zero downloads.
    """

    backend: FeatureBackend = "auto"
    max_features: int = 4096
    # Neural models (ONNX). Resolved relative to ``model_dir``.
    aliked_model: str = "aliked-n16rot-top2k-640.onnx"
    lightglue_model: str = "lightglue_for_aliked.onnx"
    model_dir: str = "models"
    # Classic fallback tuning.
    sift_contrast_threshold: float = 0.02
    sift_edge_threshold: float = 12.0
    ratio_test: float = 0.75


@dataclass(frozen=True)
class GeometryConfig:
    """Robust registration into the canonical frame.

    The model is chosen adaptively: a homography for near-planar scenes
    (shelves, walls, benches) and a fundamental matrix for scenes with real
    depth relief. USAC/MAGSAC is the estimator in both cases.
    """

    # USAC confidence / thresholds.
    ransac_confidence: float = 0.9999
    ransac_max_iters: int = 10000
    homography_reproj_threshold: float = 3.0
    fundamental_reproj_threshold: float = 1.5
    # A homography is accepted only if it keeps the canonical quad sane.
    min_inlier_ratio: float = 0.25
    min_inliers: int = 40
    # Planar scenes are the common case; prefer H unless F is clearly better.
    prefer_planar: bool = True
    canonical_long_edge: int = CANONICAL_LONG_EDGE
    # Residual thresholds (px) used for confidence shaping. The diff module
    # owns the full residual-gating thresholds.
    residual_good_px: float = 2.5
    residual_poor_px: float = 6.0
    # Ceiling on confidence for a registration whose correspondences came from a
    # pair with almost no structure to match (see ``features._structure_verdict``).
    # Such a pair can still yield a geometrically consistent match set that has
    # nothing to do with reality, so the alignment must not be reported as
    # trustworthy. Placed deliberately just above the agent's abort threshold:
    # a thin pair is never silently trusted, but it is also never discarded
    # outright - the agent gets to see that the alignment is weak and ask for a
    # better view, which is the whole point of the loop.
    low_structure_confidence_cap: float = 0.34


@dataclass(frozen=True)
class DiffConfig:
    """Canonical change detection and change-region extraction.

    Thresholds are *calibrated per image pair* rather than fixed, because the
    disagreement floor between two real captures depends on noise, resampling
    and the registration residual of that specific pair. These settings control
    how that calibration is performed.
    """

    # Photometric normalisation before differencing.
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: int = 8
    blur_ksize: int = 5

    # --- Data-calibrated thresholding ---------------------------------
    # A pixel is a change candidate only if it exceeds
    #   median + max(noise_sigma_mult * robust_sigma, min_abs_delta)
    # and a robust upper percentile of the difference image.
    noise_sigma_mult: float = 4.0
    min_abs_delta: float = 26.0
    # Percentile used as the upper anchor. Kept well below the extreme tail
    # because a warp border contributes a few enormous outlier pixels that
    # would otherwise drag the threshold above real signal.
    strong_percentile: float = 94.0
    # Hard ceiling on the calibrated threshold. Guarantees that a large,
    # obvious change can never be excluded by its own contribution to the
    # statistics.
    max_threshold: float = 78.0
    # Pixels this close to the frame edge are excluded from calibration,
    # because the warp leaves an uncovered wedge there.
    border_margin_px: int = 14
    # Extra margin above the floor required to be called a *strong* change.
    strong_margin: float = 14.0
    # Neighbourhood size for the local-coherence gate: resampling noise is
    # incoherent, genuine object change is not.
    local_gate_ksize: int = 9

    # --- Chrominance channel ------------------------------------------
    # A surface changing colour is real change that a luminance difference
    # structurally cannot see: luma is a weighted sum of the channels, so a
    # saturated blue and a saturated orange can differ by barely fifteen levels
    # once converted. This channel measures the opponent colour differences
    # (r-g and b-y), which exposure drift cancels out of exactly.
    #
    # The floor is low because chroma noise genuinely is low: measured across
    # the evaluation scene the median absolute deviation of the channel is
    # around one level, so the sigma term does the work and this constant only
    # has to exclude quantisation. A generous fixed floor would be actively
    # harmful here - it would sit far above the noise and admit a few percent
    # of the frame, turning one changed object into a scatter of fragments.
    chroma_sigma_mult: float = 3.0
    min_chroma_delta: float = 6.0
    # Morphological opening applied to the chrominance mask. Chroma evidence for
    # a real change forms a solid patch, so opening with a kernel this size
    # removes the isolated pixels that survive the threshold without eating into
    # a genuine surface's area.
    chroma_open_ksize: int = 9

    # Morphology to clean mask speckle.
    morph_open_ksize: int = 3
    morph_close_ksize: int = 9

    # --- Region filtering ---------------------------------------------
    min_region_area_px: int = 420
    max_region_fraction: float = 0.35
    # Fuse fragments of one object that thresholding and morphology split apart.
    merge_fragments: bool = True
    merge_max_gap_px: int = 26
    merge_max_area_ratio: float = 3.5
    # Minimum |polarity| for a region's evidence to count as directional when
    # deciding whether two regions may be fused. Below this the region is
    # treated as carrying no direction, so it is free to merge either way.
    merge_polarity_min: float = 0.08
    # Regions whose mean difference is only marginally above the floor are
    # weak evidence and are down-weighted.
    weak_evidence_ratio: float = 1.25
    # Registration residual gates: regions inside the poor-registration band
    # are reported as low-confidence rather than as changes.
    residual_good_px: float = 2.5
    residual_poor_px: float = 6.0
    # Floor for a region's reported confidence.
    min_confidence: float = 0.05


@dataclass(frozen=True)
class CoverageConfig:
    """Capture-coverage and uncertainty mapping for active perception.

    This is what lets the agent decide *where to look next*: a live map of
    which parts of the canonical frame are well observed, weakly observed, or
    not observed at all.
    """

    grid_rows: int = 8
    grid_cols: int = 12
    # A cell is "observed" when it holds at least this many matched keypoints
    # or this much locally-aligned texture energy.
    min_keypoints_per_cell: int = 6
    # Observed fraction above which the reconstruction clears the bar.
    confident_fraction: float = 0.72
    weak_fraction: float = 0.45
    # Frames sampled per session for coverage accumulation.
    max_frames: int = 240
    # An extra capture only counts as evidence when its own registration into the
    # canonical frame clears this. Below it, the view is recorded as attempted and
    # discarded rather than credited, because a badly-registered frame makes a
    # cell look observed when nothing in it can be trusted.
    extra_view_min_confidence: float = 0.35


@dataclass(frozen=True)
class EventConfig:
    """Change regions resolved into typed, explained events.

    Presence is measured *comparatively*: for each box the structure inside it
    is compared against the structure of the ring immediately around it in the
    same image. That makes "is there an object here" independent of how busy
    the scene's own background is. See ``events`` for the full rationale.
    """

    # Minimum drop in presence between the two visits for a region to be called
    # a removal or an insertion. A gap between two ratios, not an absolute
    # level, so it does not depend on scene texture.
    presence_gap_min: float = 0.10
    # Template-match correlation above which a shifted patch counts as the same
    # object that moved, rather than a different object.
    shift_match_min: float = 0.55
    # How much better the shifted match must be than the in-place match before
    # a translation is asserted. Guards against coincidental correlations.
    shift_advantage_min: float = 0.12
    # --- Translation pairing ---------------------------------------------
    # A translation is recognised structurally as a departure at the old
    # location plus a similar-sized arrival nearby. These bound that pairing.
    translation_min_px: float = 30.0
    translation_max_px: float = 420.0
    # Minimum area similarity between the two halves of a translation. A
    # departure box bounds a *hole* - the surface the object used to cover,
    # which is not its silhouette - so this is deliberately loose: a tight
    # bound would reject the physically accurate pairs. It exists to reject a
    # small region paired with one an order of magnitude larger.
    translation_size_ratio_min: float = 0.35
    # Minimum arrival presence gap for a region to be usable as the destination
    # half of a translation. Measured on the benchmark, correct pairs showed an
    # arrival gap of 0.065-0.45 while mis-paired ones showed 0.013-0.022, so
    # this floor sits well below the true minimum on purpose - it rejects the
    # clearly-empty arrivals without tuning to this benchmark's exact
    # distribution, which would fail on a scene with weaker evidence.
    pair_arrival_gap_min: float = 0.02
    # Minimum area, in canonical pixels, for a region to serve as the arrival
    # half of a translation. An arrival is an object standing somewhere; a
    # region this small is a fragment or noise, and pairing with one would
    # assert a translation on the strength of scattered pixels. This is set
    # from the smallest object the scene generator produces, not from the
    # benchmark's observed pairs.
    pair_arrival_area_px: float = 1500.0
    # Minimum ratio between the two halves' areas for a pair to be preferred
    # over reading them as an unrelated removal and insertion. A single object
    # leaves a hole of roughly its own size, so both halves should be
    # comparable; a pair an order of magnitude apart is better explained as two
    # separate events.
    pair_scale_ratio_min: float = 0.25
    # Appearance corroboration, retained as reported *evidence* only. It was
    # measured against the benchmark and does not discriminate: genuine
    # translations scored 0.65-0.99 while unrelated removal/insertion pairs
    # scored 0.54-0.87, and the ranges overlap too much for a threshold.
    # Reported on each merged event; deliberately not a gate.
    translation_appearance_min: float = 0.62
    # Minimum confidence for an event to be reported as a definite kind.
    min_confidence: float = 0.05
    # Retained so previously serialised configs still load; superseded by the
    # comparative presence measure.
    structure_ratio: float = 1.22


@dataclass(frozen=True)
class PipelineConfig:
    """End-to-end configuration for a single registration + diff run."""

    features: FeatureConfig = field(default_factory=FeatureConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    diff: DiffConfig = field(default_factory=DiffConfig)
    coverage: CoverageConfig = field(default_factory=CoverageConfig)
    events: EventConfig = field(default_factory=EventConfig)
    # Determinism: seed every stochastic step.
    seed: int = 20261026
    # Free-text label for the machine a run happened on, recorded in the ledger as
    # provenance - for example "x86-laptop" or "arm64-server". Purely descriptive:
    # it is never read by the math, and a run with a different label produces
    # byte-identical output.
    executor: str = "baseline-local"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        return cls(
            features=FeatureConfig(**data.get("features", {})),
            geometry=GeometryConfig(**data.get("geometry", {})),
            diff=DiffConfig(**data.get("diff", {})),
            coverage=CoverageConfig(**data.get("coverage", {})),
            events=EventConfig(**data.get("events", {})),
            seed=data.get("seed", 20261026),
            executor=data.get("executor", "baseline-local"),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PipelineConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")
