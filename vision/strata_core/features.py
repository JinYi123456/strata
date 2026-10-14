"""Pluggable local-feature backends.

Two families are supported behind one interface:

* **Neural** - OpenCV 5's ``features`` module: ALIKED for detection and
  description, LightGlue for attention-based matching. These give far more
  robust matches under real viewpoint and lighting change, which is the whole
  difficulty of comparing two site visits.
* **Classic** - SIFT + FLANN ratio-test matching. Requires no model download
  and is the guaranteed-works default, so the pipeline never becomes
  undeliverable because a 50 MB weight file was unavailable.

``resolve_backend`` picks the best available option for ``backend="auto"`` and
always reports which one it chose, so provenance is never implicit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import cv2
import numpy as np

from .config import FeatureConfig

log = logging.getLogger("strata.features")

BackendName = Literal["neural-lightglue", "classic-sift", "classic-orb"]


@dataclass
class FeatureSet:
    """Keypoints and descriptors from one image."""

    keypoints: list[cv2.KeyPoint]
    descriptors: np.ndarray | None
    image_shape: tuple[int, int]

    @property
    def points(self) -> np.ndarray:
        """Keypoint locations as an ``(N, 2)`` float32 array."""
        if not self.keypoints:
            return np.zeros((0, 2), dtype=np.float32)
        return np.asarray([kp.pt for kp in self.keypoints], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.keypoints)


@dataclass
class MatchSet:
    """A set of putative correspondences between two images."""

    src_points: np.ndarray  # (N, 2) float32
    dst_points: np.ndarray  # (N, 2) float32
    scores: np.ndarray | None = None
    backend: str = "unknown"

    def __len__(self) -> int:
        return int(self.src_points.shape[0])


def _model_path(model_dir: str | Path, name: str) -> Path | None:
    for base in _candidate_dirs(model_dir):
        candidate = base / name
        if candidate.is_file():
            return candidate
    return None


def _candidate_dirs(model_dir: str | Path) -> list[Path]:
    """Directories that may hold the ONNX weights, nearest first."""
    here = Path(__file__).resolve()
    roots = [
        Path(model_dir),
        here.parent.parent / model_dir,          # vision/models
        here.parent.parent.parent / model_dir,   # repo/models
    ]
    seen: list[Path] = []
    for r in roots:
        rp = r if r.is_absolute() else (Path.cwd() / r)
        if rp not in seen:
            seen.append(rp)
    return seen


def neural_available(cfg: FeatureConfig) -> bool:
    """True when both neural ONNX weights are present and loadable."""
    if not (hasattr(cv2, "ALIKED_create") and hasattr(cv2, "LightGlueMatcher_create")):
        return False
    aliked = _model_path(cfg.model_dir, cfg.aliked_model)
    lightglue = _model_path(cfg.model_dir, cfg.lightglue_model)
    return aliked is not None and lightglue is not None


def resolve_backend(cfg: FeatureConfig) -> BackendName:
    """Decide which backend a run will actually use."""
    if cfg.backend == "classic":
        return "classic-sift"
    if cfg.backend == "neural":
        if not neural_available(cfg):
            raise RuntimeError(
                "neural backend requested but ONNX weights are missing; "
                "run scripts/fetch_models.py or use backend='classic'"
            )
        return "neural-lightglue"
    # auto
    return "neural-lightglue" if neural_available(cfg) else "classic-sift"


class ClassicExtractor:
    """SIFT extraction with an ORB fallback if contributed modules are absent."""

    def __init__(self, cfg: FeatureConfig) -> None:
        self.cfg = cfg
        self._used_orb = False
        try:
            self.detector = cv2.SIFT_create(
                nfeatures=cfg.max_features,
                contrastThreshold=cfg.sift_contrast_threshold,
                edgeThreshold=cfg.sift_edge_threshold,
            )
        except Exception:  # pragma: no cover - defensive
            log.warning("SIFT unavailable, falling back to ORB")
            self.detector = cv2.ORB_create(nfeatures=cfg.max_features)
            self._used_orb = True

    @property
    def name(self) -> BackendName:
        return "classic-orb" if self._used_orb else "classic-sift"

    def detect(self, gray: np.ndarray) -> FeatureSet:
        kps, desc = self.detector.detectAndCompute(gray, None)
        kps = kps or []
        if desc is not None and len(kps) > self.cfg.max_features:
            idx = np.argsort([-k.response for k in kps])[: self.cfg.max_features]
            kps = [kps[i] for i in idx]
            desc = desc[idx]
        return FeatureSet(keypoints=list(kps), descriptors=desc, image_shape=gray.shape[:2])


class NeuralExtractor:
    """ALIKED detector/descriptor from OpenCV 5's features module."""

    def __init__(self, cfg: FeatureConfig) -> None:
        self.cfg = cfg
        path = _model_path(cfg.model_dir, cfg.aliked_model)
        if path is None:
            raise RuntimeError(f"ALIKED weights not found: {cfg.aliked_model}")
        self.model_path = path
        self.detector = cv2.ALIKED_create(str(path))

    @property
    def name(self) -> BackendName:
        return "neural-lightglue"

    def detect(self, gray: np.ndarray) -> FeatureSet:
        kps, desc = self.detector.detectAndCompute(gray, None)
        kps = kps or []
        if desc is not None and len(kps) > self.cfg.max_features:
            # ALIKED may return a response; if not, keep the first N.
            try:
                order = np.argsort([-k.response for k in kps])[: self.cfg.max_features]
            except Exception:
                order = np.arange(self.cfg.max_features)
            kps = [kps[i] for i in order]
            desc = desc[order]
        return FeatureSet(keypoints=list(kps), descriptors=desc, image_shape=gray.shape[:2])


def build_extractor(cfg: FeatureConfig):
    """Return the extractor matching the resolved backend."""
    backend = resolve_backend(cfg)
    if backend == "neural-lightglue":
        return NeuralExtractor(cfg)
    return ClassicExtractor(cfg)


class Matcher:
    """Matches two :class:`FeatureSet` objects using the resolved backend.

    The neural path uses :class:`cv2.LightGlueMatcher`, which returns a
    confidence for every correspondence. The classic path uses FLANN with
    Lowe's ratio test. Both return the same :class:`MatchSet` shape so the
    downstream geometry is backend-agnostic - and so a report can honestly
    state which one produced a given number.
    """

    def __init__(self, cfg: FeatureConfig, backend: BackendName | None = None) -> None:
        self.cfg = cfg
        self.backend = backend or resolve_backend(cfg)
        self._lightglue = None
        self._flann = None
        if self.backend == "neural-lightglue":
            path = _model_path(cfg.model_dir, cfg.lightglue_model)
            if path is None:
                raise RuntimeError(f"LightGlue weights not found: {cfg.lightglue_model}")
            self.model_path = path
            self._lightglue = cv2.LightGlueMatcher_create(str(path))
        else:
            index_params = dict(algorithm=1, trees=5)  # KD-tree for SIFT
            self._flann = cv2.FlannBasedMatcher(index_params, dict(checks=64))

    def match(self, a: FeatureSet, b: FeatureSet) -> MatchSet:
        if self.backend == "neural-lightglue":
            return self._match_neural(a, b)
        return self._match_classic(a, b)

    def _match_neural(self, a: FeatureSet, b: FeatureSet) -> MatchSet:
        if a.descriptors is None or b.descriptors is None or len(a) == 0 or len(b) == 0:
            return MatchSet(
                src_points=np.zeros((0, 2), np.float32),
                dst_points=np.zeros((0, 2), np.float32),
                backend="neural-lightglue",
            )
        try:
            # LightGlue needs explicit spatial context: Nx2 float keypoint
            # matrices plus both image sizes. Without setPairInfo it raises
            # "no valid context".
            ah, aw = a.image_shape
            bh, bw = b.image_shape
            self._lightglue.setPairInfo(
                a.points.astype(np.float32),
                b.points.astype(np.float32),
                (int(aw), int(ah)),
                (int(bw), int(bh)),
            )
            matches = self._lightglue.match(
                a.descriptors.astype(np.float32), b.descriptors.astype(np.float32)
            )
        except Exception as exc:
            log.warning("LightGlue match failed (%s); using classic path", exc)
            return self._match_classic(a, b, backend_override="classic-sift")

        src, dst, scores = [], [], []
        for m in matches:
            si = int(getattr(m, "queryIdx", -1))
            di = int(getattr(m, "trainIdx", -1))
            if si < 0 or di < 0 or si >= len(a) or di >= len(b):
                continue
            src.append(a.keypoints[si].pt)
            dst.append(b.keypoints[di].pt)
            scores.append(float(getattr(m, "distance", 0.0)))
        return MatchSet(
            src_points=np.asarray(src, np.float32).reshape(-1, 2),
            dst_points=np.asarray(dst, np.float32).reshape(-1, 2),
            scores=np.asarray(scores, np.float32) if scores else None,
            backend="neural-lightglue",
        )

    def _match_classic(
        self, a: FeatureSet, b: FeatureSet, backend_override: str | None = None
    ) -> MatchSet:
        name = backend_override or self.backend
        if a.descriptors is None or b.descriptors is None or len(a) < 2 or len(b) < 2:
            return MatchSet(
                src_points=np.zeros((0, 2), np.float32),
                dst_points=np.zeros((0, 2), np.float32),
                backend=name,
            )
        try:
            is_float = a.descriptors.dtype in (np.float32, np.float64)
            if is_float:
                if self._flann is None:
                    index_params = dict(algorithm=1, trees=5)  # KD-tree
                    self._flann = cv2.FlannBasedMatcher(index_params, dict(checks=64))
                knn = self._flann.knnMatch(
                    a.descriptors.astype(np.float32), b.descriptors.astype(np.float32), k=2
                )
            else:
                # Binary descriptors (ORB / ALIKED uint8) need Hamming distance.
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                knn = bf.knnMatch(a.descriptors, b.descriptors, k=2)
        except Exception as exc:
            log.warning("classic match failed (%s)", exc)
            return MatchSet(
                src_points=np.zeros((0, 2), np.float32),
                dst_points=np.zeros((0, 2), np.float32),
                backend=name,
            )
        src, dst, scores = [], [], []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair[0], pair[1]
            if m.distance < self.cfg.ratio_test * n.distance:
                src.append(a.keypoints[m.queryIdx].pt)
                dst.append(b.keypoints[m.trainIdx].pt)
                scores.append(float(m.distance))
        return MatchSet(
            src_points=np.asarray(src, np.float32).reshape(-1, 2),
            dst_points=np.asarray(dst, np.float32).reshape(-1, 2),
            scores=np.asarray(scores, np.float32) if scores else None,
            backend=name,
        )


def to_gray(image: np.ndarray) -> np.ndarray:
    """Convert BGR/gray to 8-bit single channel."""
    if image.ndim == 2:
        gray = image
    elif image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray
