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
    # Mean absolute Laplacian of the source image: a cheap scalar summary of how
    # much matchable structure it contains. Carried on the feature set because
    # the matcher needs it but only the extractor has the image, and recomputing
    # it downstream would mean holding a second copy of every frame.
    #
    # It exists to prevent a hard crash. On a featureless image - a wall, a lens
    # cap, an over-exposed frame - ALIKED still returns a full set of keypoints
    # with statistically normal descriptors, so no amount of inspecting the
    # *features* reveals the problem. LightGlue, however, normalises over the
    # scene's structure and divides by zero when there is none, which on Windows
    # surfaces as a fatal C-level exception that Python cannot catch: the
    # process dies, taking the service with it. Measuring the image itself is
    # the only reliable way to detect the condition before the call.
    #
    # The default of 0.0 is a deliberate fail-safe, not a placeholder: a
    # FeatureSet that forgot to carry the measurement is treated as structureless
    # and its pair is refused, rather than being allowed to reach a call that can
    # kill the process. Refusing a comparison is recoverable; a dead service is
    # not. Both extractors set this explicitly.
    structure_energy: float = 0.0

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
    # True when the pair was too low on structure for the neural matcher, so
    # these correspondences came from the classical fallback on a scene with
    # little to match. Measured consequence: such pairs can still yield a
    # geometrically *consistent* set of correspondences that has nothing to do
    # with reality - two unrelated thin images matched at confidence 0.95 - so
    # the flag travels with the matches and the geometry stage discounts the
    # result rather than trusting it.
    low_structure: bool = False

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


def _structure_energy(gray: np.ndarray) -> float:
    """Mean absolute Laplacian: how much matchable structure an image contains.

    Measured values on the synthetic scenes and their degenerate neighbours:
    a uniform frame gives exactly 0.0, a single small square 0.43, faint sensor
    noise around 5.0, and a real textured scene 15-20. The scale is therefore
    meaningful in absolute terms, which is what makes a fixed floor defensible
    rather than fitted.
    """
    if gray.size == 0:
        return 0.0
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    return float(np.mean(np.abs(lap)))


# Two thresholds, each anchored to a measured point rather than tuned.
#
# Below this an image has no structure at all - a uniform frame measures exactly
# 0.0 - so there is provably nothing to correspond. Matching would be fitting a
# transform to noise, which risks inventing a registration between two blank
# walls, so the pair is refused outright and reported as such.
_NO_STRUCTURE_FLOOR = 0.05

# Between the floor and this, content exists but is thin enough that LightGlue's
# normalisation is unsafe: a single small square measures 0.43. Such pairs go to
# the classical matcher, which has no such precondition and handles a handful of
# real corners correctly. Real scenes sit at 15-20 and faint sensor noise at 5.0,
# so nothing with ordinary texture is diverted.
_MIN_MATCH_STRUCTURE = 1.0

StructureVerdict = Literal["ok", "thin", "flat"]


def _structure_verdict(a: "FeatureSet", b: "FeatureSet") -> StructureVerdict:
    """Whether LightGlue may be called on this pair, and why not if it may not.

    Kept separate from the matcher so the decision can be tested without
    loading a model, and so the thresholds have exactly one home. The pair is
    judged by its *weaker* frame: LightGlue normalises over the pair, so one
    featureless frame is enough to make the call unsafe.
    """
    weakest = min(a.structure_energy, b.structure_energy)
    if weakest < _NO_STRUCTURE_FLOOR:
        return "flat"
    if weakest < _MIN_MATCH_STRUCTURE:
        return "thin"
    return "ok"


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
        return FeatureSet(
            keypoints=list(kps),
            descriptors=desc,
            image_shape=gray.shape[:2],
            structure_energy=_structure_energy(gray),
        )


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
        return FeatureSet(
            keypoints=list(kps),
            descriptors=desc,
            image_shape=gray.shape[:2],
            structure_energy=_structure_energy(gray),
        )


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

    A pair with no structure is diverted away from LightGlue before the call
    (see :data:`_MIN_MATCH_STRUCTURE`), and a pair with none at all is refused.
    The diverted cases still report which path ran, so a result can always be
    traced back to the matcher that produced it.
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
        # This guard has to sit *before* the call, not around it. On a
        # featureless image LightGlue divides by zero, and on Windows that is a
        # fatal C-level exception: no Python handler runs, the interpreter dies
        # and takes the service process with it. A try/except cannot protect
        # against a failure that never unwinds the stack.
        #
        # It also has to sit before the descriptor check below, so that the
        # *reason* a pair produced no matches is always reported accurately
        # rather than being masked by a more general "nothing to match".
        verdict = _structure_verdict(a, b)
        weakest = min(a.structure_energy, b.structure_energy)
        if verdict == "flat":
            log.warning(
                "refusing to match featureless pair (structure energy %.4f)", weakest
            )
            return MatchSet(
                src_points=np.zeros((0, 2), np.float32),
                dst_points=np.zeros((0, 2), np.float32),
                backend="degenerate-flat",
                low_structure=True,
            )
        if verdict == "thin":
            log.info(
                "thin structure (%.4f); matching with the classical backend", weakest
            )
            return self._match_classic(a, b, backend_override="classic-lowstructure")

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
        # The override name is how the low-structure diversion is signalled, so
        # the flag is derived from it rather than passed separately - one source
        # of truth for "this pair was matched on almost nothing".
        low = backend_override == "classic-lowstructure"
        if a.descriptors is None or b.descriptors is None or len(a) < 2 or len(b) < 2:
            return MatchSet(
                src_points=np.zeros((0, 2), np.float32),
                dst_points=np.zeros((0, 2), np.float32),
                backend=name,
                low_structure=low,
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
                low_structure=low,
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
            low_structure=low,
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
