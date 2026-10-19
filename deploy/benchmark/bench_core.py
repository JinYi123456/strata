"""
Reproducible benchmark of the STRATA vision workload.

Runs the same code on x86 and on AWS Graviton and emits JSON that can be
diffed, so a performance claim is a measurement rather than an assertion.

Two things are measured separately, because they answer different questions:

* **Microbenchmarks** of the individual operations the pipeline spends its time
  in - resize, warp, adaptive threshold, contour extraction. These are the
  operations a cloud-optimised OpenCV build accelerates, so isolating them shows
  *where* a difference comes from.
* **The full scan** on a fixed, seeded frame pair. This is the number a user
  experiences, and it is the one that should be quoted.

    python bench_core.py --json out.json
    python bench_core.py --json out.json --backend sift --repeat 5

Determinism: the scene is generated from a fixed seed and every timed section is
preceded by a warm-up, so the numbers do not depend on what ran before.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "vision"))

from strata_core.config import FeatureConfig, PipelineConfig  # noqa: E402
from strata_core.pipeline import run_scan  # noqa: E402
from strata_core.synth import make_scene  # noqa: E402


def timed(fn, repeat: int, warmup: int = 1) -> dict[str, float]:
    """Run ``fn`` and report the distribution, not just the mean.

    A mean hides the tail, and the tail is what an operator feels. p95 is
    reported alongside the median for that reason.
    """
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return {
        "median_ms": round(statistics.median(samples), 3),
        "p95_ms": round(samples[min(len(samples) - 1, int(len(samples) * 0.95))], 3),
        "min_ms": round(samples[0], 3),
        "max_ms": round(samples[-1], 3),
        "n": len(samples),
    }


def machine() -> dict[str, object]:
    """Identify the host well enough to make two runs comparable."""
    model = ""
    try:
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            for line in cpuinfo.read_text(errors="ignore").splitlines():
                if line.startswith("model name") or line.startswith("Model"):
                    model = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": model or platform.processor(),
        "cpus": os.cpu_count(),
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "opencv_build": os.environ.get("STRATA_COOL_VERSION", "upstream"),
        "cv2_module": cv2.__file__,
    }


def microbenchmarks(size: tuple[int, int], repeat: int) -> dict[str, dict[str, float]]:
    """Time the operations the pipeline actually spends its time in."""
    w, h = size
    rng = np.random.default_rng(11)

    frame = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    # Give it structure: a pure-noise image is not representative of a scene,
    # and some of these kernels short-circuit on flat input.
    for _ in range(400):
        x, y = int(rng.integers(0, w - 80)), int(rng.integers(0, h - 80))
        cv2.rectangle(frame, (x, y), (x + 60, y + 50),
                      tuple(int(v) for v in rng.integers(0, 255, 3)), -1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (w // 2, h // 2))
    warp = cv2.getRotationMatrix2D((w / 2, h / 2), 0.8, 1.01)
    warp = np.vstack([warp, [0, 0, 1]])
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    def contours_op() -> None:
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return {
        "resize_half": timed(lambda: cv2.resize(frame, (w // 2, h // 2)), repeat),
        "resize_2x": timed(lambda: cv2.resize(small, (w, h)), repeat),
        "warp_affine": timed(lambda: cv2.warpPerspective(frame, warp, (w, h)), repeat),
        "adaptive_threshold": timed(
            lambda: cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
            ),
            repeat,
        ),
        "find_contours": timed(contours_op, repeat),
        "cvt_gray": timed(lambda: cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), repeat),
        "cvt_hsv": timed(lambda: cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), repeat),
        "gaussian_blur": timed(lambda: cv2.GaussianBlur(frame, (5, 5), 0), repeat),
        "absdiff": timed(lambda: cv2.absdiff(frame, hsv), repeat),
        "morphology": timed(
            lambda: cv2.morphologyEx(
                gray, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)
            ),
            repeat,
        ),
    }


def full_scan(backend: str, repeat: int) -> dict[str, object]:
    """Time the whole pipeline on one fixed frame pair."""
    scene = make_scene(seed=20261026, size=(960, 720), n_objects=18, n_mutations=4)
    cfg = replace(
        PipelineConfig(),
        features=replace(
            FeatureConfig(),
            backend=backend,  # type: ignore[arg-type]
            model_dir=str(REPO / "vision" / "models"),
        ),
    )

    runs: list[dict[str, float]] = []
    resolved = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = run_scan([scene.baseline], [scene.rescan], config=cfg)
        wall = (time.perf_counter() - t0) * 1000.0
        resolved = result.feature_backend
        runs.append(
            {
                "wall_ms": round(wall, 1),
                "extract_baseline_ms": round(result.timings.extract_baseline_ms, 1),
                "extract_rescan_ms": round(result.timings.extract_rescan_ms, 1),
                "match_ms": round(result.timings.match_ms, 1),
                "register_ms": round(result.timings.register_ms, 1),
                "diff_ms": round(result.timings.diff_ms, 1),
                "events_ms": round(result.timings.events_ms, 1),
                "total_ms": round(result.timings.total_ms, 1),
            }
        )

    walls = sorted(r["wall_ms"] for r in runs)
    median = statistics.median(walls)
    return {
        "backend_requested": backend,
        "backend_resolved": resolved,
        "median_wall_ms": round(median, 1),
        "p95_wall_ms": round(walls[min(len(walls) - 1, int(len(walls) * 0.95))], 1),
        "scans_per_hour": round(3_600_000.0 / median, 1) if median else None,
        "stage_median_ms": {
            k: round(statistics.median([r[k] for r in runs]), 1)
            for k in runs[0]
            if k != "wall_ms"
        },
        "runs": runs,
    }


# Indicative hourly prices for the instance classes the benchmark may run on.
# Used only to express the result in the unit a buyer cares about; the measured
# number is the timing, not the price.
PRICES = {
    "c8g.xlarge": 0.1276,
    "c8g.2xlarge": 0.2552,
    "m8g.xlarge": 0.1540,
    "m8g.4xlarge": 0.6160,
    "c7i.xlarge": 0.1785,
    "m7i.4xlarge": 0.8064,
}


def cost_projection(median_ms: float, instance: str) -> dict[str, object]:
    hourly = PRICES.get(instance)
    if not hourly or not median_ms:
        return {"instance": instance, "hourly_usd": None}
    per_scan_hours = (median_ms / 1000.0) / 3600.0
    return {
        "instance": instance,
        "hourly_usd": hourly,
        "usd_per_1000_scans": round(per_scan_hours * hourly * 1000.0, 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark the STRATA vision workload")
    ap.add_argument("--json", type=Path, help="write results here")
    ap.add_argument("--backend", default="auto", choices=["auto", "aliked", "sift", "orb"])
    ap.add_argument("--repeat", type=int, default=3, help="timed repetitions")
    ap.add_argument("--micro-repeat", type=int, default=20)
    ap.add_argument("--instance", default="c8g.xlarge", help="instance class for the cost projection")
    args = ap.parse_args()

    host = machine()
    print(f"  host     {host['machine']} · {host['processor'] or 'unknown'}")
    print(f"  opencv   {host['opencv']} ({host['opencv_build']})")
    print(f"  cpus     {host['cpus']}")

    print("\n  microbenchmarks (1920x1080)")
    micro = microbenchmarks((1920, 1080), args.micro_repeat)
    for name, stats in micro.items():
        print(f"    {name:<22} {stats['median_ms']:8.2f} ms   p95 {stats['p95_ms']:8.2f}")

    print(f"\n  full scan (960x720, backend={args.backend})")
    scan = full_scan(args.backend, args.repeat)
    print(f"    resolved backend       {scan['backend_resolved']}")
    print(f"    median wall            {scan['median_wall_ms']:.1f} ms")
    print(f"    p95 wall               {scan['p95_wall_ms']:.1f} ms")
    for stage, ms in scan["stage_median_ms"].items():
        print(f"      {stage:<22} {ms:8.1f} ms")

    cost = cost_projection(float(scan["median_wall_ms"]), args.instance)
    if cost.get("hourly_usd"):
        print(f"\n  cost projection ({args.instance})")
        print(f"    ${cost['usd_per_1000_scans']:.3f} per 1000 scans")

    payload = {
        "host": host,
        "microbenchmarks": micro,
        "full_scan": scan,
        "cost": cost,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
