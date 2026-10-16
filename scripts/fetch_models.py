"""Fetch the neural feature weights.

The pipeline prefers ALIKED + LightGlue for matching, because the whole
difficulty of comparing two site visits is viewpoint and lighting change. Those
weights are about 50 MB and are deliberately not committed: a repository that
carries binaries is a repository that gets slow to clone, and the pipeline has a
working fallback without them.

So this script is a convenience, not a prerequisite. Skip it and the pipeline
uses SIFT + FLANN and says so in its output; run it and the neural path is used
instead. Nothing else changes.

    python scripts/fetch_models.py            # download what is missing
    python scripts/fetch_models.py --check    # report and exit non-zero if missing
    python scripts/fetch_models.py --force    # re-download everything

Sources are the ONNX exports of the ALIKED and LightGlue reference implementations,
hosted on Hugging Face. The digests below were taken from the files this project
was built and measured against, so a download that differs is rejected rather than
quietly used - a substituted or truncated model still loads and still produces
plausible-looking matches, which is a far worse failure than an error.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vision"))

from strata_core.config import FeatureConfig  # noqa: E402

BASE_URL = "https://huggingface.co/bukuroo/ALIKED-LightGlue-ONNX/resolve/main"

# Name -> SHA-256 of the file this project was built and evaluated against.
MODELS: dict[str, str] = {
    "aliked-n16rot-top2k-640.onnx": (
        "d62b44a5a7b923c60f5351d641e0199bb03fbff49c82eca7de34ec87df0f08eb"
    ),
    "lightglue_for_aliked.onnx": (
        "33fffedd24f39f25b139fb66f9090481d276799cef7b0ea56eb6bc0986987c38"
    ),
}


def model_dir() -> Path:
    cfg = FeatureConfig()
    return (ROOT / "vision" / cfg.model_dir).resolve()


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def download(name: str, target: Path, *, force: bool = False) -> bool:
    """Fetch one model. Returns True when the file is present and verified."""
    expected = MODELS[name]
    if target.is_file() and not force:
        if sha256(target) == expected:
            print(f"  present  {name}  (digest verified)")
            return True
        print(f"  re-fetch {name}  (digest mismatch)")

    url = f"{BASE_URL}/{name}"
    print(f"  fetching {name}")
    print(f"           {url}")
    tmp = target.with_suffix(target.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, tmp.open("wb") as out:
            shutil.copyfileobj(response, out, length=1 << 20)
    except urllib.error.HTTPError as exc:
        print(f"           failed: HTTP {exc.code} {exc.reason}", file=sys.stderr)
        tmp.unlink(missing_ok=True)
        return False
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"           failed: {exc}", file=sys.stderr)
        tmp.unlink(missing_ok=True)
        return False

    got = sha256(tmp)
    if got != expected:
        print(f"           digest mismatch: expected {expected}, got {got}", file=sys.stderr)
        tmp.unlink(missing_ok=True)
        return False

    tmp.replace(target)
    print(f"           saved {target.stat().st_size / 1e6:.1f} MB")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="report only; do not download")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    directory = model_dir()
    print(f"model directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)

    if args.check:
        missing = [n for n in MODELS if not (directory / n).is_file()]
        for name in MODELS:
            path = directory / name
            state = "present" if path.is_file() else "MISSING"
            size = f"{path.stat().st_size / 1e6:.1f} MB" if path.is_file() else "-"
            print(f"  {state:8s} {name}  {size}")
        if missing:
            print(
                "\nThe neural backend will not be used. The pipeline still runs on "
                "SIFT + FLANN; run this script without --check to fetch the weights.",
                file=sys.stderr,
            )
            return 1
        return 0

    ok = True
    for name in MODELS:
        ok &= download(name, directory / name, force=args.force)

    if not ok:
        print(
            "\nSome weights could not be fetched. The pipeline remains fully usable: "
            "it falls back to SIFT + FLANN and reports which backend it used.",
            file=sys.stderr,
        )
        return 1

    print("\nAll weights present. The neural backend will be selected automatically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
