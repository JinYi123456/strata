"""Build the archive Devpost's "Upload a File" field asks for.

The rules accept "a public or private judge-accessible code repository or
archive". The repository is public and linked, so this file is the second half of
that sentence - a reviewer who cannot or does not want to clone should still be
able to read everything and run the tests.

It is built from ``git ls-files``, so it is exactly what the repository tracks:
no ``node_modules``, no ``.git``, no scratch output. The demo video is excluded by
size (it is on YouTube and served by the app), and that exclusion is stated in
the manifest rather than left for someone to notice.

    python scripts/make_submission_archive.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Excluded from the archive, with the reason recorded in MANIFEST. A reviewer who
# finds a file missing and no note about it has found a gap; the same omission
# with a line explaining it is a decision.
EXCLUDE_SUFFIXES = (".mp4", ".onnx")
EXCLUDE_PREFIXES = ("service/static/media/",)


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return [line for line in out.stdout.splitlines() if line.strip()]


def keep(path: str) -> bool:
    if path.endswith(EXCLUDE_SUFFIXES):
        return False
    return not any(path.startswith(prefix) for prefix in EXCLUDE_PREFIXES)


def build(dest: Path) -> dict:
    files = [f for f in tracked_files() if keep(f)]
    skipped = [f for f in tracked_files() if not keep(f)]
    dest.parent.mkdir(parents=True, exist_ok=True)

    manifest = "\n".join([
        "STRATA - submission archive",
        f"Built {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"Commit {subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()}",
        "",
        "Repository: https://github.com/JinYi123456/strata",
        "Start with submission/README.md (the review guide), then",
        "docs/TECHNICAL_REPORT.md.",
        "",
        "Run it:",
        "  pip install -r requirements.txt",
        "  python scripts/serve.py        -> http://127.0.0.1:8000",
        "",
        f"Files: {len(files)}",
        f"Excluded: {len(skipped)} ({', '.join(skipped) or 'none'})",
        "  The demo video is excluded by size and is published separately;",
        "  the ONNX feature weights are fetched with scripts/fetch_models.py",
        "  and are not committed to the repository.",
        "",
    ]).encode("utf-8")

    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.writestr("MANIFEST.txt", manifest)
        for name in files:
            source = ROOT / name
            if source.is_file():
                zf.write(source, name)

    return {
        "path": dest,
        "files": len(files),
        "excluded": skipped,
        "bytes": dest.stat().st_size,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "build" / "strata-submission.zip")
    args = ap.parse_args()

    info = build(args.out)
    print(f"  files    {info['files']}")
    print(f"  excluded {', '.join(info['excluded']) or 'none'}")
    print(f"  archive  {info['path']}  {info['bytes'] / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
