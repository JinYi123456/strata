"""Fail when a measurement regresses.

The evaluation is reproducible to the pixel, which makes it a useful tripwire:
a change that quietly costs recall should not be mergeable. This reads the
artefact ``scripts/evaluate.py`` writes and compares it against floors that are
deliberately below the current numbers, so it catches a regression without
freezing the result or punishing an honest improvement elsewhere.

    python scripts/gate.py                       # check docs/evaluation.json
    python scripts/gate.py --artifact path.json
    python scripts/gate.py --min-f1 0.80

Exit code 0 means every floor held.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from figures import derived  # noqa: E402

# Floors, not targets. Each is comfortably below what the current build measures,
# so a failure means something broke rather than something moved by a hair.
FLOORS = {
    "f1_macro": 0.85,
    "precision_macro": 0.90,
    "recall_macro": 0.80,
    "kind": 0.60,
    "still_pair_false_positives": None,   # handled separately: must be exactly 0
}
MAX_REGISTRATION_ERROR_PX = 1.00


def check(d: dict) -> list[str]:
    failures: list[str] = []
    for key, floor in FLOORS.items():
        if floor is None:
            continue
        value = float(d[key])
        if value < floor:
            failures.append(f"{key} {value:.4f} is below its floor {floor:.2f}")
    if float(d["still_pair_false_positives"]) > 0.0:
        failures.append(
            f"still pairs produced {d['still_pair_false_positives']} confident false "
            "positives per pair; the claim under test is that they produce none"
        )
    if float(d["registration_px"]) > MAX_REGISTRATION_ERROR_PX:
        failures.append(
            f"mean registration error {d['registration_px']:.3f}px exceeds "
            f"{MAX_REGISTRATION_ERROR_PX:.2f}px"
        )
    return failures


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artifact", type=Path, default=ROOT / "docs" / "evaluation.json")
    ap.add_argument("--min-f1", type=float, default=None,
                    help="raise or lower the F1 floor for this run")
    args = ap.parse_args(argv)

    if not args.artifact.is_file():
        print(f"no artefact at {args.artifact}; run scripts/evaluate.py first",
              file=sys.stderr)
        return 2

    d = derived(json.loads(args.artifact.read_text(encoding="utf-8")))
    if args.min_f1 is not None:
        FLOORS["f1_macro"] = args.min_f1

    for key in ("registration_px", "precision_macro", "recall_macro", "f1_macro",
                "kind", "still_pair_false_positives", "perfect_cases", "merged_cases"):
        print(f"{key:>28}  {d[key]}")

    failures = check(d)
    if failures:
        print("\nFAIL")
        for line in failures:
            print(f"  - {line}")
        return 1
    print("\nPASS  every floor held")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
