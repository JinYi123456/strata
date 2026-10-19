"""Derive every published figure from the evaluation artefact.

    python scripts/figures.py                 # the numbers, and the sentences
    python scripts/figures.py --json          # machine-readable

The report, the README and the review guide all quote these figures. They were
maintained by hand, and two drifted. Deriving them is the difference between
documentation that is *probably* right and documentation that cannot be wrong
without a test failing (see ``tests/test_docs_agree_with_artifact.py``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = ROOT / "docs" / "evaluation.json"

# The IoU at which a reported box counts as the same change as an expected one,
# and the confidence floor the still-pair false-positive count is measured above.
# Both were previously disclosed nowhere, which left a reader unable to tell how
# generous the scoring was.
IOU_THRESHOLD = 0.25
STILL_PAIR_CONFIDENCE_FLOOR = 0.35


def derived(artifact: dict) -> dict:
    agg = artifact["aggregate"]
    cases = artifact["per_case"]
    still = artifact["still_pair"]

    tp = sum(c["detection"]["true_positives"] for c in cases)
    fp = sum(c["detection"]["false_positives"] for c in cases)
    fn = sum(c["detection"]["false_negatives"] for c in cases)
    micro_p = tp / (tp + fp) if (tp + fp) else 0.0
    micro_r = tp / (tp + fn) if (tp + fn) else 0.0
    micro_f1 = (
        2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
    )

    out = {
        "cases": int(agg["cases"]),
        "still_pairs": int(still["pairs"]),
        "registration_px": float(agg["mean_registration_error_px"]),
        "confidence": float(agg["mean_registration_confidence"]),
        "precision_macro": float(agg["mean_precision"]),
        "recall_macro": float(agg["mean_recall"]),
        "f1_macro": float(agg["mean_f1"]),
        "kind": float(agg["mean_event_kind_accuracy"]),
        "precision_micro": micro_p,
        "recall_micro": micro_r,
        "f1_micro": micro_f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "still_pair_false_positives": float(still["confident_false_positives_per_pair"]),
        "perfect_cases": sum(
            1 for c in cases
            if c["detection"]["f1"] == 1.0 and c["event_kind_accuracy"] == 1.0
        ),
        "merged_cases": sum(1 for c in cases if c["reported_events"] < c["expected_events"]),
        "iou_threshold": IOU_THRESHOLD,
        "still_pair_confidence_floor": STILL_PAIR_CONFIDENCE_FLOOR,
        "elapsed_seconds": float(artifact.get("elapsed_seconds", 0.0)),
    }
    for key, digits in (
        ("registration_px", 3), ("confidence", 3), ("precision_macro", 4),
        ("recall_macro", 4), ("f1_macro", 4), ("kind", 4),
        ("precision_micro", 4), ("recall_micro", 4), ("f1_micro", 4),
    ):
        out[f"{key}_{digits}dp"] = f"{out[key]:.{digits}f}"
    # The published table used to round to three decimals, and 0.8875 does not
    # round the same way in every tool - a float cannot hold it exactly, so
    # "0.888" and "0.887" both appear depending on the library. The table now
    # quotes four decimals, which is the artefact's own precision, and removes a
    # whole class of "the docs disagree with the run" findings.
    out["table"] = {
        "registration": f"{out['registration_px']:.3f}",
        "confidence": f"{out['confidence']:.4f}",
        "precision": f"{out['precision_macro']:.4f}",
        "recall": f"{out['recall_macro']:.4f}",
        "f1": f"{out['f1_macro']:.4f}",
        "kind": f"{out['kind']:.4f}",
    }
    return out


def sentences(d: dict) -> list[tuple[str, str]]:
    """The prose the docs quote, generated rather than retyped."""
    return [
        (
            "perfect-cases",
            f"{_word(d['perfect_cases'])} of the {d['cases']} cases score a perfect "
            f"F1 = 1.000 with kind = 1.000.",
        ),
        (
            "merged-cases",
            f"{_word(d['merged_cases']).capitalize()} of {d['cases']} cases report "
            f"fewer events than expected - one change is merged away.",
        ),
        (
            "scoring-disclosure",
            f"Precision, recall and F1 are macro averages over {d['cases']} cases; "
            f"the micro averages are P {d['precision_micro']:.4f}, "
            f"R {d['recall_micro']:.4f}, F1 {d['f1_micro']:.4f}. A reported box "
            f"counts as the same change as an expected one at IoU >= "
            f"{d['iou_threshold']:.2f}.",
        ),
        (
            "still-pairs",
            f"{d['still_pairs']} unchanged pairs produced "
            f"{d['still_pair_false_positives']:.1f} confident false positives per pair "
            f"(confident = a typed event at confidence >= "
            f"{d['still_pair_confidence_floor']:.2f}).",
        ),
    ]


_WORDS = "zero one two three four five six seven eight nine ten eleven twelve".split()


def _word(n: int) -> str:
    return _WORDS[n] if 0 <= n < len(_WORDS) else str(n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    d = derived(json.loads(args.artifact.read_text(encoding="utf-8")))
    if args.json:
        print(json.dumps(d, indent=2))
        return 0

    for label, value in d["table"].items():
        print(f"{label:>14}  {value}")
    print()
    print(f"  perfect cases  {d['perfect_cases']} of {d['cases']}")
    print(f"  merged cases   {d['merged_cases']} of {d['cases']}")
    print(f"  micro          P {d['precision_micro']:.4f} R {d['recall_micro']:.4f} "
          f"F1 {d['f1_micro']:.4f}")
    print(f"  still pairs    {d['still_pair_false_positives']:.1f} confident false "
          f"positives per pair over {d['still_pairs']} pairs")
    print()
    for name, text in sentences(d):
        print(f"[{name}]\n  {text}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
