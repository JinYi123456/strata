"""The documentation must not disagree with the measurement it documents.

The report, the README and the review guide all quote figures from
``docs/evaluation.json``. They were hand-copied, and two of them had drifted -
the prose said "nine" perfect cases and "seven" merged-event cases where the
committed run has ten of each. For a project whose stated position is that it
reports the numbers that hurt, a stale count in its own limitations section is
the worst kind of error: it reads as carelessness about the thing it claims to
care about.

So the figures are now derived from the artefact, and these tests fail if any
quoted number stops matching it.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from figures import derived  # noqa: E402

ARTIFACT = ROOT / "docs" / "evaluation.json"


@pytest.fixture(scope="module")
def artifact() -> dict:
    assert ARTIFACT.is_file(), "docs/evaluation.json is committed; it should be present"
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def got(artifact: dict) -> dict:
    return derived(artifact)


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


class TestDerivedFigures:
    def test_counts_come_from_the_run_not_from_memory(self, got):
        assert got["cases"] == 24
        assert got["still_pairs"] == 12
        assert got["perfect_cases"] == sum(
            1 for c in json.loads(ARTIFACT.read_text(encoding="utf-8"))["per_case"]
            if c["detection"]["f1"] == 1.0 and c["event_kind_accuracy"] == 1.0
        )

    def test_headline_figures_are_the_artefacts_own_numbers(self, artifact, got):
        agg = artifact["aggregate"]
        assert got["table"]["registration"] == f"{agg['mean_registration_error_px']:.3f}"
        assert got["table"]["precision"] == f"{agg['mean_precision']:.4f}"
        assert got["table"]["recall"] == f"{agg['mean_recall']:.4f}"
        assert got["table"]["f1"] == f"{agg['mean_f1']:.4f}"
        assert got["table"]["kind"] == f"{agg['mean_event_kind_accuracy']:.4f}"

    def test_macro_and_micro_precision_differ_and_say_so(self, got):
        assert got["precision_macro"] != got["precision_micro"]
        assert got["still_pair_false_positives"] == 0.0


class TestReportQuotesTheArtifact:
    """Every metric row in the report's results table must equal the artefact."""

    TABLE_ROWS = {
        "Registration error": "registration",
        "Registration confidence": "confidence",
        "Precision": "precision",
        "Recall": "recall",
        "F1": "f1",
        "Event-kind accuracy": "kind",
    }

    @pytest.mark.parametrize("label,key", sorted(TABLE_ROWS.items()))
    def test_report_metric_row_matches(self, got, label: str, key: str):
        text = _read("docs/TECHNICAL_REPORT.md")
        row = next(
            (line for line in text.splitlines() if line.strip().startswith(f"| {label}")),
            None,
        )
        assert row, f"the report no longer has a '{label}' row"
        quoted = re.search(r"([0-9]+\.[0-9]+)", row)
        assert quoted, f"could not read a number out of: {row}"
        value = quoted.group(1)
        assert value == got["table"][key], f"{label}: report says {value}, run says {got['table'][key]}"

    def test_still_pair_row_is_zero(self, got):
        text = _read("docs/TECHNICAL_REPORT.md")
        row = next(line for line in text.splitlines() if "False positives on still pairs" in line)
        assert "0.0" in row and got["still_pair_false_positives"] == 0.0

    def test_perfect_case_count_is_stated_correctly(self, got):
        """The sentence that had drifted."""
        for rel in ("docs/TECHNICAL_REPORT.md", "README.md"):
            text = _read(rel)
            found = re.findall(r"(\w+)\s+of the 24 cases score a perfect", text)
            for word in found:
                assert _word_to_int(word) == got["perfect_cases"], (
                    f"{rel} says '{word} of the 24'; the run has {got['perfect_cases']}"
                )

    def test_merged_event_count_is_stated_correctly(self, got):
        text = _read("docs/TECHNICAL_REPORT.md")
        for word in re.findall(r"(\w+)\s+of 24 cases report 3 events", text):
            assert _word_to_int(word) == got["merged_cases"], (
                f"the report says '{word} of 24 cases report 3 events'; "
                f"the run has {got['merged_cases']}"
            )

    def test_the_review_guide_agrees_too(self, got):
        text = _read("submission/README.md")
        for word in re.findall(r"(\w+) of 24 cases report 3 events", text):
            assert _word_to_int(word) == got["merged_cases"]


class TestNoStaleClaims:
    """Statements the code has since contradicted must not survive in prose."""

    def test_adaptive_threshold_is_not_claimed_as_a_pipeline_stage(self):
        """The detection stage calibrates per pair; it does not call adaptiveThreshold."""
        text = _read("docs/TECHNICAL_REPORT.md")
        assert "uses `cv2.adaptiveThreshold`" not in text

    def test_the_fundamental_model_claim_states_when_it_runs(self):
        text = _read("docs/TECHNICAL_REPORT.md")
        assert "prefer_planar" in text, (
            "the report must say the second model is only estimated when "
            "prefer_planar is turned off"
        )

    def test_the_backend_claim_names_both_fields(self):
        text = _read("README.md")
        assert "match_backend" in text or "matcher that ran" in text.lower()


def _word_to_int(word: str) -> int:
    words = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
        "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    }
    assert word.lower() in words, f"unexpected number word: {word!r}"
    return words[word.lower()]
