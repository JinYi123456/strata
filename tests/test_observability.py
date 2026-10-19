"""The structured log the deployment document promises, and what it must never carry.

Per-stage timings in the logs are an observability claim; a claim about logs is
verified by reading a log line. The redaction tests exist because the first
version of the formatter renamed a secret's *key* and left the value in place,
which is a leak with better spelling.
"""

from __future__ import annotations

import io
import json
import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from service.observability import (  # noqa: E402
    JsonLogFormatter,
    configure_logging,
    log_fields,
    scan_log_fields,
)


def _emit(**fields) -> str:
    record = logging.LogRecord("strata.test", logging.INFO, "p", 1, "scan completed", (), None)
    record.fields = fields
    return JsonLogFormatter().format(record)


def _parse(line: str) -> dict:
    return json.loads(line)


class TestJsonLines:
    def test_a_line_is_one_parseable_object(self):
        payload = _parse(_emit(total_ms=1234.5, event_count=4))
        assert payload["message"] == "scan completed"
        assert payload["level"] == "INFO"
        assert payload["total_ms"] == 1234.5
        assert payload["event_count"] == 4
        assert payload["ts"].endswith("Z")

    def test_keys_are_lowercased_and_values_survive(self):
        payload = _parse(_emit(Registration_Model="homography", stage_ms={"diff_ms": 12.0}))
        assert payload["registration_model"] == "homography"
        assert payload["stage_ms"] == {"diff_ms": 12.0}

    def test_non_finite_numbers_become_null(self):
        """JSON has no Infinity; a strict parser rejects the whole line without this."""
        payload = _parse(_emit(err=float("inf"), nan=float("nan"), ok=1.5))
        assert payload["err"] is None and payload["nan"] is None and payload["ok"] == 1.5
        json.dumps(payload, allow_nan=False)

    def test_an_exception_is_carried_as_a_field_not_a_second_line(self):
        record = logging.LogRecord("x", logging.ERROR, "p", 1, "boom", (), None)
        record.fields = {"scan_index": 0}
        try:
            raise ValueError("bad frame")
        except ValueError:
            record.exc_info = sys.exc_info()
        line = JsonLogFormatter().format(record)
        assert "\n" not in line, "a multi-line log record breaks every log explorer"
        assert "ValueError: bad frame" in _parse(line)["exception"]


class TestRedaction:
    @pytest.mark.parametrize("key", [
        "api_key", "ANTHROPIC_API_KEY", "authorization", "bearer_token",
        "aws_secret_access_key", "password",
    ])
    def test_a_secret_value_never_reaches_the_line(self, key):
        line = _emit(**{key: "sk-super-secret-value"})
        assert "sk-super-secret-value" not in line
        assert "[redacted]" in line

    def test_secrets_nested_in_a_dict_are_dropped_too(self):
        line = _emit(config={"model_api_key": "sk-nested", "region": "us-east-1"})
        assert "sk-nested" not in line
        assert _parse(line)["config"]["region"] == "us-east-1"

    def test_harmless_keys_are_untouched(self):
        payload = _parse(_emit(session_id="ses_1", observed_fraction=0.3))
        assert payload["session_id"] == "ses_1"
        assert payload["observed_fraction"] == 0.3


class TestScanFields:
    def test_the_summary_reduces_to_the_numbers_that_explain_a_duration(self):
        summary = {
            "feature_backend": "neural-lightglue",
            "match_backend": "classic-lowstructure",
            "events": [{}, {}, {}],
            "registration": {"model": "homography", "confidence": 0.92},
            "coverage": {"observed_fraction": 0.3},
            "timings_ms": {"total_ms": 14000.0, "match_ms": 5200.0, "diff_ms": 1900.0},
        }
        fields = scan_log_fields(summary)
        assert fields["total_ms"] == 14000.0
        assert fields["stage_ms"] == {"match_ms": 5200.0, "diff_ms": 1900.0}
        assert fields["match_backend"] == "classic-lowstructure"
        assert fields["event_count"] == 3
        assert "keypoint_counts" not in json.dumps(fields)

    def test_an_absent_summary_does_not_raise(self):
        assert scan_log_fields({})["total_ms"] is None


class TestConfigure:
    def test_it_is_idempotent_and_writes_json_to_stdout(self, monkeypatch):
        import service.observability as obs

        monkeypatch.setattr(obs, "_CONFIGURED", False)
        buffer = io.StringIO()
        monkeypatch.setattr(sys, "stdout", buffer)
        configure_logging(level="INFO", fmt="json")
        configure_logging(level="INFO", fmt="json")
        try:
            log_fields(logging.getLogger("strata.probe"), "hello", probe_value=7)
        finally:
            root = logging.getLogger()
            root.handlers = [
                h for h in root.handlers
                if getattr(h, "stream", None) is not buffer
            ]
        lines = [line for line in buffer.getvalue().splitlines() if line.strip()]
        assert lines, "nothing was logged"
        assert all(json.loads(line).get("message") for line in lines)
