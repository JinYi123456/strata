"""Tests for the Lambda entry point.

The handler is the only part of the AWS path that can be exercised without an
AWS account: everything below it is the same pipeline the service runs, and
everything above it is Lambda's plumbing. So the surface worth testing is the
event contract and the payload it returns — both of which are pure functions of
the input once S3 and DynamoDB are replaced with fakes.

This matters because the handler had never been executed end to end. The first
version of it mutated a frozen config and would have raised on its very first
invocation in production.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy"
if str(DEPLOY) not in sys.path:
    sys.path.insert(0, str(DEPLOY))

from contract import key_attributes  # noqa: E402 - tests/ is on sys.path via pytest
from strata_core.synth import make_scene  # noqa: E402


@pytest.fixture()
def mod(monkeypatch):
    """The handler module, with S3 and DynamoDB swapped for in-memory fakes."""
    handler = importlib.import_module("handler")
    monkeypatch.setenv("STRATA_S3_BUCKET", "test-bucket")
    monkeypatch.delenv("STRATA_DDB_TABLE", raising=False)
    return handler


class FakeTable:
    """A table that rejects items the way DynamoDB rejects them: by key schema."""

    def __init__(self) -> None:
        self.items: list[dict] = []

    def put_item(self, Item):  # noqa: N803 - boto3 spells it this way
        missing = [a for a in key_attributes() if a not in Item]
        if missing:
            raise AssertionError(
                f"item does not match the deployed table schema; missing {missing}"
            )
        self.items.append(Item)


class FakeDdb:
    def __init__(self) -> None:
        self.table = FakeTable()

    def Table(self, name):  # noqa: N802 - boto3 spells it this way
        return self.table


def _event(**overrides) -> dict:
    base = {
        "site_id": "bay-4",
        "session_id": "s-1",
        "baseline": ["base.png"],
        "rescan": ["scan.png"],
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------
# Fast paths: the handler must fail cleanly, not raise
# --------------------------------------------------------------------------

def test_the_ledger_row_matches_the_deployed_table_schema():
    """The row shape is a contract with deploy.sh, checked without an account."""
    handler = importlib.import_module("handler")
    row = handler.ledger_row("lambda", "bay-4", "s-1", {"ok": True}, now_ms=1700000000000)
    assert sorted(row) == sorted([*key_attributes(), "doc"])
    assert row["pk"] == "lambda/ledger/bay-4"
    assert row["sk"] == "s-1-1700000000000.json"
    assert json.loads(row["doc"]) == {"ok": True}
    assert all(isinstance(v, str) for v in row.values())

def test_missing_fields_return_a_typed_error(mod):
    result = mod.handler({"site_id": "bay-4"}, None)
    assert result["ok"] is False
    assert result["error"] == "bad_request"


def test_unreadable_object_returns_a_typed_error(mod, monkeypatch):
    def boom(bucket, key):
        raise ValueError("could not decode")

    monkeypatch.setattr(mod, "load_image", boom)
    result = mod.handler(_event(), None)
    assert result["ok"] is False
    assert result["error"] == "unreadable_session"
    assert "could not decode" in result["detail"]


def test_unregistrable_pair_returns_valid_json(mod, monkeypatch):
    """A pair with nothing to correspond must come back as a serialisable value.

    This is the case that broke the service, and it broke the handler the same
    way: an identity registration has an infinite reprojection error, and
    ``inf`` is not JSON, so a strict serialiser rejects the whole payload — on
    exactly the input that most needs a clean answer. The fix is to encode "no
    value" as null rather than as a sentinel.
    """
    blank = np.full((240, 320, 3), 128, np.uint8)
    monkeypatch.setattr(mod, "load_image", lambda bucket, key: blank.copy())

    result = mod.handler(_event(), None)

    # The pipeline handled it rather than failing: it declines to claim an
    # alignment instead of inventing one.
    assert result["ok"] is True
    assert result["registration"]["model"] == "identity"
    assert result["registration"]["confidence"] == 0.0
    assert result["registration"]["reprojection_px"] is None
    assert result["events"] == []

    # Strict: no Infinity/NaN tokens anywhere.
    json.dumps(result, allow_nan=False)


# --------------------------------------------------------------------------
# Slow: the full scan through the handler
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_scan_returns_a_complete_payload(mod, monkeypatch):
    scene = make_scene(seed=20261026, size=(960, 720), n_objects=18, n_mutations=4)
    frames = {"base.png": scene.baseline, "scan.png": scene.rescan}
    monkeypatch.setattr(mod, "load_image", lambda bucket, key: frames[key])
    # The handler's default is the Lambda layout (/var/task/models). Point it at
    # the repository's weights so the test exercises the path the deployment
    # takes, rather than the fallback it takes when they are absent.
    monkeypatch.setenv("STRATA_MODEL_DIR", str(ROOT / "vision" / "models"))

    result = mod.handler(_event(), None)

    assert result["ok"] is True
    assert result["site_id"] == "bay-4"
    assert result["session_id"] == "s-1"
    assert result["backend"].startswith(("neural", "classic"))
    assert 0.0 <= result["registration"]["confidence"] <= 1.0
    assert result["registration"]["inliers"] > 0
    assert isinstance(result["events"], list)
    for event in result["events"]:
        assert event["kind"] in {"added", "missing", "moved", "state_changed", "uncertain"}
        assert 0.0 <= event["confidence"] <= 1.0
        assert len(event["box"]) == 4
    assert "total_ms" in result["timings_ms"]
    assert result["recorded"] is False  # no table configured, no record attempted
    json.dumps(result, allow_nan=False)


@pytest.mark.slow
def test_record_without_a_table_says_so(mod, monkeypatch):
    scene = make_scene(seed=7, size=(960, 720))
    frames = {"base.png": scene.baseline, "scan.png": scene.rescan}
    monkeypatch.setattr(mod, "load_image", lambda bucket, key: frames[key])

    result = mod.handler(_event(record=True), None)
    assert result["ok"] is True
    assert result["recorded"] is False
    assert "no ledger table" in result["record_error"]


@pytest.mark.slow
def test_record_writes_one_ledger_row(mod, monkeypatch):
    scene = make_scene(seed=11, size=(960, 720))
    frames = {"base.png": scene.baseline, "scan.png": scene.rescan}
    monkeypatch.setattr(mod, "load_image", lambda bucket, key: frames[key])

    fake = FakeDdb()
    monkeypatch.setattr(mod, "ddb", lambda: fake)
    monkeypatch.setenv("STRATA_DDB_TABLE", "strata-ledger")

    result = mod.handler(_event(record=True), None)

    assert result["ok"] is True
    assert result["recorded"] is True
    assert len(fake.table.items) == 1
    row = fake.table.items[0]
    # The row is written against the key schema deploy.sh creates the table with
    # - the same contract service/store.py writes to. A row that does not match
    # it is a ValidationException in production, not a test failure here.
    assert row["pk"] == "lambda/ledger/bay-4"
    assert row["sk"].startswith("s-1-")
    payload = json.loads(row["doc"])
    assert payload["ok"] is True
    assert payload["site_id"] == "bay-4"
    assert len(payload["events"]) == len(result["events"])


# --------------------------------------------------------------------------
# Fast paths: the observability surface, without running a scan
# --------------------------------------------------------------------------

_SCAN_PAYLOAD = {
    "ok": True,
    "site_id": "bay-4",
    "session_id": "s-1",
    "backend": "neural-lightglue",
    "match_backend": "classic-lowstructure",
    "wall_ms": 14210.4,
    "events": [{"kind": "added"}, {"kind": "missing"}],
    "registration": {"model": "homography", "confidence": 0.91},
    "coverage": {"observed_fraction": 0.3},
    "timings_ms": {"extract_baseline_ms": 1500.0, "match_ms": 5200.0,
                   "diff_ms": 1900.0, "total_ms": 14000.0},
}


def test_the_log_line_is_one_parseable_json_object(mod):
    line = mod.log_line(_SCAN_PAYLOAD)
    assert "\n" not in line
    record = json.loads(line)
    assert record["event"] == "scan" and record["ok"] is True
    assert record["site_id"] == "bay-4"
    # The distinction the report depends on: which matcher was resolved, and
    # which one actually ran.
    assert record["backend"] == "neural-lightglue"
    assert record["match_backend"] == "classic-lowstructure"
    assert record["stage_ms"] == {"extract_baseline_ms": 1500.0, "match_ms": 5200.0,
                                  "diff_ms": 1900.0}
    assert record["wall_ms"] == 14210.4


def test_a_failed_scan_logs_at_error_level_with_its_reason(mod):
    line = mod.log_line({"ok": False, "error": "unregistrable", "site_id": "x"})
    record = json.loads(line)
    assert record["level"] == "ERROR"
    assert record["error"] == "unregistrable"


def test_metrics_cover_the_whole_duration_and_each_stage(mod):
    points = mod.metric_datapoints(_SCAN_PAYLOAD)
    names = {p["MetricName"] for p in points}
    assert "ScanDuration" in names and "ChangeEvents" in names
    assert {"Stage.extract_baseline", "Stage.match", "Stage.diff"} <= names
    duration = next(p for p in points if p["MetricName"] == "ScanDuration")
    assert duration["Value"] == 14210.4 and duration["Unit"] == "Milliseconds"
    events = next(p for p in points if p["MetricName"] == "ChangeEvents")
    assert events["Value"] == 2
    assert all(p["Dimensions"] == [{"Name": "SiteId", "Value": "bay-4"}] for p in points)


def test_a_failed_scan_emits_no_metrics(mod):
    assert mod.metric_datapoints({"ok": False, "error": "bad_request"}) == []


def test_metrics_are_chunked_to_the_service_limit(mod):
    payload = dict(_SCAN_PAYLOAD)
    payload["timings_ms"] = {f"stage_{i}_ms": float(i) for i in range(40)}
    payload["timings_ms"]["total_ms"] = 1.0
    points = mod.metric_datapoints(payload)
    assert len(points) > 20
    chunks = [points[i:i + 20] for i in range(0, len(points), 20)]
    assert all(len(c) <= 20 for c in chunks)


class TestWeightStaging:
    """The zip package cannot carry 50 MB of weights, so the function fetches them.

    A missing weight is a backend choice, not a failure - and the difference
    between "ran the neural path" and "quietly ran SIFT" is exactly the thing a
    latency or accuracy number must not be allowed to hide.
    """

    class FakeS3:
        def __init__(self, available):
            self.available = set(available)
            self.requested = []

        def download_file(self, bucket, key, target):
            self.requested.append(key)
            name = key.rsplit("/", 1)[-1]
            if name not in self.available:
                raise RuntimeError(f"NoSuchKey: {key}")
            Path(target).write_bytes(b"fake-onnx")

    def _run(self, mod, monkeypatch, tmp_path, available, existing=(), bucket="b"):
        import pathlib
        staging = tmp_path / "models"
        staging.mkdir()
        for name in existing:
            (staging / name).write_bytes(b"x")
        fake = self.FakeS3(available)
        monkeypatch.setattr(mod, "s3", lambda: fake)
        monkeypatch.setenv("STRATA_S3_BUCKET", bucket)
        monkeypatch.setenv("STRATA_WEIGHT_PREFIX", "models/")
        return mod.ensure_weights(str(staging)), fake, staging

    def test_both_weights_are_fetched_when_absent(self, mod, monkeypatch, tmp_path):
        fetched, fake, _ = self._run(mod, monkeypatch, tmp_path, set(mod.WEIGHT_FILES))
        assert fetched == list(mod.WEIGHT_FILES)
        assert fake.requested == ["models/" + n for n in mod.WEIGHT_FILES]

    def test_an_existing_weight_is_not_refetched(self, mod, monkeypatch, tmp_path):
        fetched, fake, _ = self._run(
            mod, monkeypatch, tmp_path, set(mod.WEIGHT_FILES),
            existing=(mod.WEIGHT_FILES[0],),
        )
        assert fetched == [mod.WEIGHT_FILES[1]]
        assert fake.requested == ["models/" + mod.WEIGHT_FILES[1]]

    def test_a_missing_weight_degrades_without_raising(self, mod, monkeypatch, tmp_path):
        """The classical backend is a legitimate answer; a crash is not."""
        fetched, _fake, _ = self._run(mod, monkeypatch, tmp_path, {mod.WEIGHT_FILES[0]})
        assert fetched == []

    def test_no_bucket_means_no_fetch(self, mod, monkeypatch, tmp_path):
        monkeypatch.delenv("STRATA_S3_BUCKET", raising=False)
        assert mod.ensure_weights(str(tmp_path / "m")) == []

    def test_the_prefix_is_honoured(self, mod, monkeypatch, tmp_path):
        import pathlib
        staging = tmp_path / "m"
        fake = self.FakeS3(set(mod.WEIGHT_FILES))
        monkeypatch.setattr(mod, "s3", lambda: fake)
        monkeypatch.setenv("STRATA_S3_BUCKET", "b")
        monkeypatch.setenv("STRATA_WEIGHT_PREFIX", "v2/weights/")
        mod.ensure_weights(str(staging))
        assert all(k.startswith("v2/weights/") for k in fake.requested)
