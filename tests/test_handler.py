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

from strata_core.synth import make_scene  # noqa: E402


@pytest.fixture()
def mod(monkeypatch):
    """The handler module, with S3 and DynamoDB swapped for in-memory fakes."""
    handler = importlib.import_module("handler")
    monkeypatch.setenv("STRATA_S3_BUCKET", "test-bucket")
    monkeypatch.delenv("STRATA_DDB_TABLE", raising=False)
    return handler


class FakeTable:
    def __init__(self) -> None:
        self.items: list[dict] = []

    def put_item(self, Item):  # noqa: N803 - boto3 spells it this way
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
    assert row["site_id"] == "bay-4"
    assert row["session_id"] == "s-1"
    assert row["event_count"] == len(result["events"])
    # The ledger stores the payload, so a row is readable on its own.
    assert json.loads(row["payload"])["ok"] is True
