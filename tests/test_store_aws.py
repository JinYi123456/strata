"""The managed storage path, exercised without an AWS account.

``S3Store`` had no test at all, and the schema it wrote against was not the
schema ``deploy/deploy.sh`` creates: the store wrote ``{pk, doc}`` items into a
table whose key schema is ``site_id`` + ``session_id``, so the first
``put_item`` in a real deployment would have been a ValidationException. A
documented-but-never-run cloud path is exactly the kind of defect that survives
review, so this module pins the three things that would have caught it:

* the table schema declared in ``deploy.sh`` is the schema the store writes;
* every DynamoDB action the store issues is granted by the IAM policy in
  ``deploy.sh``;
* the two store implementations agree, key for key, on what they store and
  retrieve - which is what "switching is a configuration change" claims.

No network, no credentials, no moto: the fake is small and it validates the way
DynamoDB validates, by rejecting items that do not match the key schema.
"""

from __future__ import annotations

import json
import re
import sys
import types
from pathlib import Path

import pytest

from contract import (
    declared_attribute_types,
    declared_key_schema,
    deployed_requirements,
    granted_actions,
)
from service.store import LocalStore, S3Store, StoreError

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------- fake boto3 ---

class ValidationException(Exception):
    """What DynamoDB raises when an item or key does not match the table."""


class FakeCondition:
    def __init__(self, attr: str) -> None:
        self.attr = attr
        self.op = None
        self.value = None

    def eq(self, value):
        clone = FakeCondition(self.attr)
        clone.op, clone.value = "eq", value
        return clone


class FakeTable:
    """A table that enforces its key schema the way the real service does."""

    def __init__(self, name: str, schema: list[tuple[str, str]]) -> None:
        self.name = name
        self.schema = schema
        self.items: dict[tuple[str, str], dict] = {}
        self.calls: list[str] = []

    def _validate(self, keys: dict, *, partial: bool = False) -> tuple[str, str]:
        required = [attr for attr, _ in self.schema]
        missing = [attr for attr in required if attr not in keys]
        if missing:
            raise ValidationException(
                f"Provided list of item keys contains {sorted(keys)}; "
                f"table '{self.name}' has key schema {required}. Missing: {missing}"
            )
        return tuple(str(keys[attr]) for attr in required)  # type: ignore[return-value]

    def put_item(self, Item: dict) -> dict:
        self.calls.append("PutItem")
        pk_sk = self._validate(Item)
        self.items[pk_sk] = dict(Item)
        return {}

    def get_item(self, Key: dict) -> dict:
        self.calls.append("GetItem")
        return {"Item": self.items.get(self._validate(Key))}

    def query(self, KeyConditionExpression: FakeCondition, **_) -> dict:
        self.calls.append("Query")
        assert KeyConditionExpression.op == "eq"
        parent = KeyConditionExpression.value
        items = [
            item for (pk, _sk), item in self.items.items() if pk == parent
        ]
        return {"Items": items}


class FakeDynamoDB:
    def __init__(self, schema: list[tuple[str, str]]) -> None:
        self.schema = schema
        self.tables: dict[str, FakeTable] = {}

    def Table(self, name: str) -> FakeTable:
        return self.tables.setdefault(name, FakeTable(name, self.schema))


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def put_object(self, Bucket: str, Key: str, Body: bytes) -> None:
        self.objects[f"{Bucket}/{Key}"] = bytes(Body)

    def get_object(self, Bucket: str, Key: str) -> dict:
        return {"Body": types.SimpleNamespace(
            read=lambda: self.objects[f"{Bucket}/{Key}"]
        )}

    def head_object(self, Bucket: str, Key: str) -> dict:
        if f"{Bucket}/{Key}" not in self.objects:
            raise KeyError(f"{Bucket}/{Key}")
        return {}


@pytest.fixture()
def aws(monkeypatch):
    """Install a fake boto3 and hand back the in-memory services behind it."""
    schema = declared_key_schema()
    assert schema, "deploy.sh declares no DynamoDB key schema"

    dynamo = FakeDynamoDB(schema)
    s3 = FakeS3()

    fake = types.ModuleType("boto3")
    fake.client = lambda service, region_name=None: s3  # type: ignore[attr-defined]
    fake.resource = lambda service, region_name=None: dynamo  # type: ignore[attr-defined]
    conditions = types.ModuleType("boto3.dynamodb.conditions")
    conditions.Key = FakeCondition  # type: ignore[attr-defined]
    dynamodb_mod = types.ModuleType("boto3.dynamodb")
    dynamodb_mod.conditions = conditions  # type: ignore[attr-defined]
    fake.dynamodb = dynamodb_mod  # type: ignore[attr-defined]

    for name, module in (
        ("boto3", fake),
        ("boto3.dynamodb", dynamodb_mod),
        ("boto3.dynamodb.conditions", conditions),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return types.SimpleNamespace(s3=s3, dynamo=dynamo, table=lambda n: dynamo.Table(n))


def store(**kw) -> S3Store:
    return S3Store("strata-bucket", "strata-ledger", region="us-east-1", **kw)


# ------------------------------------------------------------- the schema ---

class TestSchemaAgreement:
    def test_deploy_script_declares_the_key_schema_the_store_writes(self):
        """The bug this test exists to keep dead: two schemas, one table."""
        assert declared_key_schema() == [("pk", "HASH"), ("sk", "RANGE")]

    def test_attribute_types_cover_the_key_schema(self):
        types_declared = declared_attribute_types()
        for attr, _role in declared_key_schema():
            assert types_declared.get(attr) == "S", f"{attr} is not declared as S"

    def test_iam_policy_grants_every_action_the_store_issues(self, aws):
        s = store()
        s.put_json("ledger/bay-4/a.json", {"x": 1})
        s.get_json("ledger/bay-4/a.json")
        s.list_json("ledger/bay-4")
        issued = set(aws.table("strata-ledger").calls)
        assert issued == {"PutItem", "GetItem", "Query"}
        allowed = granted_actions("dynamodb")
        missing = {a.lower() for a in issued} - {a.lower() for a in allowed}
        assert not missing, f"deploy.sh does not grant {missing}"

    def test_an_item_written_against_the_wrong_schema_is_rejected(self):
        """Proves the fake validates like DynamoDB, so the test above means something."""
        wrong = FakeTable("t", [("site_id", "HASH"), ("session_id", "RANGE")])
        with pytest.raises(ValidationException):
            wrong.put_item(Item={"pk": "ledger/bay-4", "sk": "a.json", "doc": "{}"})


# ------------------------------------------------------- behaviour parity ---

RECORDS = [
    ("sessions/ses_1/state.json", {"session_id": "ses_1", "scans": 1}),
    ("sessions/ses_1/scans/0.json", {"scan_index": 0, "outcome": "declined"}),
    ("sessions/ses_1/scans/1.json", {"scan_index": 1, "outcome": "finalized"}),
    ("ledger/bay-4/led_a.json", {"entry_id": "led_a", "approved": True}),
    ("ledger/bay-4/led_b.json", {"entry_id": "led_b", "approved": False}),
    ("ledger/aisle-2/led_c.json", {"entry_id": "led_c", "approved": True}),
]


class TestS3StoreBehaviour:
    def test_records_round_trip(self, aws):
        s = store()
        for key, doc in RECORDS:
            s.put_json(key, doc)
        for key, doc in RECORDS:
            assert s.get_json(key) == doc

    def test_missing_records_read_as_none(self, aws):
        assert store().get_json("ledger/bay-4/nope.json") is None

    def test_list_returns_records_directly_under_the_prefix(self, aws):
        s = store()
        for key, doc in RECORDS:
            s.put_json(key, doc)
        ids = sorted(e["entry_id"] for e in s.list_json("ledger/bay-4"))
        assert ids == ["led_a", "led_b"]
        assert [e["entry_id"] for e in s.list_json("ledger/aisle-2")] == ["led_c"]
        assert s.list_json("ledger/empty") == []

    def test_frames_are_stored_under_the_configured_prefix(self, aws):
        s = store(prefix="prod")
        s.put_bytes("sessions/ses_1/baseline/0.png", b"\x89PNG")
        assert list(aws.s3.objects) == ["strata-bucket/prod/sessions/ses_1/baseline/0.png"]
        assert s.get_bytes("sessions/ses_1/baseline/0.png") == b"\x89PNG"
        assert s.exists("sessions/ses_1/baseline/0.png")
        assert not s.exists("sessions/ses_1/baseline/9.png")

    def test_a_bare_key_still_lands_under_the_store_prefix(self, aws):
        """The partition key is never empty: the store prefix is the fallback parent."""
        s = store()
        s.put_json("state.json", {"a": 1})
        pk, sk = aws.table("strata-ledger")._validate(
            {"pk": "strata", "sk": "state.json"}
        )
        assert (pk, sk) == ("strata", "state.json")
        assert s.get_json("state.json") == {"a": 1}

    def test_a_record_key_never_resolves_outside_the_prefix(self, aws):
        s = store(prefix="prod")
        s.put_json("state.json", {"a": 1})
        assert list(aws.table("strata-ledger").items) == [("prod", "state.json")]

    def test_keys_cannot_escape_the_prefix(self, aws):
        with pytest.raises(StoreError):
            store().put_json("../../etc/passwd", {"a": 1})


class TestTheTwoStoresAgree:
    """The claim under test: switching backends is a configuration change.

    Both implementations are driven with the same keys and the same records and
    must answer identically. A divergence here is a bug in whichever path is not
    the default - the one that is exercised least and therefore rots first.
    """

    @pytest.fixture()
    def both(self, aws, tmp_path):
        return store(), LocalStore(tmp_path / "data")

    def test_identical_reads_and_writes(self, both):
        s3s, local = both
        for key, doc in RECORDS:
            s3s.put_json(key, doc)
            local.put_json(key, doc)
        for key, doc in RECORDS:
            assert s3s.get_json(key) == local.get_json(key) == doc
        for prefix in ("ledger/bay-4", "ledger/aisle-2", "sessions/ses_1", "ledger/none"):
            assert sorted(json.dumps(e, sort_keys=True) for e in s3s.list_json(prefix)) == \
                sorted(json.dumps(e, sort_keys=True) for e in local.list_json(prefix))

    def test_identical_bytes_semantics(self, both):
        s3s, local = both
        key = "sessions/ses_1/baseline/0.png"
        s3s.put_bytes(key, b"abc")
        local.put_bytes(key, b"abc")
        assert s3s.get_bytes(key) == local.get_bytes(key) == b"abc"
        assert s3s.exists(key) == local.exists(key)
        assert s3s.exists("sessions/ses_1/baseline/1.png") is False
        assert local.exists("sessions/ses_1/baseline/1.png") is False


# --------------------------------------------------- package version parity --

def test_the_lambda_carries_the_headless_opencv_at_the_measured_version():
    """Same build as the measurements, without the X11 the runtime does not have.

    ``opencv-contrib-python`` resolves cleanly for ``manylinux_2_28_aarch64`` and
    then fails at *import* on Lambda with "libxcb.so.1: cannot open shared object
    file" - the GUI wheel links the X client libraries and the runtime ships
    without them. That was found by deploying, not by reading, and the cost of
    getting it wrong again is a function that answers nothing.

    The flavour may differ from the local environment; the version may not, or the
    numbers this project publishes stop describing what runs on Arm.
    """
    pins = deployed_requirements()
    opencv = {name: version for name, version in pins.items()
              if name.startswith("opencv")}
    assert list(opencv) == ["opencv-contrib-python-headless"], (
        f"the deployed OpenCV is {opencv}; the Lambda needs the headless wheel "
        "and nothing else"
    )
    lock = (ROOT / "requirements.lock").read_text(encoding="utf-8")
    measured = re.search(r"^opencv-contrib-python==(\S+)$", lock, re.M)
    assert measured, "requirements.lock no longer pins the OpenCV the project was measured on"
    assert opencv["opencv-contrib-python-headless"] == measured.group(1), (
        f"Lambda runs OpenCV {opencv['opencv-contrib-python-headless']} but every "
        f"published figure came from {measured.group(1)}"
    )


def test_the_lambda_carries_the_measured_numpy():
    pins = deployed_requirements()
    lock = (ROOT / "requirements.lock").read_text(encoding="utf-8")
    measured = re.search(r"^numpy==(\S+)$", lock, re.M)
    assert measured and pins.get("numpy") == measured.group(1)


def test_the_deployed_package_ships_the_service_it_serves():
    """The console's API is the same ``service`` package, so it has to be in the zip.

    ``web_handler`` imports it, and the first deploy failed with
    "No module named 'strata_agent'" for the same reason: the package copied the
    vision core and nothing else, so the zip built fine and then refused to start.
    """
    text = (ROOT / "deploy" / "deploy.py").read_text(encoding="utf-8")
    for needed in ('"strata_core"', '"strata_agent"', 'ROOT / "service"',
                   'deploy" / "web_handler.py'):
        assert needed in text, f"build_package no longer stages {needed}"
