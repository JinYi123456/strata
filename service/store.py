"""Persistence for sessions, frames and the ledger.

One interface, two implementations:

* :class:`LocalStore` writes under a directory on disk. It is the default and it
  is what the live demo runs on, so the service is fully functional with no
  cloud account at all.
* :class:`S3Store` writes frames to S3 and records to DynamoDB. It exists so the
  same code runs on managed infrastructure without a second code path in the
  handlers - the deployment target is a configuration change, not a rewrite.

Keys are validated rather than trusted. They arrive from URLs and from session
ids, and a key containing ``..`` would otherwise let a request read or write
outside the storage root.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path
from typing import Any, Protocol

log = logging.getLogger("strata.store")

_SAFE_KEY = re.compile(r"^[A-Za-z0-9._/-]+$")


class StoreError(RuntimeError):
    """A storage operation could not be completed."""


def validate_key(key: str) -> str:
    """Reject keys that could escape the storage root or collide with metadata."""
    if not key or key.startswith("/") or key.endswith("/"):
        raise StoreError(f"invalid key: {key!r}")
    if not _SAFE_KEY.match(key):
        raise StoreError(f"key contains unsupported characters: {key!r}")
    if ".." in key.split("/") or "." in key.split("/"):
        raise StoreError(f"key must not contain relative segments: {key!r}")
    return key


class Store(Protocol):
    """What the handlers need from persistence, and nothing more."""

    name: str

    def put_bytes(self, key: str, data: bytes) -> None: ...

    def get_bytes(self, key: str) -> bytes: ...

    def exists(self, key: str) -> bool: ...

    def put_json(self, key: str, obj: Any) -> None: ...

    def get_json(self, key: str) -> Any | None: ...

    def list_json(self, prefix: str) -> list[Any]: ...


class LocalStore:
    """Filesystem-backed storage.

    Writes go to a temporary file and are then renamed, so a reader never sees a
    half-written frame. That matters because the web client fetches images while
    a scan is still producing them.
    """

    name = "local"

    def __init__(self, root: str | Path) -> None:
        # Resolved once, here. A relative root would otherwise be compared against
        # the absolute paths that come back from ``_path``, and the comparison
        # raises rather than failing quietly - so the default configuration, which
        # uses the relative path "data", would break every directory listing.
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, key: str) -> Path:
        p = (self.root / validate_key(key)).resolve()
        # Belt and braces: resolve() can still land outside root via a symlink.
        if not str(p).startswith(str(self.root)):
            raise StoreError(f"key escapes the storage root: {key!r}")
        return p

    def put_bytes(self, key: str, data: bytes) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".part")
        with self._lock:
            tmp.write_bytes(data)
            tmp.replace(p)

    def get_bytes(self, key: str) -> bytes:
        p = self._path(key)
        if not p.is_file():
            raise StoreError(f"no such object: {key!r}")
        return p.read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def put_json(self, key: str, obj: Any) -> None:
        self.put_bytes(key, json.dumps(obj, indent=2, default=str).encode("utf-8"))

    def get_json(self, key: str) -> Any | None:
        if not self.exists(key):
            return None
        try:
            return json.loads(self.get_bytes(key).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            # A corrupt record must not take the service down; the caller treats
            # a missing record and an unreadable one the same way.
            log.warning("unreadable record %s: %s", key, exc)
            return None

    def list_json(self, prefix: str) -> list[Any]:
        base = self._path(prefix.rstrip("/"))
        if not base.is_dir():
            return []
        out = []
        for p in sorted(base.glob("*.json")):
            obj = self.get_json(str(p.relative_to(self.root)).replace("\\", "/"))
            if obj is not None:
                out.append(obj)
        return out


class S3Store:
    """S3 for frames, DynamoDB for records.

    boto3 is imported lazily so that a local deployment never needs it installed
    and never pays for its import at start-up.

    The DynamoDB item is ``{pk, sk, doc}`` where ``pk`` is the parent of the
    record key and ``sk`` its leaf, which is the same shape as the filesystem:
    ``list_json(prefix)`` returns the records directly under ``prefix`` and no
    others. ``deploy/deploy.sh`` creates the table with exactly this schema, and
    a test asserts the two stay in step - a record written against a key schema
    the table does not have fails on the first ``put_item`` in production, not
    here.
    """

    name = "s3"

    def __init__(
        self,
        bucket: str,
        table: str,
        *,
        region: str | None = None,
        prefix: str = "strata",
    ) -> None:
        try:
            import boto3  # noqa: PLC0415 - deliberate lazy import
        except ImportError as exc:  # pragma: no cover - depends on deployment
            raise StoreError(
                "the s3 store requires boto3; install it with "
                "'pip install -r requirements-dev.txt'"
            ) from exc
        self._boto3 = boto3
        self.bucket = bucket
        self.table = table
        self.prefix = prefix.strip("/")
        self._s3 = boto3.client("s3", region_name=region)
        self._ddb = boto3.resource("dynamodb", region_name=region)

    def _object_key(self, key: str) -> str:
        return f"{self.prefix}/{validate_key(key)}" if self.prefix else validate_key(key)

    def _record_key(self, key: str) -> tuple[str, str]:
        """Split a record key into the partition key and sort key it lives under."""
        full = self._object_key(key)
        parent, _, leaf = full.rpartition("/")
        if not parent or not leaf:
            raise StoreError(f"record keys must have a parent prefix: {key!r}")
        return parent, leaf

    def put_bytes(self, key: str, data: bytes) -> None:
        self._s3.put_object(Bucket=self.bucket, Key=self._object_key(key), Body=data)

    def get_bytes(self, key: str) -> bytes:
        try:
            return self._s3.get_object(
                Bucket=self.bucket, Key=self._object_key(key)
            )["Body"].read()
        except Exception as exc:  # pragma: no cover - network path
            raise StoreError(f"could not read {key}: {exc}") from exc

    def exists(self, key: str) -> bool:
        try:
            self._s3.head_object(Bucket=self.bucket, Key=self._object_key(key))
            return True
        except Exception:  # pragma: no cover - network path
            return False

    def put_json(self, key: str, obj: Any) -> None:
        pk, sk = self._record_key(key)
        self._ddb.Table(self.table).put_item(
            Item={"pk": pk, "sk": sk, "doc": json.dumps(obj, default=str)}
        )

    def get_json(self, key: str) -> Any | None:
        pk, sk = self._record_key(key)
        item = (
            self._ddb.Table(self.table)
            .get_item(Key={"pk": pk, "sk": sk})
            .get("Item")
        )
        if item is None:
            return None
        try:
            return json.loads(item["doc"])
        except (json.JSONDecodeError, KeyError) as exc:  # pragma: no cover
            log.warning("unreadable record %s: %s", key, exc)
            return None

    def list_json(self, prefix: str) -> list[Any]:
        from boto3.dynamodb.conditions import Key  # noqa: PLC0415

        table = self._ddb.Table(self.table)
        parent = self._object_key(prefix.rstrip("/"))
        query_kwargs: dict[str, Any] = {
            "KeyConditionExpression": Key("pk").eq(parent)
        }
        out: list[Any] = []
        while True:
            resp = table.query(**query_kwargs)
            for item in resp.get("Items", []):
                try:
                    out.append(json.loads(item["doc"]))
                except (json.JSONDecodeError, KeyError):  # pragma: no cover
                    log.warning("unreadable record under %s", parent)
            if "LastEvaluatedKey" not in resp:
                break
            query_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
        return out


def build_store(executor: str, *, root: str | Path = "data", **aws: Any) -> Store:
    """Pick a store from the configured executor."""
    if executor == "aws":
        return S3Store(
            bucket=aws["bucket"],
            table=aws["table"],
            region=aws.get("region"),
            prefix=aws.get("prefix", "strata"),
        )
    return LocalStore(root)
