"""Prove the AWS deployment actually ran, and keep the receipt.

``deploy.sh`` creates the infrastructure. This drives one real scan through it
and writes down what came back: the function's architecture and image digest,
the latency measured from outside, the change set the Arm path produced, the
ledger row that landed in DynamoDB, and the tail of the CloudWatch log.

It exists because "the deployment script is complete" and "the workload runs on
AWS" are different claims, and only the second one is worth submitting. Run it
after ``./deploy/deploy.sh``:

    python deploy/verify.py --region us-east-1 --record

Exits non-zero if any of it did not genuinely happen.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

try:
    import boto3
    from boto3.dynamodb.conditions import Key
except ImportError:  # pragma: no cover - deployment-time dependency
    sys.exit("verify.py needs boto3: pip install -r requirements-dev.txt")

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vision"))

from strata_core.synth import make_scene  # noqa: E402


def _jpeg(frame: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:  # pragma: no cover
        raise RuntimeError("could not encode a frame as JPEG")
    return buf.tobytes()


def _frames(specified: Path | None) -> dict[str, bytes]:
    """The frames to compare: two files the caller supplies, or a seeded scene."""
    if specified:
        names = ("baseline", "rescan")
        found = {n: specified / f"{n}.{ext}" for n in names for ext in ("jpg", "png")}
        picked = {}
        for name in names:
            path = next(
                (p for k, p in found.items() if k == name and p.is_file()), None
            )
            if path is None:
                raise SystemExit(f"--frames directory must contain {name}.jpg or {name}.png")
            picked[name] = path.read_bytes()
        return picked
    scene = make_scene(seed=1000, size=(960, 720))
    return {
        "baseline": _jpeg(scene.baseline),
        "rescan": _jpeg(scene.rescan),
    }


def _post(url: str, payload: dict, timeout: float) -> tuple[dict, float]:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, method="POST", headers={"Content-Type": "application/json"}
    )
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read()
    elapsed = (time.perf_counter() - started) * 1000.0
    return json.loads(raw), elapsed


def _log_tail(client, function_name: str, limit: int = 40) -> list[str]:
    """The most recent Lambda log lines - the observability half of the receipt."""
    group = f"/aws/lambda/{function_name}"
    try:
        streams = client.describe_log_streams(
            logGroupName=group, orderBy="LastEventTime", descending=True, limit=1
        )["logStreams"]
        if not streams:
            return []
        events = client.get_log_events(
            logGroupName=group,
            logStreamName=streams[0]["logStreamName"],
            startFromHead=False,
        )["events"]
        return [e["message"].rstrip() for e in events][-limit:]
    except Exception as exc:  # pragma: no cover - depends on log permissions
        return [f"could not read logs: {exc}"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--function", default="strata-vision")
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--table", default="strata-ledger")
    ap.add_argument("--site", default="verify-bay-4")
    ap.add_argument("--frames", type=Path, default=None,
                    help="directory containing baseline.jpg and rescan.jpg")
    ap.add_argument("--record", action="store_true",
                    help="also write the ledger row, and read it back")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--out", type=Path, default=Path("deploy/verify-report.json"))
    args = ap.parse_args(argv)

    session = boto3.Session(region_name=args.region)
    sts = session.client("sts")
    lam = session.client("lambda")
    s3 = session.client("s3")
    ddb = session.resource("dynamodb")
    account = sts.get_caller_identity()["Account"]

    report: dict = {
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "account": account,
        "region": args.region,
        "bucket": args.bucket,
        "table": args.table,
        "failures": [],
    }

    # 1. What is actually deployed. ------------------------------------------------
    fn = lam.get_function(FunctionName=args.function)
    cfg = fn["Configuration"]
    report["function"] = {
        "name": cfg["FunctionName"],
        "arn": cfg["Arn"],
        "architectures": cfg.get("Architectures"),
        "runtime": cfg.get("Runtime"),
        "package_type": cfg.get("PackageType"),
        "image_uri": cfg.get("Code", {}).get("ImageUri") or cfg.get("ImageConfigResponse", {}).get("RepositoryImageUri"),
        "memory_size_mb": cfg.get("MemorySize"),
        "timeout_s": cfg.get("Timeout"),
        "last_update_status": cfg.get("LastUpdateStatus"),
    }
    print(f"  function {cfg['FunctionName']} · {cfg.get('Architectures')} · {cfg.get('MemorySize')} MB")
    if not cfg.get("Architectures") or "arm64" not in cfg["Architectures"]:
        report["failures"].append("the deployed function is not arm64 (Graviton)")

    # 2. Put real images where the function can read them. -----------------------
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    keys = {}
    for name, blob in _frames(args.frames).items():
        key = f"verify/{stamp}/{name}.jpg"
        s3.put_object(Bucket=args.bucket, Key=key, Body=blob,
                      ServerSideEncryption="AES256")
        keys[name] = key
    report["objects"] = dict(keys)
    print(f"  uploaded s3://{args.bucket}/{keys['baseline']}")

    # 3. Invoke it, and time the whole round trip from outside. ------------------
    payload = {
        "site_id": args.site,
        "session_id": stamp,
        "baseline": [keys["baseline"]],
        "rescan": [keys["rescan"]],
        "record": bool(args.record),
    }
    url = None
    try:
        url = lam.get_function_url_config(FunctionName=args.function)["FunctionUrl"]
    except lam.exceptions.ResourceNotFoundException:
        url = None

    if url:
        result, latency_ms = _post(url, payload, args.timeout)
        report["invoked_via"] = "function_url"
        report["function_url"] = url
    else:  # pragma: no cover - depends on the deployment
        resp = lam.invoke(FunctionName=args.function, InvocationType="RequestResponse",
                          Payload=json.dumps(payload).encode())
        latency_ms = 0.0
        result = json.loads(resp["Payload"].read())
        report["invoked_via"] = "lambda_invoke"
        report["function_logs"] = base64_decode_logs(resp)

    report["latency_ms"] = round(latency_ms, 1)
    report["response"] = result
    print(f"  responded in {latency_ms:.0f} ms · backend={result.get('backend')} · ok={result.get('ok')}")

    if not result.get("ok"):
        report["failures"].append(f"the function reported an error: {result.get('error')}")
    else:
        reg = result.get("registration") or {}
        print(f"  registration {reg.get('model')} · {reg.get('inliers')} inliers "
              f"of {reg.get('matches')} · confidence {reg.get('confidence')}")
        print(f"  events: {len(result.get('events', []))} · "
              f"timings: {json.dumps(result.get('timings_ms', {}))}")
        if not result.get("events"):
            report["failures"].append("the scan produced no events for a scene with known changes")

    # 4. Read back what it wrote. ------------------------------------------------
    if args.record:
        pk = f"lambda/ledger/{args.site}"
        page = ddb.Table(args.table).query(
            KeyConditionExpression=Key("pk").eq(pk)
        )
        mine = [i for i in page.get("Items", []) if stamp in i.get("sk", "")]
        report["ledger_query"] = {"pk": pk, "rows": len(page.get("Items", [])),
                                  "rows_for_this_run": len(mine)}
        if mine:
            stored = json.loads(mine[0]["doc"])
            report["ledger_row"] = {"pk": mine[0]["pk"], "sk": mine[0]["sk"],
                                    "events": len(stored.get("events", []))}
            print(f"  ledger row read back: {mine[0]['pk']} / {mine[0]['sk']}")
        else:
            report["failures"].append("record=true was sent but no ledger row appeared")

    # 5. The log tail, so the receipt shows the runtime's own words. -------------
    report["cloudwatch_tail"] = _log_tail(session.client("logs"), args.function)

    report["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report["verdict"] = "PASS" if not report["failures"] else "FAIL"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(f"\n  {report['verdict']}  →  {args.out}")
    for failure in report["failures"]:
        print(f"    - {failure}")
    return 0 if report["verdict"] == "PASS" else 1


def base64_decode_logs(resp: dict) -> str:  # pragma: no cover - invoke path only
    import base64
    return base64.b64decode(resp.get("LogResult", "")).decode("utf-8", "replace")


if __name__ == "__main__":
    raise SystemExit(main())
