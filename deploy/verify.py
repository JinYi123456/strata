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
import base64
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


def _data_url(blob: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(blob).decode("ascii")


def public_api_roundtrip(endpoint: str, frames: dict[str, bytes], name: str,
                         timeout: float) -> dict:
    """Drive the console's own API over the public URL, from outside.

    ``/api/health`` answers the question "is something listening", and this
    answers the one a judge actually asks: open the console, upload two photos of
    a shelf, get a typed change set back - through the browser's CORS path, with
    the state landing in S3 and DynamoDB rather than on a laptop filesystem.
    """
    out: dict = {"base": endpoint}
    with urllib.request.urlopen(f"{endpoint}/api/health", timeout=timeout) as resp:
        doc = json.loads(resp.read())
    out["health"] = doc

    session, created_ms = _post(
        f"{endpoint}/api/sessions",
        {"name": name, "baseline_frames": [_data_url(frames["baseline"])]},
        timeout,
    )
    out["session_id"] = session.get("session_id")
    out["create_ms"] = round(created_ms, 1)

    rescan, rescan_ms = _post(
        f"{endpoint}/api/sessions/{session['session_id']}/rescan",
        {"frames": [_data_url(frames["rescan"])]},
        timeout,
    )
    out["rescan_ms"] = round(rescan_ms, 1)
    out["events"] = len(rescan.get("events", []))
    out["kinds"] = sorted({e["kind"] for e in rescan.get("events", [])})
    out["backend"] = rescan.get("backend") or (rescan.get("trace") or {}).get("backend")
    out["registration"] = rescan.get("registration")
    return out


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


def _deployment_endpoint() -> str | None:
    """The public API base recorded by the last deploy, if there was one."""
    path = ROOT / "build" / "lambda" / "deployment.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))["web_function"]["endpoint"]
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--function", default="strata-vision")
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--table", default="strata-ledger")
    ap.add_argument("--site", default="verify-bay-4")
    ap.add_argument("--api", default=None,
                    help="public API base; defaults to build/lambda/deployment.json")
    ap.add_argument("--no-api", action="store_true",
                    help="skip the public console-API round trip")
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
        "arn": cfg["FunctionArn"],
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
    frames = _frames(args.frames)
    keys = {}
    for name, blob in frames.items():
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

    result: dict | None = None
    latency_ms = 0.0
    if url:
        report["function_url"] = url
        try:
            result, latency_ms = _post(url, payload, args.timeout)
            report["invoked_via"] = "function_url"
        except Exception as exc:
            # An account can refuse anonymous Function URL invocation even with
            # the canonical "Principal": "*" policy - which is what happened here,
            # on this function and on a hello-world built to test for it. That is
            # an account setting, not a broken deployment, so say which of the two
            # the receipt is recording and carry on through the control plane.
            body = getattr(exc, "read", lambda: b"")()[:200].decode("utf-8", "replace")
            report["function_url_error"] = f"{type(exc).__name__}: {exc} {body}".strip()
            print(f"  function URL refused ({exc}); invoking through the Lambda API")

    if result is None:
        started = time.perf_counter()
        resp = lam.invoke(FunctionName=args.function, InvocationType="RequestResponse",
                          Payload=json.dumps(payload).encode())
        latency_ms = (time.perf_counter() - started) * 1000.0
        body = resp["Payload"].read().decode("utf-8", "replace")
        if resp.get("FunctionError"):
            result = {"ok": False, "error": f"function_error: {body[:300]}"}
        else:
            result = json.loads(body)
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

    # 6. The path a judge takes: the console's own API, over the public URL. -----
    if not args.no_api:
        base = args.api or _deployment_endpoint()
        if not base:
            report["failures"].append(
                "no public API base: pass --api or run deploy/deploy.py first")
        else:
            print(f"  public api {base}")
            try:
                pub = public_api_roundtrip(base, frames, f"verify-{stamp}", args.timeout)
                report["public_api"] = pub
                print(f"  health {pub['health']['status']} · backend "
                      f"{pub['health']['vision_backend']} · storage "
                      f"{pub['health']['storage']}")
                print(f"  session {pub['session_id']} created in {pub['create_ms']:.0f} ms, "
                      f"rescan in {pub['rescan_ms']:.0f} ms · {pub['events']} events "
                      f"{pub['kinds']}")
                if not pub["events"]:
                    report["failures"].append(
                        "the public API returned no events for a scene with known changes")
            except Exception as exc:
                body = getattr(exc, "read", lambda: b"")()[:200].decode("utf-8", "replace")
                report["public_api_error"] = f"{type(exc).__name__}: {exc} {body}".strip()
                report["failures"].append("the public console API did not complete a scan")

    report["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report["verdict"] = "PASS" if not report["failures"] else "FAIL"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(f"\n  {report['verdict']}  ->  {args.out}")
    for failure in report["failures"]:
        print(f"    - {failure}")
    return 0 if report["verdict"] == "PASS" else 1


def base64_decode_logs(resp: dict) -> str:  # pragma: no cover - invoke path only
    return base64.b64decode(resp.get("LogResult", "")).decode("utf-8", "replace")


if __name__ == "__main__":
    raise SystemExit(main())
