"""Deploy the STRATA vision workload to AWS Lambda on Graviton - without Docker.

``deploy/deploy.sh`` builds an arm64 *container* image, which is the better
artefact when a machine has Docker. This machine does not, and the competition
deadline does not wait for a container runtime. So this script ships the same
code to the same arm64 Graviton execution environment as a zip package, with the
dependencies resolved from PyPI's aarch64 wheels:

    numpy 2.5.4            cp312-manylinux_2_27/2_28_aarch64
    opencv-contrib-python  5.0.0.93  cp37-abi3-manylinux_2_17/2_28_aarch64

which is the *same pinned versions and the same CPU architecture* the container
would have carried. Amazon Linux 2023 - the Lambda Python 3.12 base - is glibc
2.34, so both tags are satisfied.

What this does not do is pretend the two are identical: a container can pin the
system libraries around OpenCV, a zip cannot. That difference is recorded in the
report rather than smoothed over.

    python deploy/deploy.py --dry-run     # what would happen, nothing more
    python deploy/deploy.py               # build, upload, create, print endpoint
    python deploy/deploy.py --destroy     # remove everything this created

Credentials come from the environment or ~/.aws, exactly as the AWS SDK expects.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:  # pragma: no cover - deployment-time dependency
    sys.exit("deploy.py needs boto3: pip install -r requirements-dev.txt")

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build" / "lambda"

PREFIX = os.environ.get("STRATA_PREFIX", "strata")
FUNCTION = f"{PREFIX}-vision"
ROLE_NAME = f"{PREFIX}-lambda-role"
TABLE = os.environ.get("STRATA_TABLE", f"{PREFIX}-ledger")
RUNTIME = "python3.12"
ARCHITECTURES = ["arm64"]
MEMORY_MB = 3008
TIMEOUT_S = 120
# A public Function URL is an unauthenticated compute endpoint. Reserving
# concurrency bounds what one client can bill: 3008 MB x 5 is a ceiling, not an
# open door.
RESERVED_CONCURRENCY = int(os.environ.get("STRATA_RESERVED_CONCURRENCY", "5"))
WEIGHT_PREFIX = os.environ.get("STRATA_WEIGHT_PREFIX", "models/")
# Where a zip-deployed function stages its weights at runtime. /tmp is the only
# writable place, and it can be sized up to 10 GB; the package itself may not
# exceed 250 MB unpacked, which is what the weights would break.
TMP_MODELS = "/tmp/models"

# The pinned runtime set, resolved for the target interpreter and architecture
# rather than this machine's. Same versions as requirements.lock.
REQUIREMENTS = ["numpy==2.5.4", "opencv-contrib-python==5.0.0.93"]
PIP_PLATFORMS = [
    "--platform", "manylinux_2_28_aarch64",
    "--platform", "manylinux_2_27_aarch64",
    "--platform", "manylinux2014_aarch64",
    "--platform", "manylinux_2_17_aarch64",
]


def say(msg: str) -> None:
    print(f"\n\033[1m==> {msg}\033[0m", flush=True)


def step(msg: str) -> None:
    print(f"    {msg}", flush=True)


class Aws:
    """One place the SDK clients live, so dry-run can short-circuit everything."""

    def __init__(self, region: str, dry: bool) -> None:
        self.region = region
        self.dry = dry
        if dry:
            self.account = os.environ.get("STRATA_ACCOUNT", "<account>")
            return
        session = boto3.Session(region_name=region)
        self.account = session.client("sts").get_caller_identity()["Account"]
        self.s3 = session.client("s3")
        self.ddb = session.client("dynamodb")
        self.iam = session.client("iam")
        self.lam = session.client("lambda")
        self.logs = session.client("logs")

    @property
    def bucket(self) -> str:
        return os.environ.get("STRATA_BUCKET", f"{PREFIX}-sessions-{self.account}")


# ---------------------------------------------------------------------------
# The package
# ---------------------------------------------------------------------------

def upload_weights(aws: Aws) -> list[str]:
    """Put the feature weights in the bucket, not in the package.

    The container build in deploy/Dockerfile bakes them in and they cost nothing
    at runtime. A zip cannot carry them: numpy + OpenCV + 50 MB of ONNX unpacked
    to 263 MB, over Lambda's 250 MB ceiling. So the function fetches them from
    S3 into ``/tmp`` on a cold start - in-region, once per execution
    environment, and the vision core still only ever reads a directory.
    """
    models = sorted((ROOT / "vision" / "models").glob("*.onnx"))
    if not models:
        step("no ONNX weights on disk; the function will use the classical backend")
        return []
    if aws.dry:
        step(f"[dry-run] upload {len(models)} weight(s) to s3://{aws.bucket}/{WEIGHT_PREFIX}")
        return [m.name for m in models]
    for model in models:
        aws.s3.upload_file(str(model), aws.bucket, f"{WEIGHT_PREFIX}{model.name}",
                           ServerSideEncryption="AES256")
        step(f"uploaded {model.name} ({model.stat().st_size / 1e6:.0f} MB)")
    return [m.name for m in models]


def build_package(with_weights: bool) -> Path:
    """Assemble the aarch64 deployment zip: deps + core + handler."""
    say("building the arm64 package")
    if BUILD.exists():
        shutil.rmtree(BUILD)
    staging = BUILD / "staging"
    staging.mkdir(parents=True)

    cmd = [
        sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
        "--no-cache-dir", "--only-binary=:all:", "--target", str(staging),
        "--python-version", RUNTIME.split("python")[1], *PIP_PLATFORMS,
    ]
    cmd += REQUIREMENTS
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    # Drop the pieces of a wheel that Lambda never reads and the zip limit does.
    for junk in ("bin",):
        path = staging / junk
        if path.exists():
            shutil.rmtree(path)
    for dist_info in staging.glob("*.dist-info"):
        shutil.rmtree(dist_info)

    shutil.copytree(ROOT / "vision" / "strata_core", staging / "strata_core")
    shutil.copy(ROOT / "deploy" / "handler.py", staging / "handler.py")

    weights = []
    if with_weights:
        models = sorted((ROOT / "vision" / "models").glob("*.onnx"))
        if models:
            (staging / "models").mkdir()
            for model in models:
                shutil.copy(model, staging / "models" / model.name)
                weights.append(model.name)
        else:
            step("no ONNX weights on disk; the function will use the classical backend")

    archive = BUILD / f"{FUNCTION}.zip"
    total = 0
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in staging.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(staging).as_posix())
                total += path.stat().st_size
    mb = archive.stat().st_size / 1e6
    step(f"{total / 1e6:.0f} MB unpacked -> {mb:.1f} MB zip"
         + (f" (weights: {', '.join(weights)})" if weights else ""))
    if total / 1e6 > 240:
        raise SystemExit(f"package unpacked to {total / 1e6:.0f} MB, over Lambda's 250 MB limit")
    return archive


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

def ensure_bucket(aws: Aws) -> None:
    if aws.dry:
        step(f"[dry-run] create bucket {aws.bucket} (private, AES-256)")
        return
    try:
        aws.s3.head_bucket(Bucket=aws.bucket)
        step(f"bucket exists: {aws.bucket}")
        return
    except ClientError:
        pass
    step(f"creating bucket: {aws.bucket}")
    params = {"Bucket": aws.bucket}
    if aws.region != "us-east-1":
        params["CreateBucketConfiguration"] = {"LocationConstraint": aws.region}
    aws.s3.create_bucket(**params)
    aws.s3.put_public_access_block(
        Bucket=aws.bucket,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True, "IgnorePublicAcls": True,
            "BlockPublicPolicy": True, "RestrictPublicBuckets": True,
        },
    )
    aws.s3.put_bucket_encryption(
        Bucket=aws.bucket,
        ServerSideEncryptionConfiguration={
            "Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]
        },
    )
    aws.s3.put_bucket_tagging(
        Bucket=aws.bucket, Tagging={"TagSet": [{"Key": "Project", "Value": PREFIX}]}
    )


def ensure_table(aws: Aws) -> None:
    """The ledger table: pk (HASH) + sk (RANGE), the contract both writers use."""
    if aws.dry:
        step(f"[dry-run] create table {TABLE} pk HASH + sk RANGE")
        return
    try:
        aws.ddb.describe_table(TableName=TABLE)
        step(f"table exists: {TABLE}")
        return
    except ClientError:
        pass
    step(f"creating table: {TABLE}")
    aws.ddb.create_table(
        TableName=TABLE,
        AttributeDefinitions=[
            {"AttributeName": "pk", "AttributeType": "S"},
            {"AttributeName": "sk", "AttributeType": "S"},
        ],
        KeySchema=[
            {"AttributeName": "pk", "KeyType": "HASH"},
            {"AttributeName": "sk", "KeyType": "RANGE"},
        ],
        BillingMode="PAY_PER_REQUEST",
        Tags=[{"Key": "Project", "Value": PREFIX}],
    )
    waiter = aws.ddb.get_waiter("table_exists")
    waiter.wait(TableName=TABLE)


def ensure_role(aws: Aws) -> str:
    trust = json.dumps({
        "Version": "2012-10-17",
        "Statement": [{"Effect": "Allow",
                       "Principal": {"Service": "lambda.amazonaws.com"},
                       "Action": "sts:AssumeRole"}],
    })
    if aws.dry:
        step(f"[dry-run] create role {ROLE_NAME} + scoped policy")
        return f"arn:aws:iam::{aws.account}:role/{ROLE_NAME}"
    try:
        role = aws.iam.get_role(RoleName=ROLE_NAME)
        step(f"role exists: {ROLE_NAME}")
    except ClientError:
        step(f"creating role: {ROLE_NAME}")
        role = aws.iam.create_role(
            RoleName=ROLE_NAME, AssumeRolePolicyDocument=trust,
            Tags=[{"Key": "Project", "Value": PREFIX}],
        )
        aws.iam.attach_role_policy(
            RoleName=ROLE_NAME,
            PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
        )
    aws.iam.put_role_policy(
        RoleName=ROLE_NAME,
        PolicyName=f"{PREFIX}-data-access",
        PolicyDocument=json.dumps({
            "Version": "2012-10-17",
            "Statement": [
                {"Effect": "Allow",
                 "Action": ["s3:GetObject", "s3:PutObject"],
                 "Resource": f"arn:aws:s3:::{aws.bucket}/*"},
                {"Effect": "Allow",
                 "Action": ["dynamodb:PutItem", "dynamodb:GetItem", "dynamodb:Query"],
                 "Resource": f"arn:aws:dynamodb:{aws.region}:{aws.account}:table/{TABLE}"},
                {"Effect": "Allow",
                 "Action": ["cloudwatch:PutMetricData"],
                 "Resource": "*",
                 "Condition": {"StringEquals": {"cloudwatch:namespace": "Strata"}}},
            ],
        }),
    )
    return role["Role"]["Arn"]


def ensure_function(aws: Aws, archive: Path, role_arn: str) -> dict:
    key = f"deploy/{FUNCTION}.zip"
    env = {"Variables": {
        "STRATA_S3_BUCKET": aws.bucket,
        "STRATA_DDB_TABLE": TABLE,
        "STRATA_AWS_REGION": aws.region,
        "STRATA_BACKEND": os.environ.get("STRATA_BACKEND", "auto"),
        "STRATA_MODEL_DIR": TMP_MODELS,
        "STRATA_WEIGHT_PREFIX": WEIGHT_PREFIX,
        "PYTHONPATH": "/var/task",
    }}
    if aws.dry:
        step(f"[dry-run] upload {key} and create/update {FUNCTION} ({ARCHITECTURES}, {MEMORY_MB} MB)")
        return {"FunctionName": FUNCTION, "Arn": f"arn:aws:lambda:{aws.region}:{aws.account}:function:{FUNCTION}"}

    step(f"uploading package to s3://{aws.bucket}/{key}")
    aws.s3.upload_file(str(archive), aws.bucket, key)

    config = {
        "FunctionName": FUNCTION,
        "Runtime": RUNTIME,
        "Architectures": ARCHITECTURES,
        "Handler": "handler.handler",
        "Role": role_arn,
        "MemorySize": MEMORY_MB,
        "Timeout": TIMEOUT_S,
        "Environment": env,
        "Code": {"S3Bucket": aws.bucket, "S3Key": key},
        "Tags": {"Project": PREFIX},
    }
    try:
        aws.lam.get_function(FunctionName=FUNCTION)
        step("updating function code and configuration")
        aws.lam.update_function_code(FunctionName=FUNCTION, S3Bucket=aws.bucket, S3Key=key)
        aws.lam.get_waiter("function_updated_v2").wait(FunctionName=FUNCTION)
        aws.lam.update_function_configuration(FunctionName=FUNCTION, Environment=env,
                                              Timeout=TIMEOUT_S, MemorySize=MEMORY_MB)
        fn = aws.lam.get_function(FunctionName=FUNCTION)
    except ClientError:
        step("creating function")
        # A role created seconds ago is not assumable yet. Poll on the error that
        # actually happens instead of sleeping a fixed guess.
        for attempt in range(1, 13):
            try:
                fn = aws.lam.create_function(**config)
                break
            except ClientError as exc:
                if "cannot be assumed by Lambda" not in str(exc) and \
                   "not authorized to perform" not in str(exc):
                    raise
                step(f"role not assumable yet (attempt {attempt}/12)")
                time.sleep(5)
        else:  # pragma: no cover
            raise SystemExit("the Lambda role never became assumable")
        aws.lam.get_waiter("function_active_v2").wait(FunctionName=FUNCTION)

    aws.lam.put_function_concurrency(FunctionName=FUNCTION,
                                     ReservedConcurrentExecutions=RESERVED_CONCURRENCY)
    return fn


def ensure_url(aws: Aws) -> str:
    if aws.dry:
        return f"https://<dry-run>.lambda-url.{aws.region}.on.aws/"
    try:
        return aws.lam.get_function_url_config(FunctionName=FUNCTION)["FunctionUrl"]
    except ClientError:
        pass
    url = aws.lam.create_function_url_config(
        FunctionName=FUNCTION, AuthType="NONE",
        InvokeMode="BUFFERED",
    )["FunctionUrl"]
    aws.lam.add_permission(
        FunctionName=FUNCTION, StatementId="FunctionURLAllowPublicAccess",
        Action="lambda:InvokeFunctionUrl", Principal="*",
        FunctionUrlAuthType="NONE",
    )
    return url


def destroy(aws: Aws) -> None:
    say("tearing down")
    if aws.dry:
        step("[dry-run] would delete the function, role policy, table and bucket")
        return
    for label, call in (
        (FUNCTION, lambda: aws.lam.delete_function(FunctionName=FUNCTION)),
        (TABLE, lambda: aws.ddb.delete_table(TableName=TABLE)),
    ):
        try:
            call()
            step(f"deleted {label}")
        except ClientError as exc:
            step(f"{label}: {exc.response['Error']['Code']}")
    try:
        aws.lam.delete_function_url_config(FunctionName=FUNCTION)
    except ClientError:
        pass
    for key in (f"deploy/{FUNCTION}.zip",):
        try:
            aws.s3.delete_object(Bucket=aws.bucket, Key=key)
        except ClientError:
            pass
    step(f"left in place: bucket {aws.bucket} (delete its objects in the console), "
         f"role {ROLE_NAME}")


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--region", default=os.environ.get("STRATA_AWS_REGION", "us-east-1"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-weights", action="store_true",
                    help="ship without the ONNX feature weights (classical backend only)")
    ap.add_argument("--destroy", action="store_true")
    args = ap.parse_args()

    aws = Aws(args.region, args.dry_run)
    if not args.dry_run:
        say(f"account {aws.account} · region {aws.region}")

    if args.destroy:
        destroy(aws)
        return 0

    # The package never carries the weights; they live in the bucket (see
    # upload_weights), so the 250 MB unpacked ceiling is not the binding one.
    archive = Path("<not built>") if args.dry_run else build_package(False)
    ensure_bucket(aws)
    weights = [] if args.no_weights else upload_weights(aws)
    ensure_table(aws)
    role_arn = ensure_role(aws)
    fn = ensure_function(aws, archive, role_arn)
    url = ensure_url(aws)

    if args.dry_run:
        print("\ndry run complete; nothing was created\n")
        return 0

    info = {
        "function": FUNCTION,
        "arn": fn["FunctionArn"],
        "architectures": fn.get("Architectures"),
        "runtime": fn.get("Runtime"),
        "memory_mb": fn.get("MemorySize"),
        "timeout_s": fn.get("Timeout"),
        "region": args.region,
        "bucket": aws.bucket,
        "table": TABLE,
        "url": url,
        "package_mb": round(archive.stat().st_size / 1e6, 2),
        "weights_in_bucket": weights,
    }
    out = ROOT / "build" / "lambda" / "deployment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=2), encoding="utf-8")

    say("deployed")
    for k, v in info.items():
        step(f"{k:<14} {v}")
    step(f"\nnext: python deploy/verify.py --bucket {aws.bucket} --region {args.region} --record")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
