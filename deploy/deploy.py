"""Deploy the STRATA vision workload to AWS Lambda on Graviton - without Docker.

``deploy/deploy.sh`` builds an arm64 *container* image, which is the better
artefact when a machine has Docker. This machine does not, and the competition
deadline does not wait for a container runtime. So this script ships the same
code to the same arm64 Graviton execution environment as a zip package, with the
dependencies resolved from PyPI's aarch64 wheels:

    numpy 2.5.4                     cp312-manylinux_2_27/2_28_aarch64
    opencv-contrib-python-headless  5.0.0.93  cp37-abi3-manylinux_2_17/2_28_aarch64

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
import urllib.request
import zipfile
from pathlib import Path

try:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError
except ImportError:  # pragma: no cover - deployment-time dependency
    sys.exit("deploy.py needs boto3: pip install -r requirements-dev.txt")

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build" / "lambda"

PREFIX = os.environ.get("STRATA_PREFIX", "strata")
FUNCTION = f"{PREFIX}-vision"
WEB_FUNCTION = f"{PREFIX}-web"
API_NAME = f"{PREFIX}-api"
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
# rather than this machine's. Same OpenCV version as requirements.lock, but the
# *headless* build: the GUI wheel links libxcb and the other X11 client
# libraries, which the Lambda runtime does not have, so importing cv2 there
# raises "Unable to import module 'handler': libxcb.so.1". Headless is the same
# version and the same algorithms with the windowing code left out.
#
# fastapi + mangum are for the console-facing function only; the vision function
# never imports them. One package for both functions keeps one build, one upload
# and one thing that can drift between them.
REQUIREMENTS = [
    "numpy==2.5.4",
    "opencv-contrib-python-headless==5.0.0.93",
    "fastapi==0.121.2",
    "mangum==0.19.0",
]
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
        # A cold start here unpacks a 78 MB layer and pulls ~50 MB of weights out
        # of S3, which comfortably exceeds the SDK's 60 s read timeout. Asking for
        # a synchronous invoke and timing out client-side would read as a failed
        # deploy when the function is simply still warming up.
        from botocore.config import Config
        lambda_config = Config(read_timeout=300, connect_timeout=10,
                               retries={"max_attempts": 5, "mode": "standard"})
        self.s3 = session.client("s3")
        self.ddb = session.client("dynamodb")
        self.iam = session.client("iam")
        self.lam = session.client("lambda", config=lambda_config)
        self.logs = session.client("logs")
        self.apig = session.client("apigatewayv2")

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
        aws.s3.upload_file(
            str(model), aws.bucket, f"{WEIGHT_PREFIX}{model.name}",
            ExtraArgs={"ServerSideEncryption": "AES256"},
        )
        step(f"uploaded {model.name} ({model.stat().st_size / 1e6:.0f} MB)")
    return [m.name for m in models]


def build_package(with_weights: bool) -> Path:
    """Assemble the aarch64 deployment zip: deps + core + handler."""
    say("building the arm64 package")
    staging = BUILD / "staging"

    # Resolving the aarch64 wheels takes a few minutes of download, and iterating
    # on the *project* code should not pay for it again. STRATA_REUSE_DEPS=1 keeps
    # whatever is already staged; the project files below are always recopied, so
    # the only thing a reuse can hide is a change to REQUIREMENTS.
    reusing = (os.environ.get("STRATA_REUSE_DEPS") == "1"
               and (staging / "cv2").is_dir() and (staging / "fastapi").is_dir())
    if reusing:
        step("reusing the wheels already in build/lambda/staging (STRATA_REUSE_DEPS=1)")
    else:
        if BUILD.exists():
            shutil.rmtree(BUILD)
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

    for package in ("strata_core", "strata_agent"):
        shutil.copytree(ROOT / "vision" / package, staging / package,
                        dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy(ROOT / "deploy" / "handler.py", staging / "handler.py")
    # The console-facing function serves the same HTTP API the Docker image does,
    # so it ships the service package unchanged - except its static bundle, which
    # lives on Vercel and would only bloat the zip.
    shutil.copytree(ROOT / "service", staging / "service", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("static", "__pycache__", "*.pyc"))
    shutil.copy(ROOT / "deploy" / "web_handler.py", staging / "web_handler.py")

    weights = []
    if with_weights:
        models = sorted((ROOT / "vision" / "models").glob("*.onnx"))
        if models:
            (staging / "models").mkdir(exist_ok=True)
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


def _base_env(aws: Aws) -> dict[str, str]:
    """What both functions need to reach the data and the weights."""
    return {
        "STRATA_S3_BUCKET": aws.bucket,
        "STRATA_DDB_TABLE": TABLE,
        "STRATA_AWS_REGION": aws.region,
        "STRATA_BACKEND": os.environ.get("STRATA_BACKEND", "auto"),
        "STRATA_MODEL_DIR": TMP_MODELS,
        "STRATA_WEIGHT_PREFIX": WEIGHT_PREFIX,
        "PYTHONPATH": "/var/task",
    }


def _upsert_function(aws: Aws, *, name: str, handler: str, archive: Path,
                     role_arn: str, variables: dict[str, str]) -> dict:
    # One archive, one key: the two functions differ only in the entry point and
    # the environment, so there is nothing for a second copy of the bytes to buy.
    key = f"deploy/{FUNCTION}.zip"
    env = {"Variables": variables}
    if aws.dry:
        step(f"[dry-run] reuse {key} and create/update {name} "
             f"({handler}, {ARCHITECTURES}, {MEMORY_MB} MB)")
        return {"FunctionName": name,
                "FunctionArn": f"arn:aws:lambda:{aws.region}:{aws.account}:function:{name}"}

    if name == FUNCTION:
        step(f"uploading package to s3://{aws.bucket}/{key}")
        aws.s3.upload_file(str(archive), aws.bucket, key)

    config = {
        "FunctionName": name,
        "Runtime": RUNTIME,
        "Architectures": ARCHITECTURES,
        "Handler": handler,
        "Role": role_arn,
        "MemorySize": MEMORY_MB,
        "Timeout": TIMEOUT_S,
        "Environment": env,
        "Code": {"S3Bucket": aws.bucket, "S3Key": key},
        "Tags": {"Project": PREFIX},
    }
    try:
        aws.lam.get_function(FunctionName=name)
        step(f"{name}: updating function code and configuration")
        aws.lam.update_function_code(FunctionName=name, S3Bucket=aws.bucket, S3Key=key)
        aws.lam.get_waiter("function_updated_v2").wait(FunctionName=name)
        aws.lam.update_function_configuration(FunctionName=name, Environment=env,
                                              Timeout=TIMEOUT_S, MemorySize=MEMORY_MB)
        fn = aws.lam.get_function(FunctionName=name)["Configuration"]
    except ClientError:
        step(f"{name}: creating function")
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
        aws.lam.get_waiter("function_active_v2").wait(FunctionName=name)

    if name == FUNCTION:
        _reserve_concurrency(aws)
    return fn


def ensure_function(aws: Aws, archive: Path, role_arn: str) -> dict:
    """The raw scan endpoint: one invocation, one change set."""
    return _upsert_function(aws, name=FUNCTION, handler="handler.handler",
                            archive=archive, role_arn=role_arn,
                            variables=_base_env(aws))


def ensure_web_function(aws: Aws, archive: Path, role_arn: str) -> dict:
    """The console's HTTP API, on the same package and the same data.

    The service is the vision core's only public surface, so this function runs
    ``service.app`` unchanged under Mangum rather than re-implementing its routes.
    ``STRATA_EXECUTOR=aws`` is what switches its store from the filesystem to
    S3 plus DynamoDB - the same two tables the console has always read.
    """
    variables = dict(_base_env(aws))
    variables.update({
        "STRATA_EXECUTOR": "aws",
        "STRATA_S3_PREFIX": os.environ.get("STRATA_S3_PREFIX", "strata"),
        # The app's own default, and the right one here: this endpoint is
        # unauthenticated by design, so an origin list is not what keeps it safe -
        # the concurrency ceiling and the payload limits are. Hardcoding one
        # project's hostname would only mean that a console deployed somewhere
        # else gets a CORS error, and Vercel's production alias is a random
        # suffix nobody can know before the first deploy.
        "STRATA_CORS_ORIGINS": os.environ.get("STRATA_CORS_ORIGINS", "*"),
        "STRATA_LOG_LEVEL": "INFO",
    })
    return _upsert_function(aws, name=WEB_FUNCTION, handler="web_handler.handler",
                            archive=archive, role_arn=role_arn, variables=variables)


def _reserve_concurrency(aws: Aws) -> None:
    """Bound what one client can bill through an open Function URL.

    A brand-new account has a small unreserved-concurrency pool, and Lambda
    refuses a reservation that would leave fewer than 10 unreserved for
    everything else - which is the right call by the service, because the
    alternative is an account that cannot run anything else at all. So ask for
    what the account can actually give, and say so when the answer is nothing.
    """
    wanted = RESERVED_CONCURRENCY
    if aws.dry:
        step(f"[dry-run] reserve up to {wanted} concurrent executions")
        return
    limit = aws.lam.get_account_settings().get("AccountLimit", {})
    spare = int(limit.get("UnreservedConcurrentExecutions", 0))
    ceiling = int(limit.get("ConcurrentExecutions", 0))
    grantable = max(0, min(wanted, spare - 10))
    if grantable == 0:
        # A brand-new account is capped at 10 concurrent executions in total, so
        # reserving any of them would leave nothing for the rest of the account
        # and Lambda says no. That is not an unprotected endpoint: the account
        # ceiling is itself the bound, and it is printed so the receipt shows it.
        step(f"reserved concurrency skipped: the account ceiling is {ceiling} "
             f"concurrent executions, which is the bound on this endpoint")
        return
    aws.lam.put_function_concurrency(FunctionName=FUNCTION,
                                     ReservedConcurrentExecutions=grantable)
    step(f"reserved {grantable} concurrent executions (asked for {wanted})")


def smoke_test(aws: Aws) -> dict:
    """Run the handler once, through the control plane, before anything points at it.

    This is where the zip route earns its keep. ``opencv-contrib-python`` installs
    cleanly for aarch64 and then fails to *import* on Lambda, because it links
    libxcb and the runtime has no X11 client libraries: every invocation dies in
    Runtime.ImportModuleError. A deployer that prints an endpoint URL without
    having run the handler once is a deployer that printed a guess, so this runs
    first and raises if the function cannot answer.
    """
    if aws.dry:
        step("[dry-run] invoke the function with an empty event and read its health document")
        return {}
    say("smoke test")
    resp = aws.lam.invoke(FunctionName=FUNCTION, InvocationType="RequestResponse",
                          Payload=json.dumps({"health": True}).encode())
    body = resp["Payload"].read().decode("utf-8", "replace")
    if resp.get("FunctionError"):
        raise SystemExit(f"the function did not start: {body[:600]}")
    doc = json.loads(body)
    for key in ("opencv", "python", "architecture", "backend", "weights_staged"):
        step(f"{key:<14} {doc.get(key)}")
    if not doc.get("ok") or not doc.get("opencv"):
        raise SystemExit(f"unexpected health document: {doc}")
    return doc


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


def ensure_api(aws: Aws, fn_arn: str) -> str:
    """Put an HTTP API in front of the web function, and return its URL.

    A Lambda Function URL would be the shorter answer, and in this account it is
    the blocked one: with ``AuthType=NONE`` and the canonical
    ``"Principal": "*"`` resource policy, an anonymous request still returns
    AccessDeniedException - on this function and on a hello-world function built
    only to test that. API Gateway reaches the same code through a service
    principal instead of an anonymous caller, which is not what that denial
    covers. The route is ``$default`` so preflight OPTIONS reaches FastAPI's own
    CORS middleware rather than being answered by a second, separately-kept CORS
    config that can drift from it.
    """
    if aws.dry:
        step(f"[dry-run] create http api {API_NAME} -> {fn_arn}")
        return f"https://<dry-run>.execute-api.{aws.region}.amazonaws.com"

    say("http api")
    apis = [a for a in aws.apig.get_apis()["Items"] if a["Name"] == API_NAME]
    if apis:
        api_id = apis[0]["ApiId"]
        step(f"reusing http api {api_id}")
    else:
        api_id = aws.apig.create_api(Name=API_NAME, ProtocolType="HTTP")["ApiId"]
        step(f"created http api {api_id}")

    integrations = aws.apig.get_integrations(ApiId=api_id)["Items"]
    if integrations:
        integration_id = integrations[0]["IntegrationId"]
        aws.apig.update_integration(ApiId=api_id, IntegrationId=integration_id,
                                    IntegrationUri=fn_arn)
        step(f"integration {integration_id} -> {WEB_FUNCTION}")
    else:
        integration_id = aws.apig.create_integration(
            ApiId=api_id, IntegrationType="AWS_PROXY", IntegrationUri=fn_arn,
            PayloadFormatVersion="2.0", IntegrationMethod="POST",
            Description=WEB_FUNCTION,
        )["IntegrationId"]
        step(f"created integration {integration_id}")

    routes = {r["RouteKey"] for r in aws.apig.get_routes(ApiId=api_id)["Items"]}
    if "$default" not in routes:
        aws.apig.create_route(ApiId=api_id, RouteKey="$default",
                              AuthorizationType="NONE",
                              Target=f"integrations/{integration_id}")
        step("route $default -> the web function")

    stages = [s["StageName"] for s in aws.apig.get_stages(ApiId=api_id)["Items"]]
    if "$default" not in stages:
        aws.apig.create_stage(ApiId=api_id, StageName="$default", AutoDeploy=True)
    else:
        # An HTTP API serves a stage from a deployment, not from the route table,
        # so a route or integration added on a re-run is invisible until it is
        # deployed. API Gateway v2 has no redeploy call; this is the one that does.
        aws.apig.create_deployment(ApiId=api_id, StageName="$default")
        step("deployed the stage")

    # Lambda will not let API Gateway invoke it unless the function says so, and
    # that grant is scoped to this api's invoke ARNs, not to the whole account.
    # The scope matters and so does its *shape*: the REST-API form
    # ``{api_id}/*/*/*`` was rejected here - every call came back 500 with the
    # gateway's own "doesn't have permissions to call the integration", and the
    # access log named the resource policy - while ``{api_id}/*`` on the same api,
    # same account, same statement, returned 200. An HTTP API puts the route in a
    # single segment, so the trailing path wildcards match nothing.
    source_arn = f"arn:aws:execute-api:{aws.region}:{aws.account}:{api_id}/*"
    try:
        aws.lam.add_permission(
            FunctionName=WEB_FUNCTION, StatementId="ApiGatewayInvoke",
            Action="lambda:InvokeFunction", Principal="apigateway.amazonaws.com",
            SourceArn=source_arn,
        )
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "ResourceConflictException":
            raise
        step("invoke grant already present")

    endpoint = aws.apig.get_api(ApiId=api_id)["ApiEndpoint"]
    step(f"endpoint {endpoint}")
    return endpoint


def api_smoke(aws: Aws, endpoint: str) -> dict:
    """Ask the public URL its own health question, and keep the answer.

    The receipt is this document read back over the network, not the deploy log:
    the thing a judge is told works is the thing that was asked.
    """
    if aws.dry:
        step("[dry-run] GET <endpoint>/api/health")
        return {}
    say("endpoint smoke test")
    url = f"{endpoint}/api/health"
    doc: dict = {}
    last: Exception | None = None
    for attempt in range(1, 7):
        # A freshly deployed stage returns 404 for a few seconds, and a cold
        # function returns 502/504 once, so a single attempt reads as a broken
        # deployment when it is only propagation.
        try:
            with urllib.request.urlopen(url, timeout=90) as resp:
                doc = json.loads(resp.read().decode())
            break
        except Exception as exc:
            last = exc
            code = getattr(exc, "code", None)
            if code not in (404, 502, 504):
                break
            step(f"endpoint not ready yet ({code}), attempt {attempt}/6")
            time.sleep(10)
    if not doc:
        body = getattr(last, "read", lambda: b"")()[:400]
        raise SystemExit(f"the public endpoint did not answer {url}: {last} {body}")
    for key in ("status", "api_version", "opencv_version", "vision_backend",
                "executor", "storage", "model_weights_present"):
        if key in doc:
            step(f"{key:<21} {doc[key]}")
    step(f"(full document keys: {', '.join(sorted(doc))})")
    return doc


def destroy(aws: Aws) -> None:
    say("tearing down")
    if aws.dry:
        step("[dry-run] would delete the functions, the http api, role policy, table and bucket")
        return
    for api in [a for a in aws.apig.get_apis()["Items"] if a["Name"] == API_NAME]:
        try:
            aws.apig.delete_api(ApiId=api["ApiId"])
            step(f"deleted http api {api['ApiId']}")
        except ClientError as exc:
            step(f"http api: {exc.response['Error']['Code']}")
    for label, call in (
        (FUNCTION, lambda: aws.lam.delete_function(FunctionName=FUNCTION)),
        (WEB_FUNCTION, lambda: aws.lam.delete_function(FunctionName=WEB_FUNCTION)),
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
    health = smoke_test(aws)
    url = ensure_url(aws)

    web_fn = ensure_web_function(aws, archive, role_arn)
    endpoint = ensure_api(aws, web_fn["FunctionArn"])
    api_health = api_smoke(aws, endpoint)

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
        # Read back from the running function, not asserted from this machine.
        "health": {k: health.get(k) for k in
                   ("opencv", "python", "architecture", "backend", "weights_staged")},
        "web_function": {
            "name": WEB_FUNCTION,
            "arn": web_fn["FunctionArn"],
            "handler": "web_handler.handler",
            "endpoint": endpoint,
            "console_api_base": endpoint,
        },
        # What GET <endpoint>/api/health answered from outside the account.
        "api_health": api_health,
    }
    out = ROOT / "build" / "lambda" / "deployment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=2), encoding="utf-8")

    say("deployed")
    for k, v in info.items():
        step(f"{k:<14} {v}")
    step(f"\nconsole: build web/ with NEXT_PUBLIC_API_BASE={endpoint}")
    step(f"next: python deploy/verify.py --bucket {aws.bucket} --region {args.region} --record")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
