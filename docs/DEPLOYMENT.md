# Deployment

STRATA runs in two shapes from the same code:

- **Local** — one Python process serving the API and the built console on one port.
- **AWS** — the vision workload on Graviton (arm64) Lambda, with S3, DynamoDB and
  CloudWatch behind it, plus a reproducible benchmark on Graviton4 with the
  Cloud-Optimized OpenCV build. The AWS half has two packaging routes: a
  container image (`deploy/deploy.sh`, needs Docker) and a zip built from the
  aarch64 wheels (`deploy/deploy.py`, needs only boto3). The zip route is the one
  that has been executed; §2 says what it cost to get there.

The vision core is a pure function of its inputs, so no vision code is rewritten
between the two. What does differ is the environment around it — the zip has to
carry the headless OpenCV wheel and fetch its weights at cold start, and the
service only finds those weights if `STRATA_MODEL_DIR` reaches it — and both of
those are deployment configuration, not a second implementation.

---

## 1. Local

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows
# source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
python scripts/fetch_models.py    # optional, ~52 MB
python scripts/serve.py
# → http://127.0.0.1:8000
```

Python 3.11 or newer installs the ranges. The locked set
(`requirements.lock`, the environment every published figure was measured on)
pins numpy 2.5.4, which requires 3.12+.

With `PORT` set, the service binds `0.0.0.0` automatically. A service bound to
loopback is unreachable from outside its own container, which is a silent and
total failure on every hosting platform.

| Variable | Purpose |
|---|---|
| `PORT` / `STRATA_PORT` | Port to listen on |
| `STRATA_HOST` | Bind address (default `0.0.0.0` when `PORT` is set, else loopback) |
| `STRATA_EXECUTOR` | `local` (default) or `aws` |
| `STRATA_DATA_DIR` | Where the local store writes |
| `STRATA_S3_BUCKET` / `STRATA_DDB_TABLE` / `STRATA_AWS_REGION` | Managed storage |
| `STRATA_CORS_ORIGINS` | Allowed origins |
| `STRATA_STATIC_DIR` | Built console to serve |

---

## 1.1 Deploying to a container host

The image serves the API and the prebuilt console on one port, so any container
host works with no further configuration:

```bash
docker build -t strata .
docker run -p 8000:8000 strata
# → http://127.0.0.1:8000
```

`render.yaml` is included as the shortest path to a URL: push, then
Render → New → Blueprint → pick the repository. Railway, Fly.io, Cloud Run and
ECS take the same image.

### The console on Vercel — the live app

**<https://strata-console-rouge.vercel.app>** is the application, publicly
reachable: no account, no login, nothing to install. It is the static export of
`web/`, built with `NEXT_PUBLIC_API_BASE` pointed at the deployed AWS API, so the
page and the pipeline are two hosts and the browser calls AWS directly. Verified
from outside this machine just now: the production URL answers `200` to an
anonymous request, the page renders the five sample sites, and the network log
shows `GET https://c3rdzrdswf.execute-api.us-east-1.amazonaws.com/api/sites → 200`.

```bash
# Rebuild it by hand from web/ (a push to master does this automatically now):
NEXT_PUBLIC_API_BASE=https://c3rdzrdswf.execute-api.us-east-1.amazonaws.com \
  npx vercel deploy --prod
```

Without that build-time variable the console calls `/api/...` on its own origin,
which a static export does not serve — it reaches the interface, lists nothing,
and reports that the service is unreachable. The variable is `NEXT_PUBLIC_API_BASE`
(`web/lib/api.ts`); an `STRATA_API_BASE` named it in an earlier revision of this
document, which would have sent anyone who followed it to a build that still
talked to itself.

Verified in a browser against the deployed API, not against a local server: the
sample sites load from `…execute-api…/api/sites`, a rescan returns four typed
events with their confidences, and approving the change set writes the history
row — `Recorded · scan #0 · finalized · alignment 92% · neural-lightglue`. That
last part is the ledger's semantics, and it is the reason the gate is clicked in
the check rather than assumed: an approved scan used to be recorded as declined.

**Two things worth knowing about that URL.** First, the login wall some earlier
revision of this document warned about applies to *deployment* URLs
(`strata-console-<hash>-cargomatrix1.vercel.app` → `302` to Vercel
Authentication), not to the project's production URL, which is the one above and
is open. The earlier warning was written from a deployment URL and was therefore
wrong about the thing judges are given. Second, every push to `master` used to
produce a **failed** Vercel build: the project's root directory is the repository
while the Next app lives in `web/`, so Vercel ran `next build` one level too high
and died on "Couldn't find any `pages` or `app` directory". Root Directory is a
dashboard-only setting, so the fix lives in the repository instead —
[`vercel.json`](../vercel.json) sets `installCommand`, `buildCommand` and
`outputDirectory` around `web/`, and `"framework": null` because the artefact is
a static export, not a Next server: with the Next builder still enabled the build
succeeded and Vercel then rejected the result looking for
`web/out/routes-manifest.json`, which an export does not produce. Push-triggered
production builds are green from that change onward.

The first deploy attempt was rejected by Vercel's dependency gate: `next@15.1.6`
pulls a `postcss` with published advisories, and the platform blocks the
deployment rather than warning. `web/` is on `next@16.4.2` now — `npm audit`
reports zero vulnerabilities and `tsc --noEmit` is clean.

**Give it at least 2 GB of memory.** The pipeline holds two decoded frames, a
canonical warp and up to 2000 keypoints per side, and runs two scans
concurrently by default. At 512 MB it will not fail cleanly — it will thrash.
On a 512 MB tier set `STRATA_MAX_CONCURRENT_SCANS=1` and expect the neural
backend to be slow rather than broken.

### Why not Vercel

Vercel is a good host for the console and a poor host for this backend, for
three specific reasons rather than a general one:

1. **Function duration.** A scan takes ~2–15 s depending on backend and frame
   size. Vercel's Hobby tier caps a function at 10 s; a scan would be killed
   mid-pipeline.
2. **No writable filesystem.** The local `Store` writes sessions to disk. On
   Vercel that path is read-only outside `/tmp`, and `/tmp` does not survive
   between invocations — so every scan would lose its baseline. It would need
   the S3 backend, which means running in AWS anyway.
3. **Bundle size.** OpenCV 5 plus NumPy is already ~100 MB, and the feature
   weights add ~50 MB, against a 250 MB limit that also has to hold the runtime.

If you want a Vercel deployment, the honest split is: static console on Vercel,
vision workload on a container host or Lambda. `web/` exports statically
(`npm run build` → `web/out`), and `STRATA_API_BASE` points the console at a
remote API. That is a two-service architecture and it is more moving parts than
one container, which is why the default is the single image.

---

## 2. AWS — the vision workload on Graviton

```bash
export AWS_REGION=us-east-1
./deploy/deploy.sh
```

Idempotent. It creates, or reuses if present:

| Resource | Name | Notes |
|---|---|---|
| S3 bucket | `strata-sessions-<account>` | Private, AES-256, public access blocked |
| DynamoDB table | `strata-ledger` | On-demand; `pk` (HASH) + `sk` (RANGE) |
| ECR repository | `strata-vision` | Scan on push |
| IAM role | `strata-lambda-role` | S3 + DynamoDB scoped to the two resources above |
| Lambda function | `strata-vision` | **arm64**, container image, 3008 MB, 120 s |
| Concurrency | reserved 5 | Bounds what one client can bill through an open Function URL |
| Function URL | — | Public invocation endpoint. Refused for anonymous callers in the account used here — see the note below on why the public endpoint is an HTTP API |

The `pk`/`sk` schema is the storage contract both writers use: `pk` is the
parent of a record key, `sk` its leaf. `service/store.py` and
`deploy/handler.py` write it, and `tests/contract.py` reads it back out of
`deploy.sh` so the three cannot drift apart silently — which is exactly how this
table was wrong before anything had ever been deployed.

`./deploy/deploy.sh --dry-run` prints every command it would run and touches
nothing, including the account lookup. It is the way to read the plan without
credentials.

### The route that has actually been executed: `deploy/deploy.py`

`deploy.sh` needs Docker. This machine has none, and the deadline does not wait
for a container runtime, so `deploy/deploy.py` ships the same code to the same
arm64 execution environment as a **zip package**, with the dependencies resolved
from PyPI's aarch64 wheels:

| Resource | Name | Notes |
|---|---|---|
| Lambda function | `strata-vision` | **arm64**, zip, `handler.handler`, 3008 MB, 120 s — the raw scan endpoint |
| Lambda function | `strata-web` | **arm64**, zip, `web_handler.handler` — `service/app.py` under Mangum, i.e. the console's own HTTP API |
| API Gateway HTTP API | `strata-api` | `$default` route → `strata-web`, `https://c3rdzrdswf.execute-api.us-east-1.amazonaws.com` |
| S3 / DynamoDB / role | as above | one bucket, one table, one scoped role, shared by both functions |
| Feature weights | `models/` in the bucket | ~50 MB ONNX; staged into `/tmp/models` at cold start, because a zip carrying them unpacks past Lambda's 250 MB ceiling |

Three things that only running it could teach, each now fixed in the code rather
than worked around by hand:

- **`opencv-contrib-python` does not import on Lambda.** It installs cleanly for
  `manylinux_2_28_aarch64` and then dies at import with
  `libxcb.so.1: cannot open shared object file`, because the GUI wheel links the
  X11 client libraries and the runtime has none. The deployment is
  `opencv-contrib-python-**headless**==5.0.0.93` — the same version, the same
  algorithms, the windowing left out. `deploy/Dockerfile` carries the same swap;
  it has not been rebuilt here, so treat that line as unverified until someone
  with Docker builds it.
- **An anonymous Lambda Function URL is refused in this account.** With
  `AuthType=NONE` and the canonical `"Principal": "*"` resource policy — the one
  the AWS documentation and the console both generate — every request comes back
  `403 AccessDeniedException`. It does that for `strata-vision` and it did it for
  a hello-world function created only to test the hypothesis, so it is an account
  setting, not a deployment bug. API Gateway reaches the same code through a
  service principal and is not covered by that refusal, which is why the public
  endpoint is an HTTP API and not a Function URL. `verify.py` records the refusal
  it hit rather than hiding it.
- **The resource-policy scope has a different shape for HTTP APIs.**
  `arn:aws:execute-api:…:<api_id>/*/*/*` — the REST-API form everyone copies —
  gave 500s with no Lambda invocation at all; the access log said *"the IAM role
  configured on the integration … doesn't have permissions to call the
  integration"*. `<api_id>/*` on the same api, same account, same statement
  returns 200. `ensure_api()` now grants the form that was measured to work.

One deliberate loosening: `STRATA_CORS_ORIGINS` defaults to `*` on the deployed
function. Hardcoding a project hostname was worse than useless — Vercel's
production alias is a random suffix no one knows before the first deploy, so the
console would have been CORS-blocked from its own URL. The endpoint is
unauthenticated by design, so origins were never what bounded it; the concurrency
ceiling and the payload limits in `/api/health` are.

The container is built for `linux/arm64` with `docker buildx`, so it can be built
from an x86 machine:

```dockerfile
FROM public.ecr.aws/lambda/python:3.12-arm64
RUN dnf install -y libgomp
```

`libgomp` is not incidental. Without the OpenMP runtime, OpenCV's `parallel_for`
falls back to single-threaded execution — the pipeline still produces correct
results, but contour extraction and thresholding quietly run at a fraction of
their throughput. A missing `libgomp` looks like a slow instance rather than a
missing dependency, which is why it is installed explicitly.

### Invocation

```bash
aws lambda invoke --function-name strata-vision --region us-east-1 \
  --cli-binary-format raw-in-base64-out \
  --payload '{"site_id":"bay-4","session_id":"s-002",
              "baseline":["bay-4/base.jpg"],"rescan":["bay-4/scan.jpg"],
              "record":true}' \
  /tmp/out.json && cat /tmp/out.json
```

Returns the change set with per-stage timings, and writes the ledger row when
`record` is true.

### Verify it, and keep the receipt

`deploy/verify.py` is the step that turns "the scripts are complete" into "this
ran on AWS". It uploads two real frames to the bucket, invokes the function,
times the whole round trip from outside, reads the ledger row back out of
DynamoDB, pulls the tail of the CloudWatch log, then drives the **public**
console API end to end — `GET /api/health`, `POST /api/sessions` with a baseline
data URL, `POST /api/sessions/<id>/rescan` — and writes all of it to a JSON
report. It exits non-zero if the function is not arm64, if the scan reported no
events for a scene with known changes, if `record` was sent and no row appeared,
or if the public API did not complete a scan.

```bash
python deploy/verify.py --region us-east-1 \
  --bucket strata-sessions-<account> --record \
  --out deploy/verify-report.json

# or with photographs instead of the seeded scene:
python deploy/verify.py --region us-east-1 \
  --bucket strata-sessions-<account> --frames path/to/photos --record
```

This has been run. `deploy/verify-report.json` is the committed receipt, verdict
`PASS`, on account `478726716377`, region `us-east-1`, 2026-10-20:

| What it proves | Measured |
|---|---|
| Architecture | `arm64` (Graviton), `python3.12`, Zip package, 3008 MB |
| Which vision stack ran there | `neural-lightglue` — ALIKED + LightGlue ONNX, staged from S3 into `/tmp/models` |
| Registration | homography, 1174 inliers of 1210 matches, confidence 0.9231 |
| Change set | 4 events; `added`, `missing`, `state_changed` |
| Latency, whole round trip from this machine | 9156 ms (in-function total 8779 ms) |
| Stage split | match 4941 ms, extract 1264 + 840 ms, coverage 890 ms, diff 121 ms, register 3.9 ms |
| Ledger | row read back from DynamoDB: `lambda/ledger/verify-bay-4` / `20261020-012803-…json` |
| Public API | session created in 1834 ms, rescan in 11209 ms, 4 events, `storage: s3` |
| Transport actually used | `lambda_invoke` — the Function URL returned 403, recorded in `function_url_error` |

The two functions are the same code as the container, from the same package, on
the same arm64 architecture; the difference is the packaging format, and it is
recorded here rather than smoothed over: a container can pin the system
libraries around OpenCV and can pre-bake the weights, a zip cannot, so the zip
carries the headless wheel and fetches the weights at cold start.

### Why Arm

The dominant cost in the pipeline is image processing, not inference: resize,
warp, adaptive threshold and contour extraction run on every frame pair. Those
are precisely the operations OpenCV 5 accelerates on AArch64 through the
**KleidiCV** HAL (NEON / SVE / SME), and the COOL distribution adds further
tuning on top. The workload also parallelises across frames, which suits a
many-core Graviton.

Development and the interactive console run on x86; the vision workload executes
on the Arm path. That hybrid is deliberate and is documented rather than
presented as a single architecture.

### Teardown

```bash
./deploy/teardown.sh          # asks for the bucket name to confirm
./deploy/teardown.sh --yes
```

Deletes by exact name, never by tag or prefix, so a mistyped variable cannot take
unrelated resources with it.

---

## 3. COOL benchmark — reproducible protocol

The claim under test is that the vision workload runs measurably faster on
Graviton with the Cloud-Optimized OpenCV build than on a comparable x86 instance
with upstream OpenCV 5. The protocol exists so the claim can be checked rather
than believed.

### 3.1 Baseline (x86, upstream OpenCV 5)

```bash
python deploy/benchmark/bench_core.py \
  --json deploy/benchmark/results/x86-baseline.json \
  --repeat 3 --instance c7i.xlarge
```

### 3.2 Target (Graviton4, COOL)

1. Subscribe to **Cloud Optimized OpenCV For AWS Graviton4** on AWS Marketplace.
2. Note the AMI id for your region.
3. Run:

```bash
COOL_AMI_ID=ami-xxxxxxxx \
STRATA_BUCKET=strata-sessions-<account> \
./deploy/benchmark/run_on_graviton.sh
```

The script packages the benchmark, launches `c8g.xlarge`, and the instance
benchmarks *itself* from user-data: it activates the COOL virtualenv, verifies
which `cv2` it loaded, runs the benchmark, uploads JSON to S3, and shuts down.
No SSH key is required and nothing is left running.

`STRATA_BENCH_INSTANCE` selects a different class; `STRATA_BENCH_REPEAT` sets
repetitions.

### 3.3 Comparison

```bash
python deploy/benchmark/compare.py \
  deploy/benchmark/results/x86-baseline.json \
  deploy/benchmark/results/arm-c8g.xlarge.json
```

### 3.4 What is measured, and why

| Group | Operations | Why these |
|---|---|---|
| Microbenchmarks | resize (up and down), warpPerspective, perspectiveTransform, findContours (with the morphology the pipeline runs before it), Sobel gradient, cvtColor to grey and to HSV, GaussianBlur, absdiff, morphology | These are the kernels the pipeline's own hot path calls, so isolating them shows *where* a difference comes from rather than only that one exists |
| Reference op | adaptiveThreshold | Measured and labelled as **not in this pipeline's path** — the difference is thresholded by a NumPy robust calibration, not by this call. It is reported anyway because a cloud-optimised build names it among its accelerated kernels, and a null result is still a result |
| Full scan | `run_scan` on a fixed seeded pair | The number a user experiences. This is the one to quote |

Every timed section is preceded by a warm-up, and both median and p95 are
reported. A mean hides the tail, and the tail is what an operator feels.

### 3.5 Reporting rules

- Quote the **full-scan median**, not the best microbenchmark.
- Report the **resolved backend** — a run that silently fell back to SIFT is not
  comparable to one that used ALIKED, and the harness records which ran.
- Report the instance class, region, and `cv2.__file__`, because "Graviton" is a
  family, not a machine.
- State the COOL version. The Marketplace listing's current release is **COOL
  3.1, built on OpenCV 5.0**.
- A speed-up is only meaningful against a stated baseline. Never report a ratio
  without its denominator.

### 3.6 Cost

At `c8g.xlarge` on-demand rates, a benchmark run of a few minutes costs cents of
compute. The COOL software rate is metered hourly (roughly $0.01–0.04/hr
depending on size) and the listing includes a **7-day free trial**, which covers
the measurement window. The instance terminates itself when it finishes.

---

## 4. Observability

Three things emit, and each is checked by a test rather than asserted here:

- **Structured logs.** `service/observability.py` writes one JSON object per
  line to stdout, and the service logs one per completed scan with its
  per-stage timings. `STRATA_LOG_FORMAT=text` switches to a readable format for
  a terminal; JSON is the default because a log that has to be parsed by eye is
  not a structured log. Secret-shaped fields (`api_key`, `authorization`,
  `token`, …) are dropped to `[redacted]` — value as well as key — at the
  formatter, so a mis-placed field cannot leak a model key into a log explorer.
- **The Lambda writes the same line.** `deploy/handler.py:log_line()` prints one
  JSON record per invocation to stdout, which is how CloudWatch receives it.
- **CloudWatch metrics.** `metric_datapoints()` turns the timings into a
  `ScanDuration`, a `ChangeEvents` count and one `Stage.<name>` metric per
  stage. Emission is opt-in with `STRATA_EMIT_METRICS=1`, because a metric costs
  money per data point and a grant account should not learn that from an
  invoice. The IAM policy's `cloudwatch:PutMetricData` grant now has a caller.
- **Agent trace** persisted with every run: what was observed, decided, why, by
  whom, and how long each step took.
- **Benchmark JSON** retained per run, so a performance regression is a diff
  rather than a memory.

A production deployment therefore has an answer to "why did this scan take forty
seconds" that is a query, not a reproduction.

---

## 5. Security

- No secrets in the repository; configuration is environment-only, with
  `.env.example` as the only committed template.
- The Lambda role is scoped to one bucket prefix and one table — not
  `AmazonS3FullAccess`.
- The S3 bucket blocks all public access and encrypts at rest.
- Store keys are validated before use, so a session identifier cannot escape its
  prefix.
- Uploads are size- and type-bounded at the API boundary.
- Instance metadata is locked to IMDSv2 (`HttpTokens=required`).

---

## 6. What has been verified, and what has not

Stated plainly, because a deployment document that overstates its coverage is
worse than none.

**Verified in this build**

- The full test suite passes on the locked dependency set: **402 fast tests and
  79 end-to-end** (481 total), including the Lambda handler's event contract and
  its non-finite-number case, the managed-storage contract, the structured-log
  formatter and its redaction, and the documentation-vs-artefact agreement tests.
  `.github/workflows/ci.yml` runs all of it on Python 3.13 and 3.12 against the
  locked set and on 3.11 against the declared ranges, re-runs the 24-case
  benchmark and gates it against floors, and builds the console from source.
- The service runs locally and serves the console and API on one port.
- The evaluation reproduces its published figures exactly (0.665 px, P 0.9375,
  R 0.8542, F1 0.8875, 0 still-pair false positives).
- The benchmark harness runs and produces comparable JSON on x86.

**What needs an AWS account, and what has one**

Run against a live account in this build — receipt committed at
`deploy/verify-report.json`, verdict `PASS`:

- `deploy/deploy.py`: bucket, table, role, both arm64 zip functions, the HTTP
  API, and a synchronous invoke that returns the function's own health document
  before the endpoint is created.
- `deploy/verify.py`: a real scan on Graviton, the ledger row read back out of
  DynamoDB, the CloudWatch tail, and a session-plus-rescan through the public
  API. §2 quotes the numbers.

Not run, and not claimed:

- `deploy/deploy.sh` — the container route. It needs Docker and this machine has
  none, so the image has never been built or pushed to ECR here. The zip route
  ships the same handler, the same pinned versions and the same arm64
  architecture; what it does not share is the image's ability to pin system
  libraries and pre-bake the weights.
- `deploy/benchmark/run_on_graviton.sh` — needs an EC2 instance and the
  Marketplace COOL AMI (§3.2).
- Consequently **no Arm-vs-x86 ratio is published here**. The harness produces
  both sides of it; publishing a number that was not measured would be the one
  thing this project is built not to do.

What *is* verified without an account: the handler's event contract and its
ledger row shape, the managed store's behaviour against a fake that validates
the way DynamoDB does (`tests/test_store_aws.py`), and the fact that the key
schema and IAM actions in `deploy.sh` are the ones the code actually uses —
which is how the schema bug in §2 got fixed before anything was deployed.

To reproduce the executed half: `python deploy/deploy.py --dry-run` to read the
plan, `python deploy/deploy.py` to build and deploy, then
`python deploy/verify.py --bucket <bucket> --record` to write
`deploy/verify-report.json`. To close the remaining gap, §3.2 and §3.3 for the
Arm comparison.
