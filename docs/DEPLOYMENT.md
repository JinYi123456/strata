# Deployment

STRATA runs in two shapes from the same code:

- **Local** — one Python process serving the API and the built console on one port.
- **AWS** — the vision workload as an arm64 Lambda container on Graviton, with S3,
  DynamoDB and CloudWatch behind it, plus a reproducible benchmark on Graviton4
  with the Cloud-Optimized OpenCV build.

The vision core is a pure function of its inputs, so nothing is rewritten between
the two. The only difference is configuration.

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

### The console on Vercel

The static console is deployed separately at
<https://strata-console-rouge.vercel.app>, built from `web/` on Vercel's own
Next.js pipeline. That is the split §1.1 describes: the interface on a CDN, the
vision workload on AWS.

**It is not a working demo on its own, and is not offered as one.** The console
calls `/api/...` on its own origin unless `NEXT_PUBLIC_API_BASE` is set at build
time, so on Vercel it reaches the interface, lists nothing, and reports that the
service is unreachable. Point it at the deployed AWS API and rebuild, and the
same URL becomes the live endpoint. The working demonstration is the container
above, or the film, until that rebuild happens.

The first deploy attempt was rejected by Vercel's dependency gate: `next@15.1.6`
pulls a `postcss` with published advisories, and the platform blocks the
deployment rather than warning. `web/` is on `next@16.4.2` now — `npm audit`
reports zero vulnerabilities, `tsc --noEmit` is clean, and the console was
driven end to end in a browser on the new version: five sample sites, a real
rescan producing four typed events, coverage guidance, the answer-key panel, and
the approval gate recording `Recorded · scan #0 · finalized` in the history.

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
| Function URL | — | Public invocation endpoint |

The `pk`/`sk` schema is the storage contract both writers use: `pk` is the
parent of a record key, `sk` its leaf. `service/store.py` and
`deploy/handler.py` write it, and `tests/contract.py` reads it back out of
`deploy.sh` so the three cannot drift apart silently — which is exactly how this
table was wrong before anything had ever been deployed.

`./deploy/deploy.sh --dry-run` prints every command it would run and touches
nothing, including the account lookup. It is the way to read the plan without
credentials.

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
ran on AWS". It uploads two real frames to the bucket, invokes the function over
its URL, times the whole round trip from outside, reads the ledger row back out
of DynamoDB, pulls the tail of the CloudWatch log, and writes all of it to a
JSON report. It exits non-zero if the function is not arm64, if the scan
reported no events for a scene with known changes, or if `record` was sent and
no row appeared.

```bash
python deploy/verify.py --region us-east-1 \
  --bucket strata-sessions-<account> --record \
  --out deploy/verify-report.json

# or with photographs instead of the seeded scene:
python deploy/verify.py --region us-east-1 \
  --bucket strata-sessions-<account> --frames path/to/photos --record
```

The report is a committed artefact once it exists, and it is the first thing a
judge should ask for.

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

- The full test suite passes on the locked dependency set: **389 fast tests and
  79 end-to-end** (468 total), including the Lambda handler's event contract and
  its non-finite-number case, the managed-storage contract, the structured-log
  formatter and its redaction, and the documentation-vs-artefact agreement tests.
  `.github/workflows/ci.yml` runs all of it on Python 3.13 and 3.11, re-runs the
  24-case benchmark and gates it, and builds the console from source.
- The service runs locally and serves the console and API on one port.
- The evaluation reproduces its published figures exactly (0.665 px, P 0.9375,
  R 0.8542, F1 0.8875, 0 still-pair false positives).
- The benchmark harness runs and produces comparable JSON on x86.

**Requires an AWS account to execute — and this is the honest gap**

- `deploy/deploy.sh`, `deploy/verify.py` and
  `deploy/benchmark/run_on_graviton.sh` are complete and idempotent, but have
  not been run against a live account in this build. They require: ECR push,
  Lambda create/update, S3, DynamoDB, IAM, EC2, and a Marketplace subscription
  for the COOL AMI.
- Consequently **no Arm-vs-x86 ratio is published here**, and nothing in this
  repository yet demonstrates a scan that executed on AWS. The harness produces
  both; publishing a number that was not measured would be the one thing this
  project is built not to do.
- What *is* verified without an account: the handler's event contract and its
  ledger row shape, the managed store's behaviour against a fake that validates
  the way DynamoDB does (`tests/test_store_aws.py`), and the fact that the key
  schema and IAM actions in `deploy.sh` are the ones the code actually uses.

To close the gap: `./deploy/deploy.sh --dry-run` to read the plan, then
`./deploy/deploy.sh`, then `python deploy/verify.py --record` to produce
`deploy/verify-report.json` — architecture, image digest, latency, change set,
the ledger row read back, and the CloudWatch tail. Then §3.2 and §3.3 for the
Arm comparison. Each step is minutes of work and cents of compute; none of it
has been done, and the submission should be read with that in mind.
