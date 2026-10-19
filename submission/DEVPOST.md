# Devpost submission packet — STRATA

Every field of the submission form, filled. Paste-ready.

**Status of the three things only a human account can supply:**

| Field | Blocked on |
|---|---|
| Video URL | YouTube upload of the committed film, `web/public/media/strata-demo.mp4` (see §6) — unlisted is fine and required for judge access |
| Demo URL | the AWS deployment (`deploy/deploy.sh` → `deploy/verify.py`), then `NEXT_PUBLIC_API_BASE` rebuild of the Vercel console |
| Submit | your account, and the final click |

Everything else below can be pasted as it stands.

---

## 1. Identity

**Project title** (≤ 60 characters)

```
STRATA — Version Control for the Physical World
```

**Tagline** (≤ 140 characters)

```
OpenCV 5 change detection between two visits of a place: typed events, calibrated confidence, coverage of what it did not see. Human-gated.
```

**Project tags** (Devpost allows a limited set; these are the four that map to the competition's own categories)

```
opencv
aws
computer-vision
serverless
```

---

## 2. Links

| Field | Value | Ready? |
|---|---|---|
| Source code | `https://github.com/JinYi123456/strata` | yes — public, default branch `master` |
| Technical report | `https://github.com/JinYi123456/strata/blob/master/docs/TECHNICAL_REPORT.md` | yes |
| Review guide | `https://github.com/JinYi123456/strata/blob/master/submission/README.md` | yes |
| Architecture diagram | `https://github.com/JinYi123456/strata/blob/master/docs/architecture.svg` | yes |
| Demo video | *YouTube unlisted URL — to be pasted* | **no** |
| Demo / live URL | *AWS Function URL or container URL — to be pasted* | **no** |
| Console (static) | `https://strata-console-rouge.vercel.app` | yes, but see the note |

**Do not submit the Vercel URL as the live endpoint.** The console calls `/api` on
its own origin; until it is rebuilt with `NEXT_PUBLIC_API_BASE` pointed at the AWS
deployment it shows the interface and reports the service unreachable. Either
finish that rebuild or leave the field to the AWS URL. A judge clicking a link
that errors costs more than a link that is absent.

---

## 3. Description

Devpost renders this as Markdown. Paste the block below verbatim.

```markdown
## What it does

Photograph a place. Photograph it again later. STRATA tells you **what changed**,
in the sense an engineer would accept: which object left, which arrived, which
moved, which was altered — each with a location, a measurement, a confidence, and
the reasoning that produced it.

Two visits of the same place are never the same photograph. The camera moves, the
light shifts, the exposure drifts. Subtract them and you get the difference
between two cameras, not the difference between two moments. So the problem is
solved in the order it has to be solved: register, normalise the photometry,
detect what genuinely differs, then resolve regions of difference into typed
events.

It then does the part most systems skip. It reports **what it did not see**.
Regions it could not observe well enough to compare are marked unresolved and
turned into a specific request for another capture, rather than folded into a
confident "nothing changed". Nothing enters a site's history until a person
approves it.

The result has the properties engineers already trust in a repository: an
immutable baseline, a diff against it, typed changes, provenance for every claim,
and an append-only history that includes the scans that found nothing.

## OpenCV 5

The vision core is written against OpenCV **5.0.0** (`opencv-contrib-python==5.0.0.93`)
and uses APIs that do not exist in 4.x:

- **`cv::ALIKED`** — the CNN keypoint detector and descriptor introduced in
  OpenCV 5, run at up to 2000 keypoints with 128-dimensional descriptors.
- **`cv::LightGlueMatcher`** — attention-based matching with a learned confidence
  per correspondence, given spatial context through `setPairInfo()` before
  `match()`.
- **`cv2.USAC_MAGSAC`** — robust estimation under OpenCV 5's USAC framework.
  Registration lands at **0.665 px** mean corner error across the benchmark.
- The classical path (`SIFT` + FLANN) is a working fallback, not a stub: with no
  weights on disk the whole product still runs, and every result reports which
  matcher actually executed.

Two guards exist because of measured failures rather than intuition. LightGlue
divides by zero on a featureless image, and on Windows that is a **fatal C-level
exception** — no Python handler runs, and the service process dies with it. A
single frame of a blank wall would take a deployment down. The condition is
invisible in the features (ALIKED returns 2000 well-spread keypoints with normal
descriptors on a uniform frame), so the guard tests the *image*, using mean
absolute Laplacian as a structure energy, and it sits **before** the call because
`try/except` cannot survive a failure that never unwinds the stack. Measured
separation: uniform grey 0.0000, a single small square 0.43, faint sensor noise
5.03, a real scene 15.6–20.2.

## The agent

A five-phase loop — observe → decide → act → gate → finish — where the vision
output changes what happens next. From a live run, verbatim from the persisted
trace:

```
[observe] scan            by=pipeline      "4 changes; coverage 30%; registration confidence 0.92"
[decide ] request_view    by=policy-rules  "Only 30% of the frame is well observed, against a 60% bar"
[act    ] request_view    by=loop          "Requested a further capture, but none was available"
[decide ] mark_uncertain  by=policy-rules  "…rather than accept a scan that implies the unseen areas were unchanged"
[decide ] finalize        by=policy-rules  "Committing this as the canonical record — which requires your approval"
[gate   ] approval        by=human
```

The decision layer is a **deterministic policy by default**. A model may be
consulted at exactly one call site, to justify *why* another view is wanted; if
it fails for any reason the run degrades to the rules with the error recorded in
the trace. Nothing in the product requires a model to be reachable — which is why
the demo runs with no configuration and no API key at all.

A requested capture changes what the system observes: each further frame of the
second visit is matched, registered with the same estimator, warped into the
canonical frame, and credited to coverage — and a frame that will not register is
reported as attempted rather than counted. Coverage accumulates as a union, not a
mean, so a second look can only ever add evidence.

## AWS

The vision workload targets **AWS Graviton (arm64)**: a Lambda container image
built with `docker buildx --platform linux/arm64`, S3 for session objects,
DynamoDB for the append-only ledger, CloudWatch for logs and per-stage metrics.
The hybrid split is stated rather than hidden — development and the interactive
console on x86, the vision workload on the Arm path — and the reason is technical,
not decorative: the dominant cost is image processing (resize, warp, Sobel
gradients, morphology, contour extraction), which is what OpenCV 5 accelerates on
AArch64 through the KleidiCV HAL, and the workload is embarrassingly parallel
across frame pairs.

`deploy/deploy.sh` is idempotent and has a `--dry-run` that prints every command
it would run without touching an account. `deploy/verify.py` drives one real scan
through the deployment and writes down the receipt: architecture, image digest,
latency measured from outside, the change set, the ledger row read back out of
DynamoDB, and the CloudWatch tail.

**Honest status:** the deployment scripts have not been run against a live AWS
account in this build, so no Arm-versus-x86 ratio is published. The harness that
produces it is committed and the protocol is documented; publishing a number that
was not measured would contradict the point of the project.

## Evaluation

24 synthetic scenes at 960×720 generated from seeds, each with known mutations and
a known camera pose, plus 12 still pairs where nothing changed. Every figure below
is regenerated from the committed artefact by `scripts/figures.py`, and a test
fails if a document stops matching the run.

| Metric | Result |
|---|---|
| Registration error | **0.665 px** mean corner error |
| Registration confidence | 0.9008 |
| Precision | **0.9375** (macro; micro 0.9318) |
| Recall | **0.8542** (micro 0.8542) |
| F1 | **0.8875** (micro 0.8913) |
| Event-kind accuracy | 0.6979 |
| **Confident false positives on unchanged pairs** | **0.0** per pair, 12 pairs |

A reported box counts as the same change as an expected one at IoU ≥ 0.25.

The still-pair number is the one that decides whether the system is usable. A
detector that finds things is not hard; one that reliably finds *nothing* when
nothing changed is what makes it deployable, and it is the number most systems do
not report.

**Reproduced independently.** Installing `requirements.lock` in a clean
environment and running `python scripts/evaluate.py --cases 24` reproduces the
committed artefact **case for case, aggregate and still-pair block identical**
(392.2 s against the recorded 387.5 s). The suite — 468 tests, 79 of which run the
real pipeline end to end — passes on the locked set, and CI re-runs the benchmark
and gates it against floors on every push.

## What does not work

- **Event labels are weaker than event locations.** Kind accuracy is 0.6979
  against F1 0.8875. The two are reported separately rather than blended.
- **Ten of 24 cases merge two nearby changes into one region.** The change is not
  lost; it is not split. Documented and pinned by a test that fails if the
  behaviour changes.
- **One seed fragments a move destination** into two regions, costing precision on
  that case, because the polarity guard correctly refuses to fuse opposite-polarity
  evidence but splits a genuine arrival whose fragments disagree locally.
- **The second geometric model is not exercised by the benchmark.**
  `prefer_planar` ships `True`, so all 24 cases registered as homographies and no
  model-comparison note was recorded. The fundamental-matrix branch is real and is
  covered by tests that turn it on; the benchmark contains no scene with enough
  depth structure to need it.
- **No Arm-versus-x86 measurement**, and no scan yet demonstrated on AWS.
- **The evaluation data is synthetic.** Real captures can be uploaded and go
  through the same code path, but they carry no answer key, so no real-world
  accuracy is claimed.
- **No metric scale.** Output is in canonical pixels: it says an object moved and
  roughly how far, not how many centimetres.

## Responsible use

STRATA observes physical spaces, so consent and data handling are part of the
design rather than a disclaimer. It has **no face model, no person
identification, no biometric template and no behavioural inference** — these are
structural properties, not policy promises: the schema has nowhere to put a
person-level signal. Capture guidance is an instruction to a human holding a
camera, not autonomous acquisition. Every write is human-gated and attributed
(`decision_made_by: human`), and the history is append-only, so the record of what
was observed is itself observable. Uncertainty is a first-class output: "I could
not see this well enough" is preferred over a confident guess, because a confident
negative over ground nobody photographed is actionable and wrong.

## Run it

```bash
git clone https://github.com/JinYi123456/strata.git
cd strata
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/serve.py            # → http://127.0.0.1:8000
```

No configuration, no API keys, no model download to start: five sample sites are
generated on first run and go through the same pipeline as an uploaded capture.

```bash
pip install -r requirements.lock          # the exact measured set (Python 3.12+)
python scripts/fetch_models.py            # ~52 MB, digest-verified
python scripts/evaluate.py --cases 24     # regenerate every published figure
pytest                                    # 389 fast tests
pytest -m slow                            # 79 end-to-end tests
bash scripts/verify_ui.sh                 # drive the live app in a real browser
```
```

---

## 4. Gallery images and captions

Eight images, all generated by `python scripts/make_submission_images.py` at the
ratios Devpost renders (3:2 gallery, 16:9 thumbnail), each under 1 MB against the
5 MB limit. Upload in this order; paste each caption into the image's caption
field if the form offers one.

| # | File | Caption |
|---|---|---|
| 1 | `submission/thumbnail.png` | *(project thumbnail, 1280×720)* |
| 2 | `submission/gallery/01-sites.png` | The operator console. Five sample sites are generated on first run and go through the same pipeline as an uploaded capture. Archive Row 7 is shipped as a deliberately hard case. |
| 3 | `submission/gallery/02-scan-result.png` | A rescan in progress. The pipeline registers the second visit into the first, then normalises, differences and resolves — with per-stage timings recorded on every run. |
| 4 | `submission/gallery/03-compare.png` | The aligned comparison. Both visits are in one canonical frame, so the slider compares one place at two moments rather than two photographs. |
| 5 | `submission/gallery/04-changes.png` | Four typed changes: what departed, what arrived, what moved 211 px, what was altered. Each carries its measurement, its confidence and an evidence trail. The 47% one is labelled 47%. |
| 6 | `submission/gallery/05-coverage.png` | What the system did not see. Only 30% of the frame was observed well enough to compare, so the unresolved regions become three specific capture requests instead of a confident "nothing changed". |
| 7 | `submission/gallery/06-gate.png` | The approval gate. A scan prepares a change set and stops; nothing is written to the site's history until a person answers, and the entry records who decided. |
| 8 | `submission/gallery/07-architecture.png` | The architecture: console, HTTP API, agent loop and the OpenCV 5 vision core, with the managed storage path on AWS Graviton and the COOL benchmark alongside it. |

---

## 5. Prize tracks

**Agentic Vision Award — enter.** The qualifying test is that vision results
change what the system does next, and that is demonstrated above from a persisted
trace: coverage 30% and registration confidence 0.92 select `request_view`, then
`mark_uncertain`, then `finalize`, and the human gate is a real state transition
rather than a modal. The agent layer is documented in §5 of the technical report
and rendered as a readable timeline in the interface.

**Best Use of COOL Award — do not enter yet.** The eligibility line is explicit:
COOL must execute the claimed core workload on Graviton or the Arm component of a
documented hybrid architecture. Nothing has run on Graviton in this build. The
harness, the AMI launch script and the measurement protocol are committed, and
`docs/DEPLOYMENT.md` §3 is the whole procedure — but entering this track without a
measured Arm result would be a claim the repository does not support. **Enter it
after** `run_on_graviton.sh` has produced `arm-c8g.xlarge.json` and
`compare.py` has produced a ratio.

---

## 6. Before submitting — the checklist

Requirements taken verbatim from `opencv26.devpost.com`, and where each stands.

| Requirement | Status | Where |
|---|---|---|
| Uses OpenCV 5 for substantive image/video analysis | **done** | ALIKED, LightGlueMatcher, USAC_MAGSAC on 5.0.0; `vision/strata_core/` |
| A meaningful component runs on AWS | **open** | scripts complete, never executed — `deploy/` |
| Technical report: problem, users, architecture, OpenCV 5 implementation, AWS deployment, evaluation, limitations, responsible use | **done** | `docs/TECHNICAL_REPORT.md` |
| Public or judge-accessible code repository | **done** | github.com/JinYi123456/strata (public) |
| Pinned dependencies | **done** | `requirements.txt` ranges + `requirements.lock` measured set |
| Clear build, deployment and test instructions | **done** | `README.md`, `docs/DEPLOYMENT.md`, CI workflow |
| Architecture diagram showing OpenCV 5 and AWS components, plus COOL/agent where relevant | **done** | `docs/architecture.svg` / `.png` |
| Working web endpoint **or** arranged live screen-share | **open** | local run verified in a browser; public endpoint pending AWS |
| Video ≤ 5 min, judge-accessible, showing **the team**, the application working, **its architecture**, and its principal results | **film done, upload pending** | 4:42 film; needs the YouTube unlisted URL |
| Evaluation evidence including failure cases or limitations | **done** | report §7.3, §8, `docs/evaluation.json` |
| Entrant age / eligibility, no manipulation of judging or voting | **you** | account-level attestations |
| Team representative designated | **you** | solo entry — you |
| "Based on previous work" / outside-help disclosures | **you** | answer honestly; STRATA was built in this repository during the build phase |

**Also worth doing before the deadline, in order of what they buy:**

1. `./deploy/deploy.sh` then `python deploy/verify.py --record` — this is the
   difference between meeting the AWS requirement and describing it.
2. `COOL_AMI_ID=… ./deploy/benchmark/run_on_graviton.sh` then `compare.py` — turns
   the COOL track from ineligible into entered, and gives the report its missing
   number.
3. Rebuild the Vercel console with `NEXT_PUBLIC_API_BASE` = the AWS URL, so the
   submission has a live endpoint a judge can click.
4. Two photographs of one shelf, ten minutes apart, with one object moved —
   uploaded through the app, and the result added to the report and the film. It
   is the only evidence that would answer "does this survive the real world",
   which is where the impact score is currently losing points.
