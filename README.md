# STRATA

**Version control for the physical world.**

Photograph a place once. Photograph it again later. Get a typed, confidence-scored
account of what changed — with the measurements behind every claim, and the
reasoning that produced it, shown alongside the answer.

**[Live app →](https://f3ba7914438c401ca724dd1522121433.sg2.agentos-app.run)** — opens on
five sample sites that need nothing uploaded.

### Documentation

| | |
|---|---|
| [Technical report](docs/TECHNICAL_REPORT.md) | Problem, architecture, OpenCV 5 implementation, evaluation, limitations |
| [Architecture](docs/architecture.svg) | The four layers and where the vision workload runs |
| [Deployment](docs/DEPLOYMENT.md) | Local, AWS Graviton, and the reproducible benchmark protocol |
| [Responsible use](docs/RESPONSIBLE_USE.md) | What the system does not do, and why |

---

## The problem

Comparing two photographs of the same place is easy for a person and surprisingly
hard for a machine. Two visits are never taken from the same spot, at the same
angle, under the same light. So the naive approach — subtract the images — reports
a difference everywhere and tells you nothing.

Doing it properly means solving four problems in order, and getting each one wrong
makes the next meaningless:

1. **Registration.** Find the transform that puts the second visit into the first
   visit's frame. Without this, every later step is comparing misaligned pixels.
2. **Photometric normalisation.** Absorb the exposure and white-balance drift
   between visits, so a change in lighting is not reported as a change in the world.
3. **Change detection.** Find what actually differs, at a threshold calibrated to
   *this* pair of images rather than a fixed number.
4. **Event resolution.** Turn regions of difference into typed events — something
   departed, something arrived, something moved, something was altered — and say
   how confident you are in each.

STRATA does all four, and then does the part most systems skip: it **reports what
it does not know**. Regions it could not observe well enough are marked
unresolved rather than reported as unchanged, and nothing is written into a site's
history until a person approves it.

---

## What it does

```
record a site  ──►  re-scan it later  ──►  review the change set  ──►  record it
   baseline            agent loop              evidence shown         human gate
```

- **Baseline** — one or more frames of a site, fixed as the reference. Immutable,
  because "what changed since?" needs a definite answer.
- **Rescan** — a second visit, registered against the baseline into a shared
  canonical frame.
- **Change set** — typed events with confidence, location, and the measurements
  behind them.
- **Guidance** — when coverage is too thin to compare part of the scene, the
  system asks for a specific extra capture rather than guessing.
- **Ledger** — an append-only history of what was recorded, including the scans
  that found nothing and the ones a person declined.

---

## Quick start

Requires Python 3.11+.

```bash
# 1. Environment
python -m venv .venv
.venv/Scripts/activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt

# 2. Model weights (optional, ~52 MB, digest-verified)
python scripts/fetch_models.py

# 3. Run it
python scripts/serve.py
# → http://127.0.0.1:8000
```

That is the whole setup. The service bundles a prebuilt console, so no Node
toolchain is needed to run it — the interface is served from `service/static/`
alongside the API on one port. To rebuild the console after changing anything
under `web/`, run `bash scripts/build_console.sh`.

The app opens on a set of sample sites that need nothing uploaded, so the whole
pipeline is exercisable in one click. Sites you record yourself go through exactly
the same code path.

### For development

```bash
pip install -r requirements-dev.txt   # adds the test client and boto3
pytest                                # fast unit tests
pytest -m slow                        # end-to-end pipeline, agent and service
bash scripts/verify_ui.sh             # drives the live app in a real browser
```

### Reproducing the numbers exactly

`requirements.txt` declares ranges, so the project installs on a wider range of
interpreters. To reproduce the evaluation figures exactly, install the set they
were measured on:

```bash
pip install -r requirements.lock
```

That is the environment the table above came from, on Python 3.13.12.

### Without the model weights

The feature backend is pluggable. If the ONNX weights are absent, the pipeline
falls back to SIFT + FLANN and keeps working — lower accuracy, no download. The
reported backend always says which one ran, so a number is never attributed to the
wrong matcher.

---

## How it works

### The pipeline

`vision/strata_core/pipeline.py` — pure: arrays in, arrays plus JSON out. No
filesystem, no network, no model download at call time, which is what lets it run
unchanged inside a container, a serverless function, or the local service.

| Stage | Module | What it does |
|---|---|---|
| Features | `features.py` | ALIKED + LightGlue, with a SIFT/ORB fallback and a guard against featureless frames |
| Geometry | `geometry.py` | USAC/MAGSAC homography, with a fundamental-matrix model for scenes with real depth |
| Diff | `diff.py` | Two-channel change detection (luminance **and** chrominance) with per-pair threshold calibration |
| Events | `events.py` | Regions → typed events, including separating a translation from a removal plus an insertion |
| Coverage | `coverage.py` | Per-cell observation map and the resulting capture guidance |
| Agent | `strata_agent/` | The perceive → decide → act loop, with a human approval gate |

### Two things worth knowing about the change detection

**Luminance alone is not enough.** Luminance is a weighted sum of the channels, so
a recoloured object can differ by 78 in colour and only 18.5 grey levels — and
local-contrast normalisation flattens most of that. No threshold recovers what the
conversion discarded, so a second, independent chrominance channel (opponent colour
differences, which additive exposure drift cancels exactly) runs alongside it.

**A translation looks exactly like a removal.** When an object moves, the place it
left differs in a way that is pixel-identical to the object being taken away, and
the place it arrived differs like a new object appearing. The two are separable
only by noticing that they occur *together*, which is what the pairing pass does.

### The agent

The decision layer is a **deterministic policy by default**. A model can be
consulted at exactly one call site, and if it fails — no key, rate limit, bad JSON —
the run degrades to the rules with the error recorded. Nothing in the product
requires a model to be reachable, which is why the demo works with no
configuration at all.

Every run produces a trace: what it observed, what it decided, why, who decided
(rules, model, or human), and how long each step took.

The approval gate is real. A scan prepares a change set and stops; its outcome is
`declined` because nothing was written. The ledger is appended only when a person
answers the gate.

---

## Evaluation

```bash
python scripts/evaluate.py --cases 24 --out artifacts/eval
```

24 synthetic scenes at 960×720, each with known mutations and a known camera pose,
plus 12 still pairs where nothing changed. Every scene is generated from a seed, so
a run is reproducible to the pixel.

| Metric | Result |
|---|---|
| Registration error | **0.665 px** mean corner error |
| Registration confidence | 0.901 |
| Precision | **0.938** |
| Recall | **0.854** |
| F1 | **0.888** |
| Event-kind accuracy | 0.698 |
| **False positives on still pairs** | **0.0** |

Nine of the 24 cases score a perfect F1 = 1.000 with kind = 1.000.

**The still-pair number is the one that matters.** A change detector that finds
things is not hard; one that reliably finds *nothing* when nothing changed is what
makes it usable, and it is the number most systems do not report.

### Honest failure cases

The evaluation deliberately includes the cases this system does not handle well,
because a benchmark of only successes measures the benchmark:

- **Adjacent changes can merge.** When two changes sit close enough that the
  difference mask forms one connected component, the system reports one region
  where there were two. The change is not lost; it is simply not split. This is
  pinned by a test that fails if the behaviour ever changes.
- **Low-structure pairs cannot be aligned.** Two frames of a blank wall have
  provably nothing to correspond. Rather than inventing an alignment, the system
  refuses and says so.
- **Event-kind accuracy is the weakest metric at 0.698.** Locations are found far
  more reliably than they are correctly labelled, and the two numbers are reported
  separately rather than blended into one flattering figure.

---

## Deployment

The service serves the built console from the same origin as the API, so there is
one artefact and one URL rather than two that have to be kept in step.

```bash
PORT=8000 python scripts/serve.py
```

With `PORT` set, the service binds `0.0.0.0` automatically — a service bound to
loopback is unreachable from outside its own container, which is a silent and
total failure on any hosting platform.

Configuration is environment-only:

| Variable | Purpose |
|---|---|
| `PORT` / `STRATA_PORT` | Port to listen on |
| `STRATA_HOST` | Bind address (default: `0.0.0.0` when `PORT` is set, else loopback) |
| `STRATA_EXECUTOR` | `local` (default) or `aws` |
| `STRATA_DATA_DIR` | Where the local store writes |
| `STRATA_S3_BUCKET` / `STRATA_DDB_TABLE` / `STRATA_AWS_REGION` | Managed storage |
| `STRATA_CORS_ORIGINS` | Allowed origins |
| `STRATA_STATIC_DIR` | Built console to serve |

Switching from local disk to S3 + DynamoDB is a configuration change, not a code
change: `service/store.py` defines one interface with two implementations.

### Repository layout

```
vision/
  strata_core/       the pipeline: features, geometry, diff, events, coverage
  strata_agent/      the perception loop: policy, actions, trace, brain
  models/            ONNX weights (fetched, not committed)
service/             the HTTP API, session lifecycle, and the bundled console
web/                 the console source (TypeScript, Next.js, Tailwind)
scripts/             evaluation, model fetch, serving, console build, verification
tests/               350+ tests, fast by default, `-m slow` for end-to-end
```

### Tests

```bash
pytest                      # fast unit tests
pytest -m slow              # end-to-end pipeline, agent, and service runs
bash scripts/verify_ui.sh   # drives the live app in a real browser
```

The slow tests are separated because they run the real pipeline on real frames,
which takes minutes rather than seconds. They are also where the defects that only
appear end to end get caught — four of them were found this way, including a
response that returned a 500 for exactly the case that most needs to work, and a
crash that would have taken the whole service process down.

---

## Limitations

Stated plainly, because a system that reports on the world should be judged on
where it stops working:

- **Planar assumption by default.** A homography is chosen for near-planar scenes
  and a fundamental matrix otherwise, but a scene that is neither is registered
  approximately at best.
- **Single-frame registration.** The baseline's first frame anchors the canonical
  frame; additional frames contribute to coverage but not to the alignment.
- **No metric scale.** The output is in canonical pixels. It says an object moved
  and roughly how far, not how many centimetres.
- **Synthetic evaluation data.** The benchmark scenes are generated with known
  answers, so they are the only place a ground-truth comparison is possible. Real
  captures can be uploaded through the app, but they have no answer key.
