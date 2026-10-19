# Review guide

Where everything lives, and what to look at first.

**Run the app** — `pip install -r requirements.txt && python scripts/serve.py`
→ <http://127.0.0.1:8000>. No configuration, no API keys, no model download
required to start; the five sample sites are generated on first run.

**Film (4:42, under the five-minute cap)** —
[strata-demo.mp4](../service/static/media/strata-demo.mp4) · also served at
`/media/strata-demo.mp4` when the app is running. It opens on authorship, runs
the application end to end, walks the architecture layer by layer, and closes on
the results and the failure cases. `python scripts/build_video.py --install`
rebuilds it from the committed diagram and the committed evaluation artefact.

---

## The five-minute version

1. Open <http://127.0.0.1:8000>. Pick **Loading Bay 4**.
2. Press **Run rescan**. The pipeline registers the second visit to the first and
   returns four typed changes.
3. Open **Compare** and drag the divider — the two visits are in one frame, so
   the slider compares *one place at two moments* rather than two photographs.
4. Open **Changes**. Four events, each typed, located, measured and scored. One
   of them is at 47% confidence and the interface says so.
5. Open **Coverage**. The system reports it only observed 30% of the frame well
   enough to compare, and asks for a specific extra capture.
6. Answer the gate at the bottom. Nothing is written until you do.

---

## Deliverables

| | Where |
|---|---|
| **Working application** | `python scripts/serve.py`; source in `service/` and `web/`; container recipe in `Dockerfile` |
| **Technical report** | [`docs/TECHNICAL_REPORT.md`](../docs/TECHNICAL_REPORT.md) |
| **Architecture diagram** | [`docs/architecture.svg`](../docs/architecture.svg) · [PNG](../docs/architecture.png) |
| **Source code** | This repository |
| **Pinned dependencies** | `requirements.txt` (ranges) · `requirements.lock` (the measured set) |
| **Build / deploy / test instructions** | [`README.md`](../README.md) · [`docs/DEPLOYMENT.md`](../docs/DEPLOYMENT.md) |
| **Demo video (≤5 min)** | [`service/static/media/strata-demo.mp4`](../service/static/media/strata-demo.mp4) |
| **Evaluation evidence** | [`docs/TECHNICAL_REPORT.md`](../docs/TECHNICAL_REPORT.md) §7 · [`docs/evaluation.json`](../docs/evaluation.json) (the raw run) · `scripts/evaluate.py` (regenerate) |
| **Failure cases and limitations** | Technical report §7.3 and §8 |
| **Responsible-use considerations** | [`docs/RESPONSIBLE_USE.md`](../docs/RESPONSIBLE_USE.md) |
| **Agent workflow and trace** | Technical report §5 · the trace panel in the app · `vision/strata_agent/` |
| **Cloud deployment** | [`deploy/`](../deploy) · [`docs/DEPLOYMENT.md`](../docs/DEPLOYMENT.md) |
| **Benchmark harness** | [`deploy/benchmark/`](../deploy/benchmark) · results in `deploy/benchmark/results/` |
| **UI states** | [`docs/screenshots/`](../docs/screenshots) |

---

## Reproducing the headline numbers

```bash
pip install -r requirements.lock
python scripts/fetch_models.py
python scripts/evaluate.py --cases 24 --out artifacts/eval
```

Expected, and reproduced exactly on the machine the figures were measured on:

```
registration error      0.665 px mean corner error
registration confidence 0.9008
precision               0.9375
recall                  0.8542
F1                      0.8875
event-kind accuracy     0.6979
still-pair false pos.   0.0   (12 unchanged pairs)
```

`python scripts/figures.py` prints these from the committed run rather than from
a copy, and `tests/test_docs_agree_with_artifact.py` fails if the documents stop
matching it.

The scenes are generated from seeds, so a difference in these numbers is a
difference in the code, not in the data.

---

## What to be sceptical about

Every figure quoted in these documents is generated from `docs/evaluation.json`
by `scripts/figures.py`, and `tests/test_docs_agree_with_artifact.py` fails if a
document stops matching the run. The whole suite — 468 tests, including 79 that
run the real pipeline end to end — passes on the locked dependency set, and
`.github/workflows/ci.yml` re-runs the benchmark and gates it on every push.

The report is deliberately specific about where this stops working, and those
sections are the ones worth reading closely:

- **Event-kind accuracy is 0.698** — locations are found far more reliably than
  they are labelled. The two numbers are reported separately rather than blended.
- **Ten of 24 benchmark cases report fewer events than expected.** One change is
  merged away. Documented and pinned by a test.
- **No Arm-vs-x86 performance ratio is published**, because no Graviton instance
  was available to measure it. The harness that produces it is in
  `deploy/benchmark/` and the protocol is in `docs/DEPLOYMENT.md` §3. Publishing
  a number that was not measured would contradict the point of the project.
- **Nothing in this repository yet demonstrates a scan that executed on AWS.**
  The infrastructure scripts are complete, idempotent, dry-runnable, and their
  storage contract is tested against a DynamoDB fake that rejects items the way
  the service does — but they have not been pointed at an account. `deploy/deploy.sh`
  then `python deploy/verify.py --record` is the step that closes that gap, and
  it is the one thing a reviewer should ask for first.
- **The benchmark is synthetic.** Real captures can be uploaded, but they have no
  answer key, so real-world accuracy is not claimed.
