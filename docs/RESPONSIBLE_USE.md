# Responsible use

STRATA observes physical spaces. That makes consent, retention and the limits of
the system part of the design rather than a disclaimer at the end.

---

## What the system does not do

- **No face recognition.** There is no face model in the pipeline, and none in
  the dependency set.
- **No person identification.** No biometric template, no re-identification, no
  gait or body signature.
- **No behavioural inference.** The output is a set of changed *regions* and
  their types. It says "a 10,000-pixel region arrived here", not "a person took
  something".
- **No autonomous acquisition.** Capture guidance is an instruction to a human
  holding a camera, not a command to a motorised platform.

These are structural properties, not policy promises: the schema has nowhere to
put a person-level signal.

## Consent and control

A site is created deliberately and named. Nothing is observed until someone
points a camera at it.

The **approval gate is the control surface**. A scan prepares a change set and
stops with outcome `declined` — nothing is written. The ledger is appended only
when a person answers the gate, and the resulting entry records
`decision_made_by: human` alongside the note they typed.

That means the record answers "who decided this was a change", which is the
question an audit actually asks.

## Uncertainty is surfaced, not suppressed

The most useful behaviour in a safety or compliance context is refusing to
answer. STRATA reports what it could not observe well enough to compare, and
turns that into a specific request for another capture rather than folding it
into a confident "nothing changed".

A confident negative over ground nobody photographed is worse than no answer,
because it is actionable and wrong. The system is built so that this failure is
visible in the interface rather than hidden in an aggregate score.

## Data handling

- Sessions are stored under a per-site prefix, and store keys are validated
  before use, so a session identifier cannot escape its prefix.
- Uploads are bounded by size, frame count and dimension at the API boundary.
- The S3 backend blocks all public access and encrypts at rest.
- **Nothing is sent to a third party by default.** The optional model call site
  is disabled unless a key is configured, and the product is fully functional
  without it.
- Retention is a deployment decision. The local store writes to a directory the
  operator chooses; the managed store is a bucket and a table the operator owns.
  Neither expires data on its own, which is deliberate: a record that silently
  forgets is not a record.

## Dual use

Change detection can be used for monitoring people's spaces. The mitigations are
structural and are stated here so a deployer can judge appropriateness rather
than discover the boundaries later:

- No identity signal exists to misuse.
- Every write is human-gated and attributed.
- Sites are explicitly created, not discovered.
- The audit trail is append-only, so the record of what was observed is itself
  observable.

A deployment that adds cameras in areas where people have a reasonable
expectation of privacy is not what this system is for, and the absence of a face
model does not make such a deployment appropriate.

## Honest limits

- **Synthetic evaluation.** The benchmark uses generated scenes with known
  answers. Real captures can be uploaded, but they have no answer key, so
  real-world accuracy is not claimed here.
- **No metric scale.** Output is in canonical pixels.
- **Planar scenes are the strong case.** A scene that is neither planar nor
  well-modelled by a fundamental matrix is registered approximately at best.
- **Adjacent changes can merge** into a single reported region. Documented, and
  pinned by a test.

Where the system is weak, it is documented rather than tuned around, because a
tool that reports on the world should be judged on where it stops working.
