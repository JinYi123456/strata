"""The deterministic decision policy.

This module is the agent's spine. It is not a degraded fallback that runs when
the model is unavailable - it is the reference implementation of the decision
logic, and the model is an optional layer on top of it.

That ordering matters, and it is the opposite of how these systems are usually
built. The usual arrangement is a model that decides, plus a set of guardrails
to catch it when it decides badly. The problem is that guardrails are reactive:
they can veto, but they cannot produce the right answer, so the system's
behaviour under failure is "do nothing" or "do the last safe thing".

Here the rules decide, and the model is consulted for *judgement* the rules
cannot encode - weighing a marginal trade-off, or recognising a situation the
rules were not written for. Two consequences follow:

* The system is fully functional with no API key, no network, and no model. It
  is deterministic, so the same observation always produces the same action, and
  a test can assert on it.
* When the model is present, its output is still validated against the same
  action vocabulary, and any failure - timeout, malformed response, refusal -
  degrades to these rules rather than to an error. A demo cannot break because
  someone's rate limit expired.

Each rule below carries its rationale in prose, and that prose is what the trace
records. A reader should be able to follow a run made entirely by these rules
and understand every decision without knowing a model was ever involved.
"""

from __future__ import annotations

from dataclasses import dataclass

from .actions import (
    ABORT,
    ACCEPT_SCAN,
    FINALIZE,
    MARK_UNCERTAIN,
    REQUEST_VIEW,
    Action,
    Observation,
)


@dataclass(frozen=True)
class AgentConfig:
    """Thresholds for the decision rules.

    Frozen, and every value is a deliberate choice rather than a tuned constant.
    None of them were fitted to a benchmark: they are the points at which the
    *meaning* of the underlying quantity changes, which is why they are stated
    in those terms in the comments below.
    """

    # How many extra captures the agent may ask for. Three is chosen from the
    # operator's side, not the algorithm's: a person holding a phone will
    # reasonably take a couple of extra shots, and will abandon the process if
    # asked for six. An agent that is right but ignored is worth nothing.
    capture_budget: int = 3

    # Registration confidence below which the two captures cannot be meaningfully
    # compared at all. Below this, reporting changes would be reporting noise, so
    # the honest action is to stop rather than to guess.
    abort_registration_confidence: float = 0.30

    # ...and below which the comparison is possible but not trustworthy. Here a
    # further capture genuinely helps - registration improves with more matched
    # features - so this is a reason to ask, not to give up.
    weak_registration_confidence: float = 0.55

    # Fraction of the canonical frame that must be well observed before the scan
    # is considered adequate. Two-thirds is the point at which the unobserved
    # remainder is a border region rather than a missed part of the scene.
    adequate_coverage_fraction: float = 0.60

    # An event below this confidence is one the agent should try to firm up
    # before committing, if it can.
    weak_event_confidence: float = 0.45

    # Cap on how many regions may be declared unresolvable in one run, so a
    # pathological scene cannot produce an all-uncertain result and call it a
    # completed scan.
    max_uncertain_marked: int = 4

    # Floor on registration confidence for the *finalize* step. Committing a
    # change set derived from a barely-registered pair is the one outcome worth
    # refusing outright.
    min_finalize_registration_confidence: float = 0.45


def decide(obs: Observation, cfg: AgentConfig, *, state: "PolicyState | None" = None) -> Action:
    """Choose the next action from the observation.

    Rules are evaluated in order of how badly the situation can go wrong, not in
    order of how often they fire. The first rule - "can these two captures be
    compared at all" - gates everything downstream, because every later
    judgement is meaningless if the answer is no.
    """
    state = state or PolicyState()

    # --- 1. Can the comparison be trusted at all? ------------------------
    if obs.registration_confidence < cfg.abort_registration_confidence:
        return Action(
            name=ABORT,
            args={},
            rationale=(
                f"Registration confidence is {obs.registration_confidence:.2f}, "
                f"below the {cfg.abort_registration_confidence:.2f} floor. At this "
                "level the two captures are not reliably aligned, so any difference "
                "between them would be as likely to be alignment error as a real "
                "change. Reporting nothing is the honest outcome."
            ),
            confidence=0.95,
        )

    if (
        obs.registration_confidence < cfg.weak_registration_confidence
        and obs.budget_remaining > 0
        and state.views_available
        and not state.accepted
    ):
        return Action(
            name=REQUEST_VIEW,
            args={
                "direction": "step back",
                "reason": "improve registration",
                "target_cell": _widest_gap_cell(obs),
            },
            rationale=(
                f"Registration confidence is {obs.registration_confidence:.2f}, which "
                "is workable but not trustworthy. More of the scene in frame gives the "
                "matcher more to lock onto, so one wider capture is worth spending."
            ),
            confidence=0.7,
        )

    # --- 2. Is anything still unresolved, and can a capture fix it? ------
    if obs.budget_remaining > 0 and state.views_available and not state.accepted:
        target = _next_uncertain_target(obs, state)
        if target is not None:
            return Action(
                name=REQUEST_VIEW,
                args={
                    "direction": target.get("direction", "step closer"),
                    "reason": "resolve uncertain change",
                    "target_cell": target.get("cell"),
                    "event_id": target.get("event_id"),
                },
                rationale=(
                    f"The change at {target.get('event_id', 'this location')} is "
                    f"unresolved: {target.get('reason', 'confidence is low')}. A "
                    "closer capture of that area is the cheapest way to settle it, "
                    f"and {obs.budget_remaining} capture(s) remain in the budget."
                ),
                confidence=0.75,
            )

    # --- 3. Has enough of the scene been seen? --------------------------
    if not state.accepted:
        if obs.observed_fraction < cfg.adequate_coverage_fraction:
            if obs.budget_remaining > 0 and state.views_available and obs.view_requests:
                top = obs.view_requests[0]
                return Action(
                    name=REQUEST_VIEW,
                    args={
                        "direction": top.get("direction", "pan"),
                        "reason": "improve coverage",
                        "target_cell": top.get("target_cell"),
                    },
                    rationale=(
                        f"Only {obs.observed_fraction * 100:.0f}% of the frame is well "
                        f"observed, against a {cfg.adequate_coverage_fraction * 100:.0f}% "
                        f"bar. The weakest area needs '{top.get('direction', 'another view')}'"
                        f" - {top.get('reason', 'it is under-observed')}."
                    ),
                    confidence=0.7,
                )
            # No budget left, no captures obtainable, or nowhere specific left to
            # point at. Record the shortfall once, then accept on the next pass
            # rather than repeating the same observation.
            if not state.coverage_shortfall_noted and state.uncertain_marked < cfg.max_uncertain_marked:
                return Action(
                    name=MARK_UNCERTAIN,
                    args={
                        "reason": (
                            f"coverage stalled at {obs.observed_fraction * 100:.0f}% "
                            "with no further captures obtainable"
                        ),
                        "scope": "unobserved regions",
                    },
                    rationale=(
                        f"Coverage is {obs.observed_fraction * 100:.0f}%, below the "
                        f"{cfg.adequate_coverage_fraction * 100:.0f}% bar, and no more "
                        "captures are available. Rather than accept a scan that implies "
                        "the unseen areas were unchanged, this is recorded as "
                        "unresolved - absence of evidence is not evidence of absence."
                    ),
                    confidence=0.8,
                )

        # Coverage is adequate and nothing is outstanding: accept.
        #
        # The wording distinguishes two things a reader will otherwise conflate:
        # an *uncertain event* is a change the classifier could not type, while a
        # *coverage shortfall* is an area never seen well enough to judge. A
        # shortfall does not block acceptance - it is recorded alongside the
        # result - but the rationale must say so rather than implying a clean
        # scan.
        shortfall = (
            " Part of the frame was never observed well enough to judge, and that "
            "is recorded with this result rather than presented as unchanged."
            if state.coverage_shortfall_noted
            else ""
        )
        return Action(
            name=ACCEPT_SCAN,
            args={},
            rationale=(
                f"Coverage is {obs.observed_fraction * 100:.0f}% and registration "
                f"confidence is {obs.registration_confidence:.2f}. "
                f"{obs.event_count} change(s) were found, with "
                f"{obs.uncertain_count} unresolved and "
                f"{obs.low_confidence_count} low-confidence.{shortfall} "
                "The evidence is sufficient to state what changed, so the scan is "
                "accepted."
            ),
            confidence=0.85,
        )

    # --- 4. Accepted: commit, subject to approval -----------------------
    if obs.registration_confidence < cfg.min_finalize_registration_confidence:
        return Action(
            name=ABORT,
            args={},
            rationale=(
                f"The scan was accepted, but registration confidence "
                f"({obs.registration_confidence:.2f}) is below the "
                f"{cfg.min_finalize_registration_confidence:.2f} needed to commit a "
                "change set. Declining to write a record that would not survive "
                "scrutiny."
            ),
            confidence=0.9,
        )

    return Action(
        name=FINALIZE,
        args={"event_count": obs.event_count, "uncertain_count": obs.uncertain_count},
        rationale=(
            f"Scan accepted with {obs.event_count} change(s) at "
            f"{obs.registration_confidence:.2f} registration confidence. Committing "
            "this as the canonical record - which requires your approval, because it "
            "is the only step that changes stored state."
        ),
        confidence=0.8,
    )


@dataclass
class PolicyState:
    """What the policy remembers between decisions.

    Deliberately tiny. An agent whose memory is large is an agent whose behaviour
    is hard to predict, and this one has to be predictable enough to test. It
    tracks only what the rules genuinely need: whether the scan has been
    accepted, how many regions have been declared unresolvable (so the cap can be
    enforced), and whether further captures are obtainable at all.

    ``views_available`` exists because of a specific observed failure: with no
    way to supply an extra frame, the agent asked for one, was refused, and asked
    again - five identical requests before the counter happened to stop it. The
    counter was the wrong mechanism. An agent that cannot get a capture should
    *know* it cannot, and stop offering itself the option, rather than being
    gradually throttled into giving up.
    """

    accepted: bool = False
    uncertain_marked: int = 0
    marked_event_ids: list[str] = None  # type: ignore[assignment]
    views_available: bool = True
    # Whether the coverage shortfall itself has already been recorded. Without
    # this the policy re-records "coverage is inadequate" on every pass once
    # captures are unavailable, producing a trace that says the same thing four
    # times and buries the decisions that actually differ.
    coverage_shortfall_noted: bool = False

    def __post_init__(self) -> None:
        if self.marked_event_ids is None:
            self.marked_event_ids = []

    def note_marked(self, event_id: str | None) -> None:
        self.uncertain_marked += 1
        if event_id:
            self.marked_event_ids.append(event_id)


def _next_uncertain_target(obs: Observation, state: PolicyState) -> dict | None:
    """The most valuable unresolved region to re-shoot, if any.

    Highest-priority first, skipping anything already declared unresolvable so
    the agent cannot ask for the same capture twice.
    """
    for target in obs.uncertain_targets:
        eid = target.get("event_id")
        if eid and eid in (state.marked_event_ids or []):
            continue
        return target
    return None


def _widest_gap_cell(obs: Observation) -> list[int] | None:
    """The worst-covered cell, used when registration itself is the problem."""
    if not obs.view_requests:
        return None
    return obs.view_requests[0].get("target_cell")
