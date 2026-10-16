"""The agent's observation and action vocabulary.

These two types are the entire interface between perception and decision. The
loop builds an :class:`Observation` from the deterministic pipeline, and a brain
- either the model or the fallback policy - returns an :class:`Action`. Keeping
the vocabulary in one module means the model's prompt, the fallback's rules, and
the loop's executor cannot drift apart: they all read the same catalogue.

**Why the action set is this small.** An agent is only as trustworthy as the set
of things it is allowed to do. Every action here is either *asking for more
information* or *recording a judgement*; none of them mutate the canonical
record without a human saying so. That asymmetry is deliberate. A system that
can silently overwrite the record of what a place looked like is a system nobody
should point at anything they care about, so the one action that commits
anything - :data:`FINALIZE` - is marked as requiring approval, and the loop
refuses to execute it otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# --- Action catalogue ----------------------------------------------------
#
# Each entry: (name, requires_approval, description). The description is what
# the model sees when choosing, so it is written as an instruction to a
# decision-maker rather than as reference documentation.

REQUEST_VIEW = "request_view"
MARK_UNCERTAIN = "mark_uncertain"
ACCEPT_SCAN = "accept_scan"
FINALIZE = "finalize"
ABORT = "abort"

ACTION_CATALOGUE: dict[str, dict[str, Any]] = {
    REQUEST_VIEW: {
        "requires_approval": False,
        "description": (
            "Ask the operator for one more capture, aimed at a specific "
            "under-observed region. Use when coverage is insufficient to "
            "judge an area, and budget remains."
        ),
    },
    MARK_UNCERTAIN: {
        "requires_approval": False,
        "description": (
            "Record that a specific region cannot be resolved from the evidence "
            "available, and stop trying. Use when more captures will not help - "
            "for example the area is occluded, or the budget is nearly spent."
        ),
    },
    ACCEPT_SCAN: {
        "requires_approval": False,
        "description": (
            "Declare the scan good enough to compare. Use when coverage is "
            "adequate and no region is still ambiguous, or when no further "
            "capture would improve the outcome."
        ),
    },
    FINALIZE: {
        "requires_approval": True,
        "description": (
            "Commit the accepted change set to the canonical record. This is "
            "the only action that changes stored state, and it requires explicit "
            "human approval."
        ),
    },
    ABORT: {
        "requires_approval": False,
        "description": (
            "Stop without recording anything. Use when the capture is unusable - "
            "for example registration failed, so no comparison is meaningful."
        ),
    },
}

# The subset a brain may choose from in a given situation is computed by the
# loop; this is the full ordering used when nothing else constrains it.
ALL_ACTIONS = tuple(ACTION_CATALOGUE)


def requires_approval(action: str) -> bool:
    return bool(ACTION_CATALOGUE.get(action, {}).get("requires_approval", False))


def describe(action: str) -> str:
    return str(ACTION_CATALOGUE.get(action, {}).get("description", ""))


@dataclass(frozen=True)
class Observation:
    """The complete state the agent reasons over at one decision point.

    Frozen because a decision must be explainable against the exact state that
    produced it. If this could be mutated after the fact, the trace would record
    a rationale for a situation that no longer existed.
    """

    # Registration - can these two captures even be compared?
    registration_confidence: float
    registration_model: str
    registration_error_px: float

    # Coverage - how much of the scene was actually seen?
    coverage_status: str
    observed_fraction: float
    frame_count: int
    view_requests: list[dict] = field(default_factory=list)

    # Change detection - what was found, and how sure is it?
    event_count: int = 0
    uncertain_count: int = 0
    low_confidence_count: int = 0
    mean_confidence: float = 0.0
    kind_counts: dict[str, int] = field(default_factory=dict)
    # Where the unresolved regions actually are, so a view request can be aimed
    # at them rather than at the frame in general. Each entry carries the
    # coverage-grid cell, the event id, and the reason it is unresolved.
    uncertain_targets: list[dict] = field(default_factory=list)

    # Budget - how many more captures may be requested?
    captures_used: int = 0
    capture_budget: int = 3

    @property
    def budget_remaining(self) -> int:
        return max(0, self.capture_budget - self.captures_used)

    @property
    def coverage_is_adequate(self) -> bool:
        return self.coverage_status == "observed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "registration_confidence": round(float(self.registration_confidence), 4),
            "registration_model": self.registration_model,
            "registration_error_px": round(float(self.registration_error_px), 3),
            "coverage_status": self.coverage_status,
            "observed_fraction": round(float(self.observed_fraction), 4),
            "frame_count": int(self.frame_count),
            "view_requests": self.view_requests,
            "event_count": int(self.event_count),
            "uncertain_count": int(self.uncertain_count),
            "low_confidence_count": int(self.low_confidence_count),
            "mean_confidence": round(float(self.mean_confidence), 4),
            "kind_counts": dict(self.kind_counts),
            "uncertain_targets": list(self.uncertain_targets),
            "captures_used": int(self.captures_used),
            "capture_budget": int(self.capture_budget),
            "budget_remaining": int(self.budget_remaining),
        }


@dataclass(frozen=True)
class Action:
    """A decision: what to do next, and why.

    ``confidence`` is the decider's own certainty in this choice, and it is
    recorded rather than acted on - the loop does not threshold it. It exists so
    a reader can see where the agent was unsure, which is often more informative
    than the decision itself.
    """

    name: str
    args: dict = field(default_factory=dict)
    rationale: str = ""
    confidence: float = 0.0

    @property
    def requires_approval(self) -> bool:
        return requires_approval(self.name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "args": self.args,
            "rationale": self.rationale,
            "confidence": round(float(self.confidence), 4),
            "requires_approval": self.requires_approval,
        }


def parse_action(payload: dict[str, Any]) -> Action:
    """Build an :class:`Action` from an untrusted mapping.

    A model's output is untrusted input, so this validates rather than trusts:
    an unknown action name is rejected outright, and the free-text fields are
    coerced to strings and length-capped so a runaway generation cannot bloat
    the trace. Returning a rejection is better than silently defaulting to some
    action, because a default would make a malformed response look like a
    deliberate decision.
    """
    if not isinstance(payload, dict):
        raise ValueError("action payload must be a mapping")
    name = payload.get("name") or payload.get("action")
    if name not in ACTION_CATALOGUE:
        raise ValueError(f"unknown action {name!r}")
    args = payload.get("args") or {}
    if not isinstance(args, dict):
        raise ValueError("action args must be a mapping")
    rationale = str(payload.get("rationale") or "").strip()[:600]
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return Action(
        name=str(name),
        args={str(k): v for k, v in args.items()},
        rationale=rationale,
        confidence=max(0.0, min(1.0, confidence)),
    )
