"""The agent loop: perceive, decide, act, repeat.

This is where the deterministic pipeline and the decision layer meet. The shape
of the loop is deliberately narrow:

    observe  -> run the pipeline, summarise the state
    decide   -> ask a brain for the next action
    act      -> carry it out, then either loop or stop

Three properties of this loop are load-bearing, and each exists because of a
specific failure it prevents.

**It terminates.** Every action either consumes a capture from a finite budget,
consumes one of a finite number of "declare this unresolvable" slots, or is
terminal. A step ceiling backs that up. An agent that can loop forever is an
agent that will, and the operator standing in front of it has no way to tell the
difference between "thinking" and "stuck".

**It never writes without consent.** The only action that changes stored state is
``finalize``, and it is refused unless an approver says yes. If no approver is
supplied, the loop records the request and stops - it does not assume consent.
Silence is not permission.

**It degrades rather than fails.** A brain that errors produces a rule-based
decision (see :mod:`strata_agent.brain`), a capture request that cannot be
satisfied is recorded as such, and a run that hits its budget still produces a
usable result with the shortfall stated explicitly. Nothing here raises because
an optional component misbehaved.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

import numpy as np

from strata_core.config import PipelineConfig
from strata_core.coverage import CoverageMap, coverage_summary, direction_to_cell, view_requests
from strata_core.events import ChangeEvent
from strata_core.pipeline import ScanResult, run_scan

from .actions import (
    ABORT,
    ACCEPT_SCAN,
    ALL_ACTIONS,
    FINALIZE,
    MARK_UNCERTAIN,
    REQUEST_VIEW,
    Action,
    Observation,
)
from .brain import Brain, BrainResult, make_brain
from .policy import AgentConfig, PolicyState
from .trace import AgentTrace

log = logging.getLogger("strata.agent.loop")


class FrameProvider(Protocol):
    """Supplies an extra capture when the agent asks for one."""

    def __call__(
        self, direction: str, target_cell: Sequence[int] | None, reason: str
    ) -> np.ndarray | None:  # pragma: no cover - protocol
        ...


Approver = Callable[[Action], bool]


@dataclass
class AgentRun:
    """Everything one agent run produced."""

    trace: AgentTrace
    result: ScanResult | None = None
    events: list[ChangeEvent] = field(default_factory=list)
    coverage: CoverageMap | None = None
    accepted: bool = False
    finalized: bool = False
    outcome: str = "running"
    finalize_action: Action | None = None

    def to_dict(self) -> dict:
        return {
            "outcome": self.outcome,
            "accepted": self.accepted,
            "finalized": self.finalized,
            "event_count": len(self.events),
            "trace": self.trace.to_dict(),
        }


# --- Observation construction -------------------------------------------


def build_observation(
    result: ScanResult,
    *,
    captures_used: int,
    capture_budget: int,
    weak_event_confidence: float = 0.45,
) -> Observation:
    """Summarise a scan into the state the agent reasons over.

    Every number here is measured, never inferred. The agent's value depends on
    the model being able to trust this summary completely - if it contained an
    estimate or a guess, the model would be reasoning about a fiction and its
    rationale would read as confident nonsense.
    """
    reg = result.registration
    cmap = result.coverage
    requests = view_requests(cmap, result.config.coverage)
    summary = coverage_summary(cmap, requests)

    events = list(result.events)
    kinds: dict[str, int] = {}
    for e in events:
        kinds[e.kind] = kinds.get(e.kind, 0) + 1

    confident = [e for e in events if e.kind != "uncertain"]
    uncertain = [e for e in events if e.kind == "uncertain"]
    weak = [e for e in events if e.kind != "uncertain" and e.confidence < weak_event_confidence]

    return Observation(
        registration_confidence=float(reg.confidence),
        registration_model=str(reg.model),
        registration_error_px=float(reg.reproj_error_px),
        coverage_status=str(cmap.status),
        observed_fraction=float(cmap.observed_fraction),
        frame_count=int(cmap.frame_count),
        view_requests=[r.to_dict() for r in requests],
        event_count=len(events),
        uncertain_count=len(uncertain),
        low_confidence_count=len(weak),
        mean_confidence=(
            float(np.mean([e.confidence for e in confident])) if confident else 0.0
        ),
        kind_counts=kinds,
        uncertain_targets=_uncertain_targets(uncertain, weak, cmap),
        captures_used=int(captures_used),
        capture_budget=int(capture_budget),
        views_credited=sum(1 for v in result.extra_views if v.get("registered")),
    )


def _uncertain_targets(
    uncertain: list[ChangeEvent],
    weak: list[ChangeEvent],
    cmap: CoverageMap,
) -> list[dict]:
    """Where the unresolved regions are, aimed at the coverage grid.

    Both genuinely-uncertain events and merely low-confidence ones are included.
    They are different in kind - the first says "I cannot tell what this is", the
    second says "I think I know, but not firmly" - but they call for the same
    remedy, which is another look at that spot. Ordering puts the genuinely
    uncertain first, because those are the ones that could change the report.
    """
    out: list[dict] = []
    h, w = cmap.rows, cmap.cols
    # Cell coordinates come from the canonical frame, which the event centroids
    # are already expressed in.
    for e in uncertain + weak:
        cell = _centroid_to_cell(e.centroid, cmap)
        direction, reason = direction_to_cell(cmap, cell)
        out.append(
            {
                "event_id": e.event_id,
                "cell": [int(cell[0]), int(cell[1])],
                "direction": direction,
                "kind": e.kind,
                "confidence": round(float(e.confidence), 4),
                "reason": (
                    "the classifier could not decide what changed here"
                    if e.kind == "uncertain"
                    else f"confidence is only {e.confidence:.2f} ({e.kind})"
                ),
            }
        )
    return out


def _centroid_to_cell(centroid: Sequence[float], cmap: CoverageMap) -> tuple[int, int]:
    """Map a canonical-space centroid onto the coverage grid.

    The canonical size is read from the map rather than assumed. Event centroids
    are in canonical pixels, and canonical size depends on configuration, so
    hard-coding it here would silently mis-place guidance the moment the
    canonical long edge changed - and a view request aimed at the wrong part of
    the frame is worse than no request, because it looks like it worked.
    """
    cw, ch = cmap.canonical_size
    cx, cy = float(centroid[0]), float(centroid[1])
    col = int(np.clip(cx / max(1.0, float(cw)) * cmap.cols, 0, cmap.cols - 1))
    row = int(np.clip(cy / max(1.0, float(ch)) * cmap.rows, 0, cmap.rows - 1))
    return row, col


# --- The loop ------------------------------------------------------------


def run_agent(
    baseline_frames: list[np.ndarray],
    rescan_frames: list[np.ndarray],
    *,
    config: PipelineConfig | None = None,
    agent_config: AgentConfig | None = None,
    brain: Brain | None = None,
    frame_provider: FrameProvider | None = None,
    approver: Approver | None = None,
    goal: str = "Determine what changed at this location between the two visits.",
    max_steps: int = 24,
) -> AgentRun:
    """Run the perceive-decide-act loop to completion.

    Args:
        baseline_frames: the reference visit.
        rescan_frames: the comparison visit. Extra captures requested during the
            run are appended to this list.
        config: pipeline configuration.
        agent_config: decision thresholds.
        brain: decision maker. Defaults to the best available, which is the
            deterministic policy when no model is configured.
        frame_provider: supplies extra captures. If absent, a view request is
            recorded but cannot be satisfied, and the loop proceeds on the
            evidence it has rather than stalling.
        approver: consulted before any state-changing action. If absent, such an
            action is proposed and recorded but not carried out.
        goal: the objective, recorded in the trace and shown to the model.
        max_steps: hard ceiling on iterations.
    """
    cfg = config or PipelineConfig()
    acfg = agent_config or AgentConfig()
    brain = brain or make_brain(cfg=acfg)

    trace = AgentTrace(goal=goal)
    state = PolicyState()
    rescan = list(rescan_frames)
    run = AgentRun(trace=trace)
    captures_used = 0

    # The scan is re-run only when the inputs actually change. A decision such as
    # "mark this unresolved" or "accept" does not alter the frames, so re-running
    # the pipeline would burn seconds of compute to produce a byte-identical
    # result - and, in a demo, would make the agent look slow for no reason. The
    # observation is derived from the same cached result, which is correct
    # because the only observation field that can change without the frames
    # changing is ``captures_used``, and that changes only when a capture is
    # supplied.
    result: ScanResult | None = None
    last_frame_count = -1

    for _ in range(max_steps):
        # --- observe -----------------------------------------------------
        t0 = time.perf_counter()
        if result is None or len(rescan) != last_frame_count:
            result = run_scan(baseline_frames, rescan, config=cfg)
            last_frame_count = len(rescan)
            rescanned = True
        else:
            rescanned = False
        obs = build_observation(
            result,
            captures_used=captures_used,
            capture_budget=acfg.capture_budget,
            weak_event_confidence=acfg.weak_event_confidence,
        )
        run.result = result
        run.events = list(result.events)
        run.coverage = result.coverage
        trace.add(
            phase="observe",
            action="scan" if rescanned else "scan (cached)",
            rationale=(
                f"{len(result.events)} change(s) found; coverage "
                f"{obs.observed_fraction * 100:.0f}% ({obs.coverage_status}); "
                f"registration confidence {obs.registration_confidence:.2f}."
                + (
                    f" {obs.views_credited} further view(s) of this visit were "
                    "registered in and credited as coverage."
                    if obs.views_credited
                    else ""
                )
                + ("" if rescanned else " Frames unchanged, so this is the previous result.")
            ),
            decided_by="pipeline",
            observation=obs.to_dict(),
            result={
                "event_count": len(result.events),
                "uncertain_count": obs.uncertain_count,
                "coverage_status": obs.coverage_status,
                "rescanned": rescanned,
            },
            duration_ms=(time.perf_counter() - t0) * 1000.0,
        )

        # --- decide ------------------------------------------------------
        allowed = _allowed_actions(obs, state, acfg)
        t0 = time.perf_counter()
        try:
            br: BrainResult = brain.decide(obs, acfg, allowed=allowed, state=state)
        except Exception as exc:  # noqa: BLE001
            # A brain is never allowed to break the loop. If even the fallback
            # chain fails, the run stops with an honest outcome rather than
            # continuing on a decision nobody made.
            log.exception("brain chain failed entirely")
            trace.add(
                phase="decide",
                action="error",
                rationale=f"No decision could be made: {exc}",
                decided_by="none",
            )
            run.outcome = "brain_failed"
            trace.outcome = run.outcome
            trace.summary = "The decision layer failed; no action was taken."
            return run

        action = br.action
        trace.add(
            phase="decide",
            action=action.name,
            rationale=action.rationale,
            decided_by=br.decided_by,
            args=action.args,
            confidence=action.confidence,
            result={"error": br.error} if br.error else {},
            duration_ms=(time.perf_counter() - t0) * 1000.0,
        )

        # --- act ---------------------------------------------------------
        t0 = time.perf_counter()
        if action.name == REQUEST_VIEW:
            supplied = None
            if frame_provider is not None:
                supplied = frame_provider(
                    str(action.args.get("direction", "")),
                    action.args.get("target_cell"),
                    str(action.args.get("reason", "")),
                )
            if supplied is None:
                # No provider, or it declined. Mark captures as unavailable so
                # the agent stops offering itself an option it cannot exercise,
                # then continue with the evidence already gathered. Retrying a
                # request that cannot be satisfied wastes the budget and fills
                # the trace with identical steps - which is exactly what an
                # earlier version of this loop did, five times over.
                state.views_available = False
                trace.add(
                    phase="act",
                    action=REQUEST_VIEW,
                    rationale=(
                        "Requested a further capture, but none was available. No "
                        "further captures will be requested; continuing with the "
                        "evidence already gathered."
                    ),
                    decided_by="loop",
                    result={"satisfied": False, "views_available": False},
                    duration_ms=(time.perf_counter() - t0) * 1000.0,
                )
                continue

            rescan = rescan + [supplied]
            captures_used += 1
            trace.add(
                phase="act",
                action=REQUEST_VIEW,
                rationale=(
                    f"Capture received ({action.args.get('direction', 'extra view')}); "
                    f"re-running the comparison with {len(rescan)} frame(s)."
                ),
                decided_by="loop",
                result={
                    "satisfied": True,
                    "captures_used": captures_used,
                    "budget_remaining": max(0, acfg.capture_budget - captures_used),
                },
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )
            continue

        if action.name == MARK_UNCERTAIN:
            state.note_marked(action.args.get("event_id"))
            if action.args.get("scope"):
                state.coverage_shortfall_noted = True
            trace.add(
                phase="act",
                action=MARK_UNCERTAIN,
                rationale="Recorded as unresolved rather than assumed unchanged.",
                decided_by="loop",
                result={
                    "marked": state.uncertain_marked,
                    "scope": action.args.get("scope", "region"),
                },
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )
            if state.uncertain_marked >= acfg.max_uncertain_marked:
                # Enough has been declared unresolvable; stop asking.
                state.accepted = True
            continue

        if action.name == ACCEPT_SCAN:
            state.accepted = True
            run.accepted = True
            trace.add(
                phase="act",
                action=ACCEPT_SCAN,
                rationale="Scan accepted; the change set is ready to commit.",
                decided_by="loop",
                result={"accepted": True},
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )
            continue

        if action.name == FINALIZE:
            run.finalize_action = action
            approved = bool(approver(action)) if approver is not None else False
            trace.add(
                phase="gate",
                action="approval",
                rationale=(
                    "Human approval granted."
                    if approved
                    else "Human approval not granted; nothing was written."
                ),
                decided_by="human",
                args={"requested": action.name},
                result={"approved": approved},
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )
            if approved:
                run.finalized = True
                run.outcome = "finalized"
                trace.outcome = run.outcome
                trace.summary = (
                    f"Committed {len(run.events)} change(s) after "
                    f"{captures_used} extra capture(s)."
                )
            else:
                run.outcome = "declined"
                trace.outcome = run.outcome
                trace.summary = (
                    "The change set was prepared but not committed, because "
                    "approval was withheld."
                )
            return run

        if action.name == ABORT:
            run.outcome = "aborted"
            trace.add(
                phase="finish",
                action=ABORT,
                rationale=action.rationale or "Stopped without recording anything.",
                decided_by="loop",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )
            trace.outcome = run.outcome
            trace.summary = "No change set was produced."
            return run

        # An action in the catalogue that the loop does not implement is a bug in
        # the catalogue, not something to paper over.
        raise RuntimeError(f"unhandled action {action.name!r}")

    # Step ceiling reached.
    run.outcome = "step_limit"
    trace.outcome = run.outcome
    trace.summary = (
        f"Stopped after {max_steps} steps without reaching a terminal action. "
        f"{len(run.events)} change(s) were found and are reported as-is."
    )
    return run


def _allowed_actions(
    obs: Observation, state: PolicyState, cfg: AgentConfig
) -> tuple[str, ...]:
    """Which actions make sense right now.

    Narrowing the menu is what keeps a model's choice meaningful. Offering
    "request another capture" when the budget is spent invites a decision that
    cannot be carried out, and offering "finalize" before the scan is accepted
    invites committing something unreviewed. The model chooses; the loop decides
    what is choosable.
    """
    allowed: list[str] = []

    if obs.registration_confidence < cfg.abort_registration_confidence:
        # Nothing else is defensible when the captures do not align.
        return (ABORT,)

    if not state.accepted:
        if obs.budget_remaining > 0 and state.views_available:
            allowed.append(REQUEST_VIEW)
        allowed.append(MARK_UNCERTAIN)
        allowed.append(ACCEPT_SCAN)
        allowed.append(ABORT)
        return tuple(allowed)

    # Accepted: the only remaining question is whether to commit.
    allowed.append(FINALIZE)
    allowed.append(ABORT)
    return tuple(allowed)


def allowed_actions_for(
    obs: Observation, state: PolicyState | None = None, cfg: AgentConfig | None = None
) -> tuple[str, ...]:
    """Public wrapper over the constraint logic, for the API and the tests."""
    return _allowed_actions(obs, state or PolicyState(), cfg or AgentConfig())
