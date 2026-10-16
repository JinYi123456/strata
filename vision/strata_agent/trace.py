"""The agent's reasoning trace.

An agent that cannot explain itself is not an agent - it is a black box with a
schedule. Everything STRATA's agent does is recorded here as an ordered list of
steps, each carrying the observation it saw, the action it chose, the reason it
chose it, and what happened as a result.

This is deliberately not a log. A log is written for a developer debugging a
crash; a trace is written for a person deciding whether to trust the system.
That difference drives three choices:

* **Every step is self-contained.** A step records the full observation it
  reasoned over, not a diff against the previous one, so a reader can jump to
  any point and understand the state of the world without replaying history.
* **The rationale is prose.** ``rationale`` is a sentence a non-expert can read,
  produced by whichever brain made the decision. When the deterministic policy
  decides, the sentence explains the rule that fired; when the model decides, it
  explains its own reasoning. A reader should not be able to tell which is which
  from the format - only from the recorded ``decided_by`` field.
* **Provenance is explicit.** Each decision names the brain that made it, so a
  run can be audited after the fact and a mixed run (model for some steps,
  fallback for others) is visible rather than hidden.

The trace is the artefact the UI renders as the reasoning timeline, so its shape
is also an interface contract: it must serialise to plain JSON with no numpy
types and no non-deterministic ordering.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

# Phases of the perceive -> decide -> act cycle. Recording the phase separately
# from the action name lets the UI group steps without knowing every action.
PHASES = ("observe", "decide", "act", "gate", "finish")


@dataclass
class AgentStep:
    """One recorded step of the agent's cycle."""

    index: int
    phase: str
    action: str
    rationale: str
    decided_by: str
    observation: dict = field(default_factory=dict)
    args: dict = field(default_factory=dict)
    result: dict = field(default_factory=dict)
    confidence: float = 0.0
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": int(self.index),
            "phase": self.phase,
            "action": self.action,
            "rationale": self.rationale,
            "decided_by": self.decided_by,
            "observation": self.observation,
            "args": self.args,
            "result": self.result,
            "confidence": round(float(self.confidence), 4),
            "duration_ms": round(float(self.duration_ms), 3),
        }


@dataclass
class AgentTrace:
    """The ordered record of one agent run."""

    goal: str
    steps: list[AgentStep] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    outcome: str = "running"
    summary: str = ""

    def add(
        self,
        *,
        phase: str,
        action: str,
        rationale: str,
        decided_by: str,
        observation: dict | None = None,
        args: dict | None = None,
        result: dict | None = None,
        confidence: float = 0.0,
        duration_ms: float = 0.0,
    ) -> AgentStep:
        """Append a step and return it.

        The index is assigned here rather than by the caller so the sequence can
        never develop gaps or duplicates, whatever order the phases run in.
        """
        if phase not in PHASES:
            raise ValueError(f"unknown phase {phase!r}; expected one of {PHASES}")
        step = AgentStep(
            index=len(self.steps) + 1,
            phase=phase,
            action=action,
            rationale=rationale,
            decided_by=decided_by,
            observation=observation or {},
            args=args or {},
            result=result or {},
            confidence=confidence,
            duration_ms=duration_ms,
        )
        self.steps.append(step)
        return step

    # --- Read-only views -------------------------------------------------

    @property
    def decisions(self) -> list[AgentStep]:
        return [s for s in self.steps if s.phase == "decide"]

    @property
    def actions(self) -> list[AgentStep]:
        return [s for s in self.steps if s.phase == "act"]

    @property
    def brains_used(self) -> list[str]:
        """Every brain that made a decision, in order of first use.

        A run that fell back mid-way shows both, which is exactly the thing a
        reader should be able to notice.
        """
        seen: list[str] = []
        for s in self.decisions:
            if s.decided_by not in seen:
                seen.append(s.decided_by)
        return seen

    @property
    def used_fallback(self) -> bool:
        return any("policy" in b or "fallback" in b for b in self.brains_used)

    def elapsed_s(self) -> float:
        return time.time() - self.started_at

    # --- Serialisation ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "outcome": self.outcome,
            "summary": self.summary,
            "brains_used": self.brains_used,
            "used_fallback": self.used_fallback,
            "elapsed_s": round(self.elapsed_s(), 3),
            "step_count": len(self.steps),
            "decision_count": len(self.decisions),
            "steps": [s.to_dict() for s in self.steps],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    def render(self) -> str:
        """A compact human-readable rendering, for a terminal or a test failure."""
        lines = [f"goal: {self.goal}", f"outcome: {self.outcome}"]
        for s in self.steps:
            mark = {"observe": "o", "decide": "?", "act": ">", "gate": "|", "finish": "#"}[s.phase]
            line = f"  {mark} [{s.index:02d}] {s.action}"
            if s.phase == "decide":
                line += f"  ({s.decided_by}, conf {s.confidence:.2f})"
            lines.append(line)
            if s.rationale:
                lines.append(f"        {s.rationale}")
        if self.summary:
            lines.append(f"summary: {self.summary}")
        return "\n".join(lines)
