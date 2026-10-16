"""STRATA's agent layer: active perception with a legible reasoning trace.

The pipeline in :mod:`strata_core` answers "what changed". This package answers
"and what should we do about it" - whether to look again, from where, whether to
give up on a region, and when the evidence is good enough to commit.

The design principle throughout is that the agent's reasoning is a *product*,
not an implementation detail. Every decision is recorded with the observation it
saw and a rationale a person can read, because a system that changes the record
of what a place looked like has to be able to justify each change.

Public surface:

* :func:`run_agent` - run the perceive-decide-act loop to completion.
* :class:`AgentRun` / :class:`AgentTrace` - the result and its reasoning.
* :class:`Observation` / :class:`Action` - the decision interface.
* :class:`AgentConfig` - the decision thresholds.
* :func:`make_brain` - build the best available decision maker.
"""

from __future__ import annotations

from .actions import (
    ABORT,
    ACCEPT_SCAN,
    ACTION_CATALOGUE,
    ALL_ACTIONS,
    FINALIZE,
    MARK_UNCERTAIN,
    REQUEST_VIEW,
    Action,
    Observation,
)
from .brain import (
    PROVIDERS,
    Brain,
    BrainResult,
    FallbackBrain,
    OpenAICompatBrain,
    PolicyBrain,
    make_brain,
)
from .loop import (
    AgentRun,
    FrameProvider,
    allowed_actions_for,
    build_observation,
    run_agent,
)
from .policy import AgentConfig, PolicyState
from .trace import AgentStep, AgentTrace

__all__ = [
    # entry point
    "run_agent",
    "build_observation",
    "allowed_actions_for",
    # results
    "AgentRun",
    "AgentTrace",
    "AgentStep",
    # decision interface
    "Observation",
    "Action",
    "AgentConfig",
    "PolicyState",
    # actions
    "ACTION_CATALOGUE",
    "ALL_ACTIONS",
    "REQUEST_VIEW",
    "MARK_UNCERTAIN",
    "ACCEPT_SCAN",
    "FINALIZE",
    "ABORT",
    # brains
    "Brain",
    "BrainResult",
    "PolicyBrain",
    "OpenAICompatBrain",
    "FallbackBrain",
    "make_brain",
    "PROVIDERS",
    "FrameProvider",
]
