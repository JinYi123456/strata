"""End-to-end tests for the agent loop.

These run the real pipeline, so they are marked ``slow``. Frame sizes are kept
small to keep them tractable.

The behaviours asserted here are the ones that make the agent trustworthy rather
than merely functional: that it terminates, that it never writes without
approval, that a failed brain degrades instead of crashing, and that it does not
repeat a request it cannot satisfy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vision"))

from strata_agent import (  # noqa: E402
    ABORT,
    ACCEPT_SCAN,
    FINALIZE,
    REQUEST_VIEW,
    AgentConfig,
    PolicyBrain,
    run_agent,
)
from strata_core.config import PipelineConfig  # noqa: E402
from strata_core.synth import make_scene  # noqa: E402

pytestmark = pytest.mark.slow

SIZE = (320, 240)


def scene_pair(seed: int = 1000):
    sc = make_scene(seed=seed, size=SIZE, n_objects=8, n_mutations=4)
    return sc.baseline, sc.rescan


class TestBasicRun:
    def test_runs_to_a_terminal_outcome(self):
        base, res = scene_pair()
        run = run_agent([base], [res], brain=PolicyBrain())
        assert run.outcome in {"finalized", "declined", "aborted", "step_limit"}
        assert run.trace.outcome == run.outcome

    def test_produces_a_trace_with_all_phases(self):
        base, res = scene_pair()
        run = run_agent([base], [res], brain=PolicyBrain())
        phases = {s.phase for s in run.trace.steps}
        assert "observe" in phases
        assert "decide" in phases
        assert "act" in phases

    def test_records_events_and_coverage(self):
        base, res = scene_pair()
        run = run_agent([base], [res], brain=PolicyBrain())
        assert run.result is not None
        assert run.coverage is not None
        assert isinstance(run.events, list)

    def test_is_deterministic_with_the_policy_brain(self):
        base, res = scene_pair()
        a = run_agent([base], [res], brain=PolicyBrain())
        b = run_agent([base], [res], brain=PolicyBrain())
        assert a.outcome == b.outcome
        assert [(s.action, s.phase) for s in a.trace.steps] == [
            (s.action, s.phase) for s in b.trace.steps
        ]

    def test_trace_serialises(self):
        import json

        base, res = scene_pair()
        run = run_agent([base], [res], brain=PolicyBrain())
        json.dumps(run.to_dict())


class TestApprovalGate:
    def test_without_an_approver_nothing_is_committed(self):
        # Silence is not permission.
        base, res = scene_pair()
        run = run_agent([base], [res], brain=PolicyBrain(), approver=None)
        assert run.finalized is False
        assert run.outcome == "declined"

    def test_a_declining_approver_blocks_the_write(self):
        base, res = scene_pair()
        run = run_agent([base], [res], brain=PolicyBrain(), approver=lambda a: False)
        assert run.finalized is False
        assert run.outcome == "declined"

    def test_an_approving_approver_commits(self):
        base, res = scene_pair()
        run = run_agent([base], [res], brain=PolicyBrain(), approver=lambda a: True)
        assert run.finalized is True
        assert run.outcome == "finalized"

    def test_the_approver_is_only_asked_about_state_changing_actions(self):
        base, res = scene_pair()
        seen: list[str] = []

        def approver(action):
            seen.append(action.name)
            return True

        run_agent([base], [res], brain=PolicyBrain(), approver=approver)
        # The gate exists to guard the write, so it must only ever be consulted
        # for the action that writes.
        assert seen == [FINALIZE] if seen else True

    def test_the_gate_step_is_recorded(self):
        base, res = scene_pair()
        run = run_agent([base], [res], brain=PolicyBrain(), approver=lambda a: True)
        gate = [s for s in run.trace.steps if s.phase == "gate"]
        assert len(gate) == 1
        assert gate[0].result.get("approved") is True
        assert gate[0].decided_by == "human"

    def test_a_withheld_gate_is_recorded_as_such(self):
        base, res = scene_pair()
        run = run_agent([base], [res], brain=PolicyBrain(), approver=lambda a: False)
        gate = [s for s in run.trace.steps if s.phase == "gate"]
        assert gate[0].result.get("approved") is False


class TestAbortPath:
    def test_an_unregistrable_pair_is_aborted(self):
        # A blank baseline against a textured rescan cannot be aligned, and the
        # agent should refuse rather than report differences that are really
        # alignment error.
        base, res = scene_pair()
        blank = np.full_like(base, 127)
        run = run_agent([blank], [res], brain=PolicyBrain())
        assert run.outcome == "aborted"

    def test_the_abort_rationale_explains_why(self):
        base, res = scene_pair()
        blank = np.full_like(base, 127)
        run = run_agent([blank], [res], brain=PolicyBrain())
        decisions = run.trace.decisions
        assert decisions
        assert "registration confidence" in decisions[0].rationale.lower()


class TestViewRequests:
    def test_a_satisfied_request_adds_a_capture(self):
        base, res = scene_pair()
        supplied = 0

        def provider(direction, cell, reason):
            nonlocal supplied
            supplied += 1
            return res  # a real provider would return a genuinely new frame

        run = run_agent([base], [res], brain=PolicyBrain(), frame_provider=provider)
        acts = [s for s in run.trace.steps if s.phase == "act" and s.action == REQUEST_VIEW]
        assert acts, "the agent should have asked for a capture"
        assert any(a.result.get("satisfied") for a in acts)

    def test_an_unsatisfiable_request_is_recorded_and_not_repeated(self):
        # The observed defect this guards against: the loop asked five times.
        base, res = scene_pair()
        run = run_agent([base], [res], brain=PolicyBrain(), frame_provider=None)
        acts = [
            s for s in run.trace.steps
            if s.phase == "act" and s.action == REQUEST_VIEW
        ]
        assert len(acts) <= 1, f"the request was repeated {len(acts)} times"

    def test_a_declining_provider_is_handled(self):
        base, res = scene_pair()
        run = run_agent(
            [base], [res], brain=PolicyBrain(), frame_provider=lambda *a: None
        )
        acts = [s for s in run.trace.steps if s.action == REQUEST_VIEW]
        assert any(a.result.get("satisfied") is False for a in acts if a.phase == "act")

    def test_the_capture_budget_is_respected(self):
        base, res = scene_pair()
        cfg = AgentConfig(capture_budget=1)

        def provider(direction, cell, reason):
            return res

        run = run_agent(
            [base], [res], brain=PolicyBrain(), agent_config=cfg, frame_provider=provider
        )
        satisfied = [
            s for s in run.trace.steps
            if s.phase == "act" and s.action == REQUEST_VIEW and s.result.get("satisfied")
        ]
        assert len(satisfied) <= 1


class TestRobustness:
    def test_a_failing_brain_degrades_to_the_rules(self):
        from strata_agent.brain import FallbackBrain

        class Boom:
            name = "boom"

            def decide(self, obs, cfg, *, allowed=None, state=None):
                raise RuntimeError("model unavailable")

        base, res = scene_pair()
        run = run_agent(
            [base], [res], brain=FallbackBrain(Boom(), PolicyBrain()), approver=lambda a: True
        )
        # The run must complete on rule-based decisions, and the trace must say so.
        assert run.outcome in {"finalized", "declined"}
        assert run.trace.used_fallback is True
        assert any("fallback" in s.decided_by for s in run.trace.decisions)

    def test_a_totally_broken_brain_ends_the_run_honestly(self):
        class Boom:
            name = "boom"

            def decide(self, obs, cfg, *, allowed=None, state=None):
                raise RuntimeError("no decisions available")

        base, res = scene_pair()
        run = run_agent([base], [res], brain=Boom())
        assert run.outcome == "brain_failed"
        assert run.finalized is False

    def test_the_step_ceiling_is_honoured(self):
        base, res = scene_pair()
        run = run_agent([base], [res], brain=PolicyBrain(), max_steps=2)
        assert run.outcome in {"step_limit", "declined", "finalized", "aborted"}
        assert len(run.trace.steps) <= 2 * 3  # observe + decide + act per iteration

    def test_a_brain_choosing_a_forbidden_action_degrades(self):
        from strata_agent.actions import Action, Observation
        from strata_agent.brain import BrainResult, FallbackBrain

        class Bad:
            name = "bad"

            def decide(self, obs, cfg, *, allowed=None, state=None):
                # Choose something the loop did not offer.
                forbidden = next(
                    a for a in ("request_view", "finalize") if a not in (allowed or ())
                )
                return BrainResult(action=Action(name=forbidden), decided_by="bad")

        base, res = scene_pair()
        # A raw bad brain is not wrapped, so the loop must not accept its choice
        # silently - it either executes a legal action or ends the run.
        run = run_agent([base], [res], brain=FallbackBrain(Bad(), PolicyBrain()))
        assert run.outcome in {"finalized", "declined", "aborted", "step_limit"}


class TestObservationBuilding:
    def test_the_observation_matches_the_result(self):
        from strata_agent.loop import build_observation

        base, res = scene_pair()
        run = run_agent([base], [res], brain=PolicyBrain())
        obs = build_observation(run.result, captures_used=0, capture_budget=3)
        assert obs.event_count == len(run.result.events)
        assert obs.registration_confidence == pytest.approx(
            run.result.registration.confidence, abs=1e-6
        )
        assert 0.0 <= obs.observed_fraction <= 1.0

    def test_uncertain_targets_are_inside_the_grid(self):
        from strata_agent.loop import build_observation

        base, res = scene_pair()
        run = run_agent([base], [res], brain=PolicyBrain())
        obs = build_observation(run.result, captures_used=0, capture_budget=3)
        rows, cols = run.coverage.rows, run.coverage.cols
        for t in obs.uncertain_targets:
            r, c = t["cell"]
            assert 0 <= r < rows
            assert 0 <= c < cols

    def test_uncertain_targets_carry_guidance(self):
        from strata_agent.loop import build_observation

        base, res = scene_pair()
        run = run_agent([base], [res], brain=PolicyBrain())
        obs = build_observation(run.result, captures_used=0, capture_budget=3)
        for t in obs.uncertain_targets:
            assert t["direction"]
            assert t["reason"]

    def test_allowed_actions_shrink_with_the_budget(self):
        from strata_agent.loop import allowed_actions_for
        from strata_agent.policy import PolicyState
        from strata_agent.actions import Observation

        def mk(used, budget):
            return Observation(
                registration_confidence=0.9,
                registration_model="homography",
                registration_error_px=0.5,
                coverage_status="weak",
                observed_fraction=0.3,
                frame_count=1,
                captures_used=used,
                capture_budget=budget,
            )

        with_budget = allowed_actions_for(mk(0, 3), PolicyState())
        without = allowed_actions_for(mk(3, 3), PolicyState())
        assert REQUEST_VIEW in with_budget
        assert REQUEST_VIEW not in without

    def test_finalize_is_only_offered_once_accepted(self):
        from strata_agent.loop import allowed_actions_for
        from strata_agent.policy import PolicyState
        from strata_agent.actions import Observation

        o = Observation(
            registration_confidence=0.9,
            registration_model="homography",
            registration_error_px=0.5,
            coverage_status="observed",
            observed_fraction=0.9,
            frame_count=1,
        )
        assert FINALIZE not in allowed_actions_for(o, PolicyState())
        assert FINALIZE in allowed_actions_for(o, PolicyState(accepted=True))

    def test_only_abort_is_offered_when_registration_fails(self):
        from strata_agent.loop import allowed_actions_for
        from strata_agent.actions import Observation

        o = Observation(
            registration_confidence=0.05,
            registration_model="homography",
            registration_error_px=99.0,
            coverage_status="unobserved",
            observed_fraction=0.0,
            frame_count=1,
        )
        assert allowed_actions_for(o) == (ABORT,)
