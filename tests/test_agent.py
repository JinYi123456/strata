"""Unit tests for the agent layer: trace, actions, policy, and brains.

These are all fast - no pipeline, no models, no images. The agent's decision
logic is pure functions over a frozen observation, so it can be tested exactly,
which is the point of keeping the model out of it.

The loop's end-to-end behaviour is covered separately in
``test_agent_loop.py``, which is marked ``slow``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vision"))

from strata_agent.actions import (  # noqa: E402
    ABORT,
    ACCEPT_SCAN,
    ACTION_CATALOGUE,
    ALL_ACTIONS,
    FINALIZE,
    MARK_UNCERTAIN,
    REQUEST_VIEW,
    Action,
    Observation,
    describe,
    parse_action,
    requires_approval,
)
from strata_agent.brain import (  # noqa: E402
    PROVIDERS,
    FallbackBrain,
    PolicyBrain,
    _extract_json,
    _redact,
    make_brain,
)
from strata_agent.policy import AgentConfig, PolicyState, decide  # noqa: E402
from strata_agent.trace import AgentTrace  # noqa: E402


def obs(**kw) -> Observation:
    """A healthy observation, with fields overridable per test."""
    base = dict(
        registration_confidence=0.90,
        registration_model="homography",
        registration_error_px=0.8,
        coverage_status="observed",
        observed_fraction=0.85,
        frame_count=1,
        view_requests=[],
        event_count=3,
        uncertain_count=0,
        low_confidence_count=0,
        mean_confidence=0.8,
        kind_counts={"missing": 1, "added": 1, "state_changed": 1},
        uncertain_targets=[],
        captures_used=0,
        capture_budget=3,
    )
    base.update(kw)
    return Observation(**base)


class TestActionCatalogue:
    def test_every_action_is_described(self):
        for name in ALL_ACTIONS:
            assert describe(name), f"{name} has no description for the model"

    def test_only_finalize_requires_approval(self):
        # This is the safety property the whole gate rests on. If another action
        # ever becomes state-changing, this test should fail and force a
        # conscious decision rather than a silent one.
        approving = [n for n in ALL_ACTIONS if requires_approval(n)]
        assert approving == [FINALIZE]

    def test_unknown_action_does_not_require_approval_lookup_to_crash(self):
        assert requires_approval("not_an_action") is False


class TestParseAction:
    def test_parses_a_well_formed_payload(self):
        a = parse_action(
            {"name": REQUEST_VIEW, "args": {"direction": "pan left"},
             "rationale": "coverage is low", "confidence": 0.7}
        )
        assert a.name == REQUEST_VIEW
        assert a.args["direction"] == "pan left"
        assert a.confidence == pytest.approx(0.7)

    def test_accepts_the_action_key_as_an_alias(self):
        assert parse_action({"action": ACCEPT_SCAN, "rationale": "fine"}).name == ACCEPT_SCAN

    def test_rejects_an_unknown_action(self):
        # Untrusted input: an unknown name must be refused, not defaulted.
        with pytest.raises(ValueError):
            parse_action({"name": "delete_everything", "rationale": "x"})

    def test_rejects_a_non_mapping(self):
        with pytest.raises(ValueError):
            parse_action(["not", "a", "mapping"])  # type: ignore[arg-type]

    def test_rejects_non_mapping_args(self):
        with pytest.raises(ValueError):
            parse_action({"name": ACCEPT_SCAN, "args": "oops"})

    def test_confidence_is_clamped(self):
        assert parse_action({"name": ACCEPT_SCAN, "confidence": 5.0}).confidence == 1.0
        assert parse_action({"name": ACCEPT_SCAN, "confidence": -3}).confidence == 0.0

    def test_a_non_numeric_confidence_becomes_zero(self):
        assert parse_action({"name": ACCEPT_SCAN, "confidence": "very"}).confidence == 0.0

    def test_rationale_is_length_capped(self):
        a = parse_action({"name": ACCEPT_SCAN, "rationale": "x" * 5000})
        assert len(a.rationale) <= 600

    def test_missing_rationale_is_allowed_here(self):
        # The parser does not require a rationale; the *model brain* does, because
        # a decision without a reason is not auditable. The parser stays
        # permissive so the fallback path can construct actions freely.
        assert parse_action({"name": ACCEPT_SCAN}).rationale == ""


class TestObservation:
    def test_budget_remaining_is_derived(self):
        assert obs(captures_used=1, capture_budget=3).budget_remaining == 2

    def test_budget_remaining_never_goes_negative(self):
        assert obs(captures_used=9, capture_budget=3).budget_remaining == 0

    def test_coverage_adequacy_tracks_status(self):
        assert obs(coverage_status="observed").coverage_is_adequate is True
        assert obs(coverage_status="weak").coverage_is_adequate is False

    def test_is_frozen(self):
        o = obs()
        with pytest.raises(Exception):
            o.event_count = 99  # type: ignore[misc]

    def test_to_dict_is_json_safe(self):
        json.dumps(obs().to_dict())


class TestTrace:
    def test_indices_are_sequential_from_one(self):
        t = AgentTrace(goal="g")
        for i in range(4):
            t.add(phase="observe", action="scan", rationale="", decided_by="pipeline")
        assert [s.index for s in t.steps] == [1, 2, 3, 4]

    def test_rejects_an_unknown_phase(self):
        t = AgentTrace(goal="g")
        with pytest.raises(ValueError):
            t.add(phase="nonsense", action="x", rationale="", decided_by="y")

    def test_decisions_and_actions_are_separable(self):
        t = AgentTrace(goal="g")
        t.add(phase="observe", action="scan", rationale="", decided_by="pipeline")
        t.add(phase="decide", action=ACCEPT_SCAN, rationale="", decided_by="policy-rules")
        t.add(phase="act", action=ACCEPT_SCAN, rationale="", decided_by="loop")
        assert len(t.decisions) == 1
        assert len(t.actions) == 1

    def test_brains_used_is_deduplicated_in_order(self):
        t = AgentTrace(goal="g")
        for who in ("groq:x", "groq:x", "policy-rules"):
            t.add(phase="decide", action=ACCEPT_SCAN, rationale="", decided_by=who)
        assert t.brains_used == ["groq:x", "policy-rules"]

    def test_fallback_is_detected(self):
        t = AgentTrace(goal="g")
        t.add(phase="decide", action=ACCEPT_SCAN, rationale="",
              decided_by="policy-rules (fallback)")
        assert t.used_fallback is True

    def test_a_clean_run_is_not_flagged_as_fallback(self):
        t = AgentTrace(goal="g")
        t.add(phase="decide", action=ACCEPT_SCAN, rationale="", decided_by="groq:x")
        assert t.used_fallback is False

    def test_to_json_round_trips(self):
        t = AgentTrace(goal="g")
        t.add(phase="decide", action=ACCEPT_SCAN, rationale="fine", decided_by="policy-rules")
        payload = json.loads(t.to_json())
        assert payload["goal"] == "g"
        assert len(payload["steps"]) == 1

    def test_render_includes_rationales(self):
        t = AgentTrace(goal="g")
        t.add(phase="decide", action=ACCEPT_SCAN, rationale="because reasons",
              decided_by="policy-rules")
        assert "because reasons" in t.render()


class TestPolicyRules:
    """One test per rule, in the order the rules are evaluated."""

    def test_bad_registration_aborts(self):
        a = decide(obs(registration_confidence=0.10), AgentConfig())
        assert a.name == ABORT
        assert "0.10" in a.rationale

    def test_weak_registration_asks_for_a_wider_view(self):
        a = decide(obs(registration_confidence=0.45), AgentConfig())
        assert a.name == REQUEST_VIEW
        assert a.args.get("reason") == "improve registration"

    def test_weak_registration_without_budget_does_not_ask(self):
        a = decide(
            obs(registration_confidence=0.45, captures_used=3, capture_budget=3),
            AgentConfig(),
        )
        assert a.name != REQUEST_VIEW

    def test_an_uncertain_region_is_targeted(self):
        target = {"event_id": "evt-002", "cell": [2, 3], "direction": "pan left",
                  "reason": "the classifier could not decide"}
        a = decide(obs(uncertain_count=1, uncertain_targets=[target]), AgentConfig())
        assert a.name == REQUEST_VIEW
        assert a.args.get("event_id") == "evt-002"
        assert "evt-002" in a.rationale

    def test_an_already_marked_region_is_not_re_targeted(self):
        target = {"event_id": "evt-002", "cell": [2, 3], "direction": "pan left",
                  "reason": "unresolved"}
        state = PolicyState(marked_event_ids=["evt-002"], uncertain_marked=1)
        a = decide(obs(uncertain_count=1, uncertain_targets=[target]), AgentConfig(), state=state)
        assert a.args.get("event_id") is None

    def test_poor_coverage_asks_for_a_view(self):
        a = decide(
            obs(observed_fraction=0.3, coverage_status="weak",
                view_requests=[{"direction": "pan right", "target_cell": [3, 11],
                                "reason": "under-observed"}]),
            AgentConfig(),
        )
        assert a.name == REQUEST_VIEW
        assert a.args.get("reason") == "improve coverage"

    def test_poor_coverage_without_budget_records_the_shortfall(self):
        a = decide(
            obs(observed_fraction=0.3, coverage_status="weak",
                captures_used=3, capture_budget=3),
            AgentConfig(),
        )
        assert a.name == MARK_UNCERTAIN
        assert "absence of evidence" in a.rationale

    def test_the_shortfall_is_recorded_only_once(self):
        o = obs(observed_fraction=0.3, coverage_status="weak",
                captures_used=3, capture_budget=3)
        state = PolicyState(coverage_shortfall_noted=True, uncertain_marked=1)
        a = decide(o, AgentConfig(), state=state)
        assert a.name == ACCEPT_SCAN

    def test_a_healthy_scan_is_accepted(self):
        a = decide(obs(), AgentConfig())
        assert a.name == ACCEPT_SCAN

    def test_accept_rationale_mentions_a_coverage_shortfall(self):
        state = PolicyState(coverage_shortfall_noted=True, uncertain_marked=1)
        a = decide(obs(observed_fraction=0.5, coverage_status="weak"), AgentConfig(), state=state)
        assert a.name == ACCEPT_SCAN
        assert "never observed" in a.rationale

    def test_an_accepted_scan_is_finalized(self):
        a = decide(obs(), AgentConfig(), state=PolicyState(accepted=True))
        assert a.name == FINALIZE
        assert a.requires_approval is True

    def test_finalize_is_refused_on_marginal_registration(self):
        state = PolicyState(accepted=True)
        a = decide(obs(registration_confidence=0.40), AgentConfig(), state=state)
        assert a.name == ABORT

    def test_no_captures_available_stops_asking(self):
        # The observed defect: without this, the agent asked five times.
        state = PolicyState(views_available=False)
        o = obs(observed_fraction=0.3, coverage_status="weak",
                view_requests=[{"direction": "pan right", "target_cell": [1, 1]}])
        a = decide(o, AgentConfig(), state=state)
        assert a.name != REQUEST_VIEW

    def test_policy_is_deterministic(self):
        o = obs(observed_fraction=0.3, coverage_status="weak")
        cfg = AgentConfig()
        first = decide(o, cfg)
        for _ in range(5):
            assert decide(o, cfg).name == first.name

    def test_every_action_carries_a_rationale(self):
        cases = [
            obs(registration_confidence=0.1),
            obs(registration_confidence=0.45),
            obs(observed_fraction=0.2, coverage_status="weak"),
            obs(),
            obs(),
        ]
        states = [None, None, None, None, PolicyState(accepted=True)]
        for o, s in zip(cases, states):
            a = decide(o, AgentConfig(), state=s)
            assert a.rationale.strip(), f"{a.name} produced no rationale"

    def test_action_is_always_in_the_catalogue(self):
        for o, s in [
            (obs(registration_confidence=0.05), None),
            (obs(observed_fraction=0.1, coverage_status="weak"), None),
            (obs(), None),
            (obs(), PolicyState(accepted=True)),
        ]:
            assert decide(o, AgentConfig(), state=s).name in ACTION_CATALOGUE


class TestPolicyBrain:
    def test_name_is_stable(self):
        assert PolicyBrain().name == "policy-rules"

    def test_returns_a_brain_result(self):
        r = PolicyBrain().decide(obs(), AgentConfig())
        assert r.action.name == ACCEPT_SCAN
        assert r.decided_by == "policy-rules"
        assert r.degraded is False

    def test_raises_if_the_caller_forbids_the_chosen_action(self):
        # A caller whose constraint contradicts the rules has a bug; failing
        # loudly beats silently substituting a different action.
        with pytest.raises(RuntimeError):
            PolicyBrain().decide(obs(), AgentConfig(), allowed=(REQUEST_VIEW,))


class _BoomBrain:
    name = "boom"

    def decide(self, obs, cfg, *, allowed=ALL_ACTIONS, state=None):
        raise RuntimeError("provider exploded with token Bearer sk-abcdef1234567890")


class _BadShapeBrain:
    name = "bad-shape"

    def decide(self, obs, cfg, *, allowed=ALL_ACTIONS, state=None):
        from strata_agent.brain import BrainResult

        return BrainResult(action=Action(name="not_real"), decided_by="bad-shape")


class TestFallbackBrain:
    def test_falls_back_on_exception(self):
        fb = FallbackBrain(_BoomBrain(), PolicyBrain())
        r = fb.decide(obs(), AgentConfig())
        assert r.action.name == ACCEPT_SCAN
        assert "fallback" in r.decided_by
        assert r.degraded is True

    def test_records_the_error(self):
        fb = FallbackBrain(_BoomBrain(), PolicyBrain())
        r = fb.decide(obs(), AgentConfig())
        assert "provider exploded" in r.error

    def test_redacts_credentials_from_the_error(self):
        fb = FallbackBrain(_BoomBrain(), PolicyBrain())
        r = fb.decide(obs(), AgentConfig())
        assert "sk-abcdef1234567890" not in r.error
        assert "redacted" in r.error.lower()

    def test_passes_through_a_successful_primary(self):
        fb = FallbackBrain(PolicyBrain(), _BoomBrain())
        r = fb.decide(obs(), AgentConfig())
        assert r.degraded is False

    def test_does_not_swallow_a_successful_result(self):
        fb = FallbackBrain(PolicyBrain(), PolicyBrain())
        r = fb.decide(obs(), AgentConfig())
        assert "fallback" not in r.decided_by


class TestExtractJson:
    def test_plain_object(self):
        assert _extract_json('{"a": 1}') == {"a": 1}

    def test_fenced_block(self):
        assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_with_a_preamble_and_trailing_text(self):
        text = 'Sure! Here you go:\n{"a": 1, "b": {"c": 2}}\nHope that helps.'
        assert _extract_json(text) == {"a": 1, "b": {"c": 2}}

    def test_nested_braces_are_balanced(self):
        assert _extract_json('{"a": {"b": {"c": 1}}}') == {"a": {"b": {"c": 1}}}

    def test_no_object_raises(self):
        with pytest.raises(ValueError):
            _extract_json("I cannot help with that.")

    def test_unterminated_object_raises(self):
        with pytest.raises(ValueError):
            _extract_json('{"a": 1')

    def test_malformed_json_raises_rather_than_being_repaired(self):
        # A parser, not a repairer: guessing at intent here would turn a failed
        # response into a silent wrong decision.
        with pytest.raises(json.JSONDecodeError):
            _extract_json('{"a": }')


class TestRedact:
    def test_bearer_token_is_removed(self):
        assert "abcdef123456" not in _redact("failed with Bearer abcdef1234567890xyz")

    def test_common_key_prefixes_are_removed(self):
        for key in ("sk-abcdefghijklmnop", "gsk_abcdefghijklmnop", "AIzaabcdefghijklmnop"):
            assert key not in _redact(f"auth failed: {key}")

    def test_ordinary_text_survives(self):
        assert "connection refused" in _redact("connection refused")

    def test_length_is_capped(self):
        assert len(_redact("x" * 5000)) <= 240


class TestMakeBrain:
    def test_returns_the_policy_when_nothing_is_configured(self, monkeypatch):
        for spec in PROVIDERS.values():
            monkeypatch.delenv(spec["key_env"], raising=False)
        brain = make_brain()
        assert isinstance(brain, PolicyBrain)

    def test_unknown_provider_falls_back_to_the_policy(self, monkeypatch):
        for spec in PROVIDERS.values():
            monkeypatch.delenv(spec["key_env"], raising=False)
        assert isinstance(make_brain("not-a-provider"), PolicyBrain)

    def test_returns_a_fallback_wrapper_when_a_key_exists(self, monkeypatch):
        for spec in PROVIDERS.values():
            monkeypatch.delenv(spec["key_env"], raising=False)
        monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
        brain = make_brain("groq")
        # The model brain is wrapped so any failure degrades to the rules.
        assert isinstance(brain, FallbackBrain)
        assert isinstance(brain.fallback, PolicyBrain)

    def test_the_wrapped_brain_still_works_without_network(self, monkeypatch):
        # The model call will fail (no real key / no network), and the run must
        # still produce a rule-based decision rather than raising.
        for spec in PROVIDERS.values():
            monkeypatch.delenv(spec["key_env"], raising=False)
        monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
        r = make_brain("groq").decide(obs(), AgentConfig())
        assert r.action.name in ACTION_CATALOGUE

    def test_provider_registry_entries_are_complete(self):
        for name, spec in PROVIDERS.items():
            assert spec.get("base_url"), f"{name} has no base_url"
            assert spec.get("key_env"), f"{name} has no key_env"
            assert spec.get("model"), f"{name} has no default model"
