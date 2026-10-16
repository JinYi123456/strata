"""The agent's decision brains.

There is exactly **one** place in STRATA where a language model is used, and this
is it. Everything else - registration, change detection, event typing, coverage,
and the decision rules themselves - is deterministic. That is a deliberate
architectural choice rather than a limitation, and the reasoning is worth
stating because it shapes the whole module.

A vision pipeline has to be reproducible. If the same two photographs can
produce different change sets on different runs, the system cannot be trusted
about anything it reports, and no amount of explanation fixes that. So the model
is confined to the one task where reproducibility is not the point: choosing
which action to take next, and explaining that choice in a sentence a person can
read.

What that buys:

* The pipeline's output is identical run to run, whether or not a model is
  configured.
* The model's contribution is visible and bounded - it appears in the trace as a
  decision with a rationale, not as a transformation of the evidence.
* A model failure is a non-event. Every brain here is wrapped so that any
  exception, timeout, malformed response, or disallowed action degrades to the
  deterministic policy. There is no code path where an API problem produces a
  wrong answer instead of a rule-based one.

The model is also never asked to invent facts. It is given a structured
observation and a closed list of actions, and asked to pick one. It cannot
report a change that was not detected, because reporting is not an action it can
take.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from .actions import (
    ALL_ACTIONS,
    Action,
    Observation,
    describe,
    parse_action,
)
from .policy import AgentConfig, PolicyState, decide as policy_decide

log = logging.getLogger("strata.agent.brain")


@dataclass
class BrainResult:
    """An action together with the provenance of the decision."""

    action: Action
    decided_by: str
    error: str = ""

    @property
    def degraded(self) -> bool:
        return bool(self.error)


class Brain(Protocol):
    """Anything that can choose an action from an observation."""

    name: str

    def decide(
        self,
        obs: Observation,
        cfg: AgentConfig,
        *,
        allowed: Sequence[str] = ALL_ACTIONS,
        state: PolicyState | None = None,
    ) -> BrainResult:  # pragma: no cover - protocol
        ...


# --- The reference brain -------------------------------------------------


class PolicyBrain:
    """The deterministic rules, exposed as a brain.

    This is not a fallback in the sense of being second-best. It is the
    reference implementation: the behaviour every other brain is measured
    against, and the one that defines what the agent does when nothing else is
    available.
    """

    name = "policy-rules"

    def decide(self, obs, cfg, *, allowed=ALL_ACTIONS, state=None) -> BrainResult:
        action = policy_decide(obs, cfg, state=state)
        if action.name not in allowed:
            # The rules should never propose something the caller has forbidden.
            # If they do, that is a bug in the caller's constraint, so fail
            # loudly here rather than silently substituting.
            raise RuntimeError(
                f"policy proposed {action.name!r}, which is not in the allowed set "
                f"{tuple(allowed)!r}"
            )
        return BrainResult(action=action, decided_by=self.name)


# --- Known model providers ----------------------------------------------
#
# All of these speak the OpenAI chat-completions shape, so one client covers
# them. The registry exists so a provider can be selected by name in
# configuration without hard-coding URLs at the call site, and so the default
# model for each is stated in one place.
#
# Keys are read from the environment only. They are never written to config
# files, never logged, and never included in the trace - see _redact below.

PROVIDERS: dict[str, dict[str, str]] = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "model": "llama-3.3-70b-versatile",
    },
    "featherless": {
        "base_url": "https://api.featherless.ai/v1",
        "key_env": "FEATHERLESS_API_KEY",
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "key_env": "GEMINI_API_KEY",
        "model": "gemini-2.0-flash",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "model": "meta-llama/llama-3.3-70b-instruct:free",
    },
}

SYSTEM_PROMPT = """\
You are the decision layer of a spatial change-detection system.

A deterministic pipeline has already done all measurement. It registered two \
captures of the same place, detected where they differ, and typed each change. \
You are given the resulting state. You do not measure anything, and you must not \
speculate about changes that were not reported.

Your only job is to choose the single best next action from the list provided.

How to choose:
- If the captures cannot be aligned reliably, stop. Reporting changes from a bad \
alignment is worse than reporting nothing.
- If a region is unresolved and you still have capture budget, ask for one more \
capture aimed at it. That is almost always better than guessing.
- If coverage is adequate and nothing is outstanding, accept the scan.
- Only finalize once the scan has been accepted.
- Declaring a region unresolvable is a legitimate, honest outcome. Prefer it to \
pretending an area was checked.

Reply with a single JSON object and nothing else:
{"name": "<action>", "args": {...}, "rationale": "<one or two sentences>", "confidence": <0.0-1.0>}

The rationale is read by a person deciding whether to trust this. Write it in \
plain language, state the specific numbers you relied on, and do not pad it."""


class OpenAICompatBrain:
    """A model brain speaking the OpenAI chat-completions protocol.

    Deliberately thin. It builds a prompt, makes one request, parses one JSON
    object, and validates it against the action vocabulary. There is no
    retry-with-repair loop and no multi-turn negotiation, because both would
    make the agent's latency and behaviour harder to reason about, and the
    deterministic rules already provide a perfectly good answer when the model
    does not.

    ``httpx`` is imported lazily so that the package can be imported, and the
    whole agent exercised, with no HTTP client installed at all.
    """

    def __init__(
        self,
        *,
        provider: str = "groq",
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_s: float = 12.0,
    ) -> None:
        if provider not in PROVIDERS and base_url is None:
            raise ValueError(
                f"unknown provider {provider!r}; known: {sorted(PROVIDERS)}"
            )
        spec = PROVIDERS.get(provider, {})
        self.provider = provider
        self.base_url = (base_url or spec.get("base_url", "")).rstrip("/")
        self.model = model or spec.get("model", "")
        self._api_key = api_key or os.environ.get(spec.get("key_env", ""), "")
        self.timeout_s = timeout_s
        self.name = f"{provider}:{self.model}" if self.model else provider

    @property
    def available(self) -> bool:
        """Whether this brain can be called at all.

        Checked before every use so a missing key is treated as "use the rules"
        rather than as an error. A system that refuses to run because an
        optional component is unconfigured is worse than one that runs without
        it.
        """
        return bool(self.base_url and self.model and self._api_key)

    def decide(self, obs, cfg, *, allowed=ALL_ACTIONS, state=None) -> BrainResult:
        if not self.available:
            raise RuntimeError(
                f"{self.provider} brain is not configured "
                f"(needs base_url, model, and an API key)"
            )
        payload = self._build_payload(obs, cfg, allowed)
        text = self._post(payload)
        action = self._parse(text, allowed)
        return BrainResult(action=action, decided_by=self.name)

    # --- Internals -------------------------------------------------------

    def _build_payload(self, obs: Observation, cfg: AgentConfig, allowed: Sequence[str]) -> dict:
        menu = "\n".join(f"- {name}: {describe(name)}" for name in allowed)
        user = (
            "Current state:\n"
            f"{json.dumps(obs.to_dict(), indent=2)}\n\n"
            "Constraints:\n"
            f"- capture budget remaining: {obs.budget_remaining}\n"
            f"- you may choose only from: {', '.join(allowed)}\n\n"
            "Available actions:\n"
            f"{menu}\n\n"
            "Choose the single best next action."
        )
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "max_tokens": 320,
            "response_format": {"type": "json_object"},
        }

    def _post(self, payload: dict) -> str:
        import httpx  # lazy: keeps the package importable without an HTTP client

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.post(
                f"{self.base_url}/chat/completions", headers=headers, json=payload
            )
            resp.raise_for_status()
            data = resp.json()
        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"unexpected response shape from {self.provider}") from exc

    def _parse(self, text: str, allowed: Sequence[str]) -> Action:
        action = parse_action(_extract_json(text))
        if action.name not in allowed:
            raise ValueError(
                f"model chose {action.name!r}, which is not permitted here; "
                f"allowed: {tuple(allowed)!r}"
            )
        if not action.rationale:
            # A decision without a reason is not auditable, and the trace is the
            # product here. Treat it as a failed response rather than accepting a
            # silent decision.
            raise ValueError("model returned an action with no rationale")
        return action


class FallbackBrain:
    """Runs a primary brain, and falls back to a second on any failure.

    The fallback is unconditional - it catches every exception, not a curated
    list of expected ones. That is intentional: the failure modes of an external
    API are not enumerable (rate limits, DNS, TLS, a provider shipping a new
    response shape, a model returning prose instead of JSON), and a missed
    exception here would mean the agent crashes mid-run in front of a user. The
    only thing that matters is that a failure produces a rule-based decision and
    is *recorded* rather than hidden.

    The error is carried in :attr:`BrainResult.error` and surfaces in the trace,
    so a degraded run is visible to anyone reading it.
    """

    def __init__(self, primary: Brain, fallback: Brain, *, label: str | None = None) -> None:
        self.primary = primary
        self.fallback = fallback
        self.name = label or f"{primary.name}->{fallback.name}"

    def decide(self, obs, cfg, *, allowed=ALL_ACTIONS, state=None) -> BrainResult:
        try:
            result = self.primary.decide(obs, cfg, allowed=allowed, state=state)
            return result
        except Exception as exc:  # noqa: BLE001 - see the class docstring
            reason = _redact(str(exc))
            log.warning("primary brain %s failed: %s", self.primary.name, reason)
            result = self.fallback.decide(obs, cfg, allowed=allowed, state=state)
            return BrainResult(
                action=result.action,
                decided_by=f"{result.decided_by} (fallback)",
                error=reason,
            )


# --- Helpers -------------------------------------------------------------


def _extract_json(text: str) -> dict:
    """Pull one JSON object out of a model's reply.

    Models wrap JSON in prose or code fences often enough that refusing anything
    but a bare object would make the brain fail constantly for no good reason.
    The first balanced ``{...}`` span is taken, which handles a fenced block, a
    preamble, and a trailing explanation alike.

    This is a *parser*, not a repairer: it does not fix malformed JSON, invent
    missing fields, or guess at intent. If what is inside the braces is not valid
    JSON, that is a failed response and the caller falls back.
    """
    text = text.strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in model response")
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("unterminated JSON object in model response")


def _redact(message: str, *, limit: int = 240) -> str:
    """Trim an error message for the trace.

    Error strings are the most likely place for a credential to leak into a
    record - an HTTP client quoting a request, a provider echoing a header - so
    anything resembling a key is replaced before the message is stored or logged.

    The separator after the prefix is optional because providers disagree about
    it: OpenAI uses ``sk-``, Groq ``gsk_``, Hugging Face ``hf_``, and Google's
    ``AIza`` keys have no separator at all. An earlier version of this function
    required an underscore and therefore redacted Groq and Hugging Face keys
    while letting OpenAI's through - which a test caught, and which is exactly
    the kind of bug that looks harmless until a key ends up in a shared log.

    A bare ``api`` prefix is deliberately *not* matched. It redacted realistic
    diagnostics such as ``api_key_missing_from_config``, turning a useful error
    into an opaque one, while adding nothing for the providers actually
    configured here. Over-redaction is the safer error, but only when it does
    not cost the information the message exists to carry.
    """
    import re

    cleaned = re.sub(r"(?i)bearer\s+[A-Za-z0-9._\-]{8,}", "Bearer <redacted>", message)
    cleaned = re.sub(
        r"\b(?:sk|gsk|hf|AIza)[-_]?[A-Za-z0-9._\-]{8,}", "<redacted>", cleaned
    )
    cleaned = " ".join(cleaned.split())
    return cleaned[:limit]


def make_brain(
    provider: str | None = None,
    *,
    model: str | None = None,
    api_key: str | None = None,
    cfg: AgentConfig | None = None,
) -> Brain:
    """Build the best available brain, always returning something usable.

    Selection order is by *reliability of the free tier*, not by model quality,
    because a marginally better answer that arrives late or not at all is worse
    than a good answer that arrives every time. If nothing is configured, this
    returns the deterministic policy rather than raising - callers never have to
    handle the no-key case.
    """
    policy = PolicyBrain()
    if provider is None:
        provider = _first_configured_provider()
    if provider is None:
        return policy

    try:
        model_brain = OpenAICompatBrain(
            provider=provider, model=model, api_key=api_key
        )
    except ValueError:
        return policy

    if not model_brain.available:
        return policy
    return FallbackBrain(model_brain, policy)


def _first_configured_provider() -> str | None:
    """The first provider in the registry with a key present in the environment."""
    for name, spec in PROVIDERS.items():
        if os.environ.get(spec.get("key_env", "")):
            return name
    return None
