"""Running the pipeline and the agent, and shaping the result for the API.

This is the only place the two libraries are wired together, which keeps the
vision core and the agent free of any HTTP concerns and keeps the handlers free
of any vision concerns.

Two decisions worth stating:

* **The agent runs without an approver.** The approval gate is a human moment in
  the UI, not something the server can answer, so the run prepares the change set
  and stops. Its outcome is ``declined`` - nothing was written - and the client
  then asks the operator. That is the honest mapping: the server never pretends
  a write happened.
* **No frame provider.** A mid-run view request needs a human to go and take a
  photograph, which cannot happen inside one request. So the request is recorded,
  the affected targets are marked unresolved, and the guidance is returned to the
  client to act on. The alternative - blocking the request on a capture that will
  never arrive - would hang.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from strata_agent.actions import MARK_UNCERTAIN, Action
from strata_agent.brain import Brain, make_brain
from strata_agent.loop import AgentRun, run_agent
from strata_agent.policy import AgentConfig
from strata_agent.trace import AgentTrace
from strata_core.config import PipelineConfig
from strata_core.coverage import render_guidance
from strata_core.events import ChangeEvent
from strata_core.pipeline import ScanResult

from . import encoding

log = logging.getLogger("strata.service.runners")

# The image panels every scan produces. Named so the client never has to guess
# which is which.
PANEL_BASELINE = "baseline"
PANEL_RESCAN = "rescan"
PANEL_OVERLAY = "overlay"
PANEL_SIDE_BY_SIDE = "side_by_side"
PANEL_GUIDANCE = "guidance"
PANELS = (PANEL_BASELINE, PANEL_RESCAN, PANEL_OVERLAY, PANEL_SIDE_BY_SIDE, PANEL_GUIDANCE)


@dataclass
class ScanOutcome:
    """One executed scan: the agent's run, the pipeline's result, and the panels.

    ``result`` is optional because the loop can legitimately stop before a scan
    exists - an unregistrable pair is aborted, and a failing brain stops the run.
    In both cases the trace is the useful artefact and there are no panels to
    draw, which is a valid outcome rather than an error.
    """

    run: AgentRun
    result: ScanResult | None = None
    panels: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def has_result(self) -> bool:
        return self.result is not None

    @property
    def events(self) -> list[ChangeEvent]:
        if self.run.events:
            return self.run.events
        return list(self.result.events) if self.result is not None else []

    @property
    def uncertain_ids(self) -> set[str]:
        """Events the agent explicitly refused to resolve.

        Read back out of the trace rather than tracked separately: the trace is
        the record of what happened, so deriving from it means the two can never
        disagree.
        """
        return uncertain_ids_from_trace(self.run.trace)


def execute_scan(
    baseline_frames: list[np.ndarray],
    rescan_frames: list[np.ndarray],
    *,
    config: PipelineConfig | None = None,
    agent_config: AgentConfig | None = None,
    brain: Brain | None = None,
    goal: str | None = None,
) -> ScanOutcome:
    """Run the agent loop over a pair of visits and collect the panels."""
    cfg = config or PipelineConfig()
    acfg = agent_config or AgentConfig()

    kwargs: dict[str, Any] = {}
    if goal:
        kwargs["goal"] = goal

    run = run_agent(
        baseline_frames,
        rescan_frames,
        config=cfg,
        agent_config=acfg,
        brain=brain if brain is not None else make_brain(cfg=acfg),
        frame_provider=None,
        approver=None,
        **kwargs,
    )

    if run.result is None:
        # The loop stopped before producing a scan. The trace explains why and
        # there is nothing to draw, so this is reported rather than raised.
        return ScanOutcome(run=run, result=None)

    return ScanOutcome(run=run, result=run.result, panels=_build_panels(run))


def _build_panels(run: AgentRun) -> dict[str, np.ndarray]:
    result: ScanResult = run.result  # type: ignore[assignment]
    panels = {
        PANEL_BASELINE: result.baseline,
        PANEL_RESCAN: result.rescan_canonical,
        PANEL_OVERLAY: result.overlay(),
        PANEL_SIDE_BY_SIDE: result.side_by_side(),
    }
    # Guidance is drawn on the baseline, because that is the view the operator is
    # being asked to improve - the request is about coverage of the *site*, not
    # of the comparison.
    coverage = run.coverage or result.coverage
    requests = result.view_requests
    if coverage is not None:
        try:
            panels[PANEL_GUIDANCE] = render_guidance(result.baseline, coverage, requests)
        except Exception:  # pragma: no cover - drawing must never fail a scan
            log.warning("could not render the guidance overlay", exc_info=True)
    return panels


def uncertain_ids_from_trace(trace: AgentTrace) -> set[str]:
    """Events the trace records as explicitly unresolved."""
    out: set[str] = set()
    for step in trace.steps:
        if step.action == MARK_UNCERTAIN:
            event_id = (step.args or {}).get("event_id")
            if event_id:
                out.add(str(event_id))
    return out


# Panels a human reads as photographs can take lossy compression; panels carrying
# thin annotation strokes and small text cannot, because JPEG ringing shows up
# exactly on high-contrast edges. Encoding is chosen per panel rather than
# globally so neither group pays for the other's needs.
_LOSSLESS_PANELS = {PANEL_OVERLAY, PANEL_GUIDANCE}


def encode_panel(name: str, image: np.ndarray) -> bytes:
    """Encode one panel for transport, choosing the format by its content."""
    if name in _LOSSLESS_PANELS:
        return encoding.encode_png(image)
    return encoding.encode_jpeg(image, quality=86)


def panel_extension(name: str) -> str:
    """The file extension :func:`encode_panel` will produce for ``name``."""
    return ".png" if name in _LOSSLESS_PANELS else ".jpg"
