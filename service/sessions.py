"""Session lifecycle: sites, baselines, rescans, and the ledger.

A session is one site under observation. The shape of the model matters:

* **The baseline is written once.** Everything a rescan is compared against is
  fixed at session creation, so "what changed since?" has a definite answer.
* **Every scan is recorded, including the ones that were not committed.** A scan
  that found nothing, or that the operator declined, is still history - knowing
  that someone looked and saw no change is often the point of looking.
* **The ledger is append-only.** Committing appends; it never edits.

Scans are serialised behind a semaphore. A full scan is CPU-bound and takes
seconds, and letting an impatient client launch ten at once would make all ten
slow rather than one fast, while competing for the same cores as everything else
on the box.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

from strata_agent.brain import Brain
from strata_agent.policy import AgentConfig
from strata_core.config import PipelineConfig

from . import demo, encoding, runners
from .observability import log_fields, scan_log_fields
from .store import Store, StoreError

log = logging.getLogger("strata.service.sessions")

MAX_CONCURRENT_SCANS = 2


class SessionError(RuntimeError):
    """A request could not be satisfied against the current session state."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


@dataclass
class Session:
    """The persisted form of a session."""

    session_id: str
    site_id: str
    name: str
    created_at: str
    canonical_size: tuple[int, int]
    baseline_frames: int
    is_demo: bool
    # The source frame's own size, kept because it is what the client displays
    # the baseline at. The canonical size is a derived working resolution and is
    # usually larger, so reporting it as the baseline's size would be wrong.
    baseline_size: tuple[int, int] = (0, 0)
    scans: int = 0
    committed: bool = False
    baseline_coverage: dict[str, Any] | None = None

    def to_state(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "site_id": self.site_id,
            "name": self.name,
            "created_at": self.created_at,
            "canonical_size": list(self.canonical_size),
            "baseline_frames": self.baseline_frames,
            "baseline_size": list(self.baseline_size),
            "is_demo": self.is_demo,
            "scans": self.scans,
            "committed": self.committed,
            "baseline_coverage": self.baseline_coverage,
        }

    @classmethod
    def from_state(cls, d: dict[str, Any]) -> "Session":
        return cls(
            session_id=d["session_id"],
            site_id=d["site_id"],
            name=d["name"],
            created_at=d["created_at"],
            canonical_size=tuple(d["canonical_size"]),  # type: ignore[arg-type]
            baseline_frames=int(d["baseline_frames"]),
            is_demo=bool(d.get("is_demo", False)),
            baseline_size=tuple(d.get("baseline_size", (0, 0))),  # type: ignore[arg-type]
            scans=int(d.get("scans", 0)),
            committed=bool(d.get("committed", False)),
            baseline_coverage=d.get("baseline_coverage"),
        )


class SessionManager:
    """All session state transitions, in one place."""

    def __init__(
        self,
        store: Store,
        *,
        pipeline_config: PipelineConfig | None = None,
        agent_config: AgentConfig | None = None,
        brain: Brain | None = None,
    ) -> None:
        self.store = store
        self.pipeline_config = pipeline_config or PipelineConfig()
        self.agent_config = agent_config or AgentConfig()
        self.brain = brain
        self._scan_gate = threading.Semaphore(MAX_CONCURRENT_SCANS)
        # Cached because generating a scene is not free and the picker asks for
        # the list on every page load.
        self._demo_scenes: dict[str, Any] = {}
        self._demo_lock = threading.Lock()

    # --- Demo sites ----------------------------------------------------

    def _demo_scene(self, site_id: str):
        with self._demo_lock:
            if site_id not in self._demo_scenes:
                site = demo.get_demo_site(site_id)
                if site is None:
                    raise SessionError(f"unknown demo site {site_id!r}")
                self._demo_scenes[site_id] = site.build()
            return self._demo_scenes[site_id]

    def list_sites(self) -> list[dict[str, Any]]:
        """Every site the operator can open, with a preview thumbnail."""
        out: list[dict[str, Any]] = []
        for site in demo.DEMO_SITES:
            thumb = demo.thumbnail(self._demo_scene(site.site_id))
            out.append(
                {
                    "site_id": site.site_id,
                    "name": site.name,
                    "description": site.description,
                    "is_demo": True,
                    "baseline_frames": 1,
                    "thumbnail": encoding.to_data_url(thumb, fmt="jpg", quality=80),
                }
            )
        # Sites created from uploads are listed too, so a session opened earlier
        # is still reachable after a restart.
        for state in self.store.list_json("sessions"):
            if state.get("is_demo"):
                continue
            sid = state.get("session_id")
            if not sid:
                continue
            out.append(
                {
                    "site_id": state.get("site_id", sid),
                    "name": state.get("name", "Uploaded site"),
                    "description": "Uploaded captures.",
                    "is_demo": False,
                    "baseline_frames": int(state.get("baseline_frames", 0)),
                    "thumbnail": None,
                }
            )
        return out

    # --- Session creation ----------------------------------------------

    def create_session(
        self,
        *,
        site_id: str | None = None,
        name: str | None = None,
        baseline_frames: list[np.ndarray] | None = None,
    ) -> Session:
        """Open a session, from a site or from uploaded baseline frames."""
        if site_id and baseline_frames:
            raise SessionError("provide either a site_id or baseline frames, not both")
        if not site_id and not baseline_frames:
            raise SessionError("provide a site_id or at least one baseline frame")

        if site_id:
            site = demo.get_demo_site(site_id)
            if site is None:
                # An uploaded session addressed by its own site id.
                return self._resume_uploaded(site_id)
            scene = self._demo_scene(site_id)
            frames = [scene.baseline]
            session_name = name or site.name
            is_demo_site = True
        else:
            frames = list(baseline_frames or [])
            session_name = name or "Uploaded site"
            is_demo_site = False

        if not frames:
            raise SessionError("no baseline frames")

        session_id = _new_id("ses")
        canonical = _canonical_size(frames[0], self.pipeline_config)
        bh, bw = frames[0].shape[:2]

        session = Session(
            session_id=session_id,
            site_id=site_id or session_id,
            name=session_name,
            created_at=_now(),
            canonical_size=canonical,
            baseline_frames=len(frames),
            is_demo=is_demo_site,
            baseline_size=(int(bw), int(bh)),
        )

        for i, frame in enumerate(frames):
            self.store.put_bytes(self._baseline_key(session_id, i), encoding.encode_png(frame))
        self.store.put_json(self._state_key(session_id), session.to_state())
        return session

    def _resume_uploaded(self, site_id: str) -> Session:
        """Find an earlier upload-only session by its site id."""
        for state in self.store.list_json("sessions"):
            if state.get("site_id") == site_id and not state.get("is_demo"):
                return Session.from_state(state)
        raise SessionError(f"unknown site {site_id!r}")

    def get_session(self, session_id: str) -> Session:
        state = self.store.get_json(self._state_key(session_id))
        if state is None:
            raise SessionError(f"no such session {session_id!r}")
        return Session.from_state(state)

    def baseline_frames(self, session: Session) -> list[np.ndarray]:
        out = []
        for i in range(session.baseline_frames):
            raw = self.store.get_bytes(self._baseline_key(session.session_id, i))
            out.append(encoding.decode_bytes(raw))
        return out

    # --- Rescan ---------------------------------------------------------

    def rescan(
        self,
        session_id: str,
        *,
        frames: list[np.ndarray] | None = None,
        min_confidence: float = 0.0,
    ) -> tuple[runners.ScanOutcome, int]:
        """Run a rescan and persist it. Returns the outcome and the scan index."""
        session = self.get_session(session_id)
        baseline = self.baseline_frames(session)

        if frames:
            rescan_frames = list(frames)
        elif session.is_demo:
            site = demo.get_demo_site(session.site_id)
            if site is None:
                raise SessionError("this session has no stored rescan")
            rescan_frames = [self._demo_scene(session.site_id).rescan]
        else:
            raise SessionError(
                "this session was created from an upload, so a rescan must supply frames"
            )

        goal = self._goal_for(session)

        # Serialised: a scan is CPU-bound and a burst would make every request
        # slow instead of one fast.
        with self._scan_gate:
            outcome = runners.execute_scan(
                baseline,
                rescan_frames,
                config=self.pipeline_config,
                agent_config=self.agent_config,
                brain=self.brain,
                goal=goal,
            )

        scan_index = session.scans
        self._persist_scan(session, scan_index, outcome, min_confidence)
        # One structured line per scan, so "why did that take forty seconds" is
        # answered by the log rather than by reproducing the run.
        log_fields(
            log,
            "scan completed",
            session_id=session_id,
            site_id=session.site_id,
            scan_index=scan_index,
            outcome=str(outcome.run.outcome),
            **scan_log_fields(outcome.result.to_summary() if outcome.has_result else {}),
        )

        session.scans = scan_index + 1
        self.store.put_json(self._state_key(session_id), session.to_state())
        return outcome, scan_index

    def _goal_for(self, session: Session) -> str:
        return (
            f"Determine what changed at {session.name} between the baseline visit "
            f"and this rescan, and report anything that cannot be resolved with "
            f"confidence."
        )

    def _persist_scan(
        self,
        session: Session,
        scan_index: int,
        outcome: runners.ScanOutcome,
        min_confidence: float,
    ) -> None:
        sid = session.session_id
        record: dict[str, Any] = {
            "session_id": sid,
            "site_id": session.site_id,
            "scan_index": scan_index,
            "recorded_at": _now(),
            "outcome": outcome.run.outcome,
            "min_confidence": float(min_confidence),
            "trace": outcome.run.trace.to_dict(),
            "uncertain_ids": sorted(outcome.uncertain_ids),
            "registration": (
                outcome.result.registration.to_summary() if outcome.has_result else None
            ),
            "events": [e.to_dict() for e in outcome.events],
            "event_totals": _totals(outcome.events),
            "timings_ms": (
                outcome.result.timings.to_dict() if outcome.has_result else {}
            ),
            "coverage": (
                {
                    "status": outcome.result.coverage.status,
                    "observed_fraction": round(
                        float(outcome.result.coverage.observed_fraction), 4
                    ),
                    "frame_count": int(outcome.result.coverage.frame_count),
                    "view_requests": [r.to_dict() for r in outcome.result.view_requests],
                }
                if outcome.has_result
                else None
            ),
        }
        self.store.put_json(self._scan_key(sid, scan_index), record)

        for name, panel in outcome.panels.items():
            self.store.put_bytes(
                self._panel_key(sid, scan_index, name),
                runners.encode_panel(name, panel),
            )

    # --- Ledger ---------------------------------------------------------

    def commit(self, session_id: str, *, approved: bool, note: str = "") -> dict[str, Any]:
        """Record the operator's decision on the most recent scan."""
        session = self.get_session(session_id)
        if session.scans == 0:
            raise SessionError("nothing to decide: this session has no scans yet")

        scan_index = session.scans - 1
        record = self.store.get_json(self._scan_key(session_id, scan_index))
        if record is None:
            raise SessionError(f"scan {scan_index} is missing from storage")

        # A scan's stored outcome is "declined" while it sits at the gate, meaning
        # only "prepared, nothing written yet". Carrying that label into the ledger
        # recorded an approved change set as a declined one - the history said the
        # opposite of what the operator decided, in the one field an auditor reads.
        scan_outcome = record.get("outcome", "unknown")
        if approved and scan_outcome == "aborted":
            raise SessionError(
                "this scan stopped before it produced a change set; there is "
                "nothing to approve"
            )
        outcome = "finalized" if approved else scan_outcome

        registration = record.get("registration") or {}
        entry = {
            "entry_id": _new_id("led"),
            "session_id": session_id,
            "site_id": session.site_id,
            "scan_index": scan_index,
            "recorded_at": _now(),
            "outcome": outcome,
            "approved": bool(approved),
            "note": note,
            "event_totals": record.get("event_totals", {}),
            "registration_confidence": float(registration.get("confidence", 0.0)),
            "decision_made_by": "human",
        }
        self.store.put_json(self._ledger_key(session.site_id, entry["entry_id"]), entry)

        session.committed = bool(approved)
        self.store.put_json(self._state_key(session_id), session.to_state())
        return entry

    def ledger(self, site_id: str) -> list[dict[str, Any]]:
        entries = self.store.list_json(f"ledger/{site_id}")
        return sorted(entries, key=lambda e: e.get("recorded_at", ""))

    def scan_record(self, session_id: str, scan_index: int) -> dict[str, Any]:
        record = self.store.get_json(self._scan_key(session_id, scan_index))
        if record is None:
            raise SessionError(f"no scan {scan_index} in session {session_id!r}")
        return record

    def read_panel(self, session_id: str, scan_index: int, name: str) -> bytes:
        if name not in runners.PANELS:
            raise SessionError(f"unknown panel {name!r}")
        # Both extensions are probed because the encoder chooses per panel and a
        # future change to that choice should not silently 404 a stored scan.
        for ext in (".png", ".jpg"):
            key = self._panel_base(session_id, scan_index, name) + ext
            if self.store.exists(key):
                return self.store.get_bytes(key)
        raise StoreError(f"panel {name!r} is not available for scan {scan_index}")

    def read_baseline(self, session_id: str, frame: int = 0) -> bytes:
        """Raw bytes of one stored baseline frame."""
        session = self.get_session(session_id)
        if not 0 <= frame < session.baseline_frames:
            raise SessionError(
                f"frame {frame} is out of range; this session has "
                f"{session.baseline_frames} baseline frame(s)"
            )
        return self.store.get_bytes(self._baseline_key(session_id, frame))

    # --- Key layout -----------------------------------------------------

    def _state_key(self, session_id: str) -> str:
        return f"sessions/{session_id}/state.json"

    def _baseline_key(self, session_id: str, index: int) -> str:
        return f"sessions/{session_id}/baseline/{index}.png"

    def _scan_key(self, session_id: str, scan_index: int) -> str:
        return f"sessions/{session_id}/scans/{scan_index}.json"

    def _panel_base(self, session_id: str, scan_index: int, name: str) -> str:
        """Panel path without an extension.

        Split from the extension on purpose: the writer appends the encoder's
        chosen extension and the reader probes for either, and having one
        function return a path that already ends in an extension is how the
        reader ends up looking for ``baseline.jpg.jpg``.
        """
        return f"sessions/{session_id}/scans/{scan_index}/{name}"

    def _panel_key(self, session_id: str, scan_index: int, name: str) -> str:
        return self._panel_base(session_id, scan_index, name) + runners.panel_extension(name)

    def _ledger_key(self, site_id: str, entry_id: str) -> str:
        return f"ledger/{site_id}/{entry_id}.json"


def _canonical_size(frame: np.ndarray, cfg: PipelineConfig) -> tuple[int, int]:
    from strata_core.geometry import canonical_size_for

    return canonical_size_for(frame.shape[:2], cfg.geometry.canonical_long_edge)


def _totals(events) -> dict[str, int]:
    out: dict[str, int] = {}
    for e in events:
        out[e.kind] = out.get(e.kind, 0) + 1
    return out
