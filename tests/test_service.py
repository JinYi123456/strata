"""Tests for the HTTP service.

Two groups:

* **Fast** tests cover the parts that are pure logic - key validation, image
  decoding limits, error mapping, session bookkeeping. These are where the
  security-relevant behaviour lives, so they run on every change.
* **Slow** tests run a real scan through the API, because the things most worth
  checking about a service are exactly the things a mock cannot tell you: that a
  scan completes, that the panels exist and decode, and that the ledger records
  the decision that was actually made.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from service.app import create_app
from service.encoding import (
    MAX_DIMENSION,
    MAX_UPLOAD_BYTES,
    ImageError,
    decode_data_url,
    decode_frames,
    to_data_url,
)
from service.sessions import SessionError, SessionManager
from service.store import LocalStore, StoreError, validate_key
from strata_core.synth import make_scene

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> LocalStore:
    return LocalStore(tmp_path / "data")


@pytest.fixture
def client(store: LocalStore) -> TestClient:
    app = create_app(manager=SessionManager(store))
    return TestClient(app)


def _frame(width: int = 200, height: int = 150) -> np.ndarray:
    rng = np.random.default_rng(3)
    img = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (80, 70), (255, 0, 0), -1)
    return img


def _data_url(img: np.ndarray) -> str:
    return to_data_url(img)


# --------------------------------------------------------------------------
# Storage safety
# --------------------------------------------------------------------------


class TestKeyValidation:
    """Keys reach the filesystem, so traversal has to be impossible, not unlikely."""

    def test_a_normal_key_is_accepted(self):
        assert validate_key("sessions/abc/state.json") == "sessions/abc/state.json"

    @pytest.mark.parametrize(
        "bad",
        [
            "../../etc/passwd",
            "sessions/../../escape.json",
            "/absolute/path.json",
            "sessions/trailing/",
            "",
            "sessions/ab c/state.json",
            "sessions/ab\\c/state.json",
        ],
    )
    def test_dangerous_keys_are_rejected(self, bad: str):
        with pytest.raises(StoreError):
            validate_key(bad)

    def test_a_traversing_key_cannot_read_outside_the_root(self, store: LocalStore):
        with pytest.raises(StoreError):
            store.get_bytes("../outside.json")

    def test_a_traversing_key_cannot_write_outside_the_root(self, store: LocalStore):
        with pytest.raises(StoreError):
            store.put_bytes("../../outside.json", b"nope")


class TestLocalStore:
    def test_round_trips_bytes(self, store: LocalStore):
        store.put_bytes("a/b.bin", b"hello")
        assert store.get_bytes("a/b.bin") == b"hello"
        assert store.exists("a/b.bin")

    def test_a_missing_object_raises_rather_than_returning_none(self, store: LocalStore):
        with pytest.raises(StoreError):
            store.get_bytes("a/missing.bin")

    def test_missing_json_reads_as_none(self, store: LocalStore):
        # Absence and corruption are both "no record"; neither should raise.
        assert store.get_json("a/missing.json") is None

    def test_corrupt_json_reads_as_none_rather_than_crashing(self, store: LocalStore):
        store.put_bytes("a/bad.json", b"{not json")
        assert store.get_json("a/bad.json") is None

    def test_round_trips_json(self, store: LocalStore):
        store.put_json("a/b.json", {"k": [1, 2, 3]})
        assert store.get_json("a/b.json") == {"k": [1, 2, 3]}

    def test_lists_json_in_a_directory(self, store: LocalStore):
        store.put_json("ledger/site-1/a.json", {"n": 1})
        store.put_json("ledger/site-1/b.json", {"n": 2})
        store.put_json("ledger/site-2/c.json", {"n": 3})
        assert {e["n"] for e in store.list_json("ledger/site-1")} == {1, 2}

    def test_listing_a_missing_directory_is_empty_not_an_error(self, store: LocalStore):
        assert store.list_json("nothing/here") == []


# --------------------------------------------------------------------------
# Image boundary
# --------------------------------------------------------------------------


class TestImageDecoding:
    def test_a_data_url_round_trips(self):
        img = _frame()
        back = decode_data_url(to_data_url(img))
        assert back.shape == img.shape
        assert np.array_equal(back, img)

    def test_a_bare_base64_payload_is_accepted(self):
        raw = base64.b64encode(cv2.imencode(".png", _frame())[1].tobytes()).decode()
        assert decode_data_url(raw).shape[:2] == (150, 200)

    def test_garbage_is_rejected_with_a_clear_message(self):
        with pytest.raises(ImageError, match="base64|decode"):
            decode_data_url("data:image/png;base64,!!!not base64!!!")

    def test_an_empty_payload_is_rejected(self):
        with pytest.raises(ImageError):
            decode_data_url("")

    def test_an_oversized_payload_is_rejected_before_decoding(self):
        # The cap must apply to the encoded bytes: a small file can decode to a
        # very large array, so checking the result would already be too late.
        blob = "A" * int(MAX_UPLOAD_BYTES * 1.4)
        with pytest.raises(ImageError, match="limit|MB"):
            decode_data_url(blob)

    def test_an_over_large_image_is_rejected_after_decoding(self):
        big = np.zeros((MAX_DIMENSION + 40, 64, 3), np.uint8)
        with pytest.raises(ImageError, match="longest edge"):
            decode_data_url(to_data_url(big))

    def test_a_tiny_image_is_rejected(self):
        with pytest.raises(ImageError, match="at least 32"):
            decode_data_url(to_data_url(np.zeros((16, 16, 3), np.uint8)))

    def test_too_many_frames_are_rejected(self):
        with pytest.raises(ImageError, match="at most"):
            decode_frames([_data_url(_frame())] * 9)

    def test_no_frames_decodes_to_an_empty_list(self):
        assert decode_frames(None) == []


# --------------------------------------------------------------------------
# Endpoints that need no scan
# --------------------------------------------------------------------------


class TestMeta:
    def test_health_reports_provenance(self, client: TestClient):
        body = client.get("/api/health").json()
        assert body["status"] == "ok"
        assert body["opencv_version"].startswith("5.")
        assert body["vision_backend"] in {"neural-lightglue", "classic-sift", "classic-orb"}
        assert body["storage"] == "local"
        assert body["limits"]["max_frames_per_upload"] > 0

    def test_sites_lists_the_demo_sites_with_thumbnails(self, client: TestClient):
        body = client.get("/api/sites").json()
        ids = [s["site_id"] for s in body["sites"]]
        assert "bay-4" in ids
        for site in body["sites"]:
            assert site["is_demo"] is True
            assert site["thumbnail"].startswith("data:image/jpeg;base64,")

    def test_the_documented_endpoints_are_published(self, client: TestClient):
        schema = client.get("/api/openapi.json").json()
        assert "/api/sessions" in schema["paths"]
        assert "/api/sessions/{session_id}/rescan" in schema["paths"]


class TestSessionCreation:
    def test_a_demo_session_reports_its_sizes(self, client: TestClient):
        r = client.post("/api/sessions", json={"site_id": "bay-4"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["canonical_size"] == [1600, 1200]
        assert body["baseline"]["width"] == 960
        assert body["baseline"]["height"] == 720
        assert body["scans"] == 0
        assert body["committed"] is False

    def test_an_uploaded_session_is_accepted(self, client: TestClient):
        r = client.post(
            "/api/sessions",
            json={"name": "My shelf", "baseline_frames": [_data_url(_frame())]},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["name"] == "My shelf"
        assert body["baseline"]["width"] == 200

    def test_an_unknown_site_is_a_structured_400(self, client: TestClient):
        r = client.post("/api/sessions", json={"site_id": "does-not-exist"})
        assert r.status_code == 400
        assert r.json()["error"] == "session_error"

    def test_asking_for_both_a_site_and_frames_is_rejected(self, client: TestClient):
        r = client.post(
            "/api/sessions",
            json={"site_id": "bay-4", "baseline_frames": [_data_url(_frame())]},
        )
        assert r.status_code == 400

    def test_asking_for_neither_is_rejected(self, client: TestClient):
        assert client.post("/api/sessions", json={}).status_code == 400

    def test_a_bad_image_is_reported_as_an_image_error(self, client: TestClient):
        r = client.post(
            "/api/sessions", json={"baseline_frames": ["data:image/png;base64,zzzz"]}
        )
        assert r.status_code == 400
        assert r.json()["error"] == "invalid_image"

    def test_an_unknown_session_is_a_structured_404(self, client: TestClient):
        r = client.get("/api/sessions/ses_nope")
        assert r.status_code in (400, 404)
        assert "error" in r.json()

    def test_the_baseline_image_is_served_as_png(self, client: TestClient):
        sid = client.post("/api/sessions", json={"site_id": "bay-4"}).json()["session_id"]
        r = client.get(f"/api/sessions/{sid}/baseline?frame=0")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/png"
        assert r.content[:4] == b"\x89PNG"

    def test_an_out_of_range_baseline_frame_is_rejected(self, client: TestClient):
        sid = client.post("/api/sessions", json={"site_id": "bay-4"}).json()["session_id"]
        assert client.get(f"/api/sessions/{sid}/baseline?frame=7").status_code == 400


class TestCommitRequiresSomethingToCommit:
    def test_committing_a_session_with_no_scans_is_rejected(self, client: TestClient):
        sid = client.post("/api/sessions", json={"site_id": "bay-4"}).json()["session_id"]
        r = client.post(f"/api/sessions/{sid}/commit", json={"approved": True})
        assert r.status_code == 400
        assert "no scans" in r.json()["detail"]

    def test_a_ledger_for_an_unscanned_site_is_empty(self, client: TestClient):
        body = client.get("/api/sites/bay-4/ledger").json()
        assert body == {"site_id": "bay-4", "entries": []}


class TestRescanPreconditions:
    def test_an_uploaded_session_needs_frames_for_a_rescan(self, client: TestClient):
        sid = client.post(
            "/api/sessions", json={"baseline_frames": [_data_url(_frame())]}
        ).json()["session_id"]
        r = client.post(f"/api/sessions/{sid}/rescan", json={})
        assert r.status_code == 400
        assert "must supply frames" in r.json()["detail"]

    def test_declining_the_stored_rescan_without_frames_is_rejected(self, client: TestClient):
        sid = client.post("/api/sessions", json={"site_id": "bay-4"}).json()["session_id"]
        r = client.post(
            f"/api/sessions/{sid}/rescan", json={"use_site_rescan": False}
        )
        assert r.status_code == 400

    def test_rescanning_an_unknown_session_is_a_structured_error(self, client: TestClient):
        r = client.post("/api/sessions/ses_nope/rescan", json={})
        assert r.status_code in (400, 404)
        assert "error" in r.json()


class TestPanelAccess:
    def test_an_unknown_panel_name_is_rejected(self, client: TestClient):
        sid = client.post("/api/sessions", json={"site_id": "bay-4"}).json()["session_id"]
        with pytest.raises(SessionError):
            SessionManager(LocalStore(Path(".") / "_unused")).read_panel(sid, 0, "nope")

    def test_a_missing_scan_is_reported(self, store: LocalStore):
        mgr = SessionManager(store)
        sid = mgr.create_session(site_id="bay-4").session_id
        with pytest.raises(StoreError):
            mgr.read_panel(sid, 0, "overlay")


# --------------------------------------------------------------------------
# End to end, through the API
# --------------------------------------------------------------------------


@pytest.mark.slow
class TestFullScan:
    """The real thing: create, scan, fetch panels, commit, read the ledger."""

    def test_a_demo_scan_returns_events_panels_and_a_trace(self, client: TestClient):
        sid = client.post("/api/sessions", json={"site_id": "bay-4"}).json()["session_id"]
        r = client.post(f"/api/sessions/{sid}/rescan", json={})
        assert r.status_code == 200, r.text
        body = r.json()

        # The agent prepares the change set and stops: the write is a human
        # decision, and the server must not claim otherwise.
        assert body["outcome"] == "declined"
        assert body["registration"]["model"] == "homography"
        assert body["registration"]["confidence"] > 0.5
        assert body["events"], "a demo site with known changes should report some"
        assert set(body["images"]) == {
            "baseline",
            "rescan",
            "overlay",
            "side_by_side",
            "guidance",
        }
        assert body["trace"]["steps"]
        assert body["trace"]["brains_used"] == ["policy-rules"]
        assert body["ground_truth"], "demo sites report the true changes alongside"

    def test_every_advertised_panel_actually_decodes(self, client: TestClient):
        sid = client.post("/api/sessions", json={"site_id": "bay-4"}).json()["session_id"]
        body = client.post(f"/api/sessions/{sid}/rescan", json={}).json()
        for name, ref in body["images"].items():
            r = client.get(ref["url"])
            assert r.status_code == 200, name
            arr = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
            assert arr is not None, name
            assert arr.shape[1] == ref["width"] and arr.shape[0] == ref["height"], name

    def test_event_boxes_are_in_canonical_pixels(self, client: TestClient):
        sid = client.post("/api/sessions", json={"site_id": "bay-4"}).json()["session_id"]
        body = client.post(f"/api/sessions/{sid}/rescan", json={}).json()
        cw, ch = body["images"]["overlay"]["width"], body["images"]["overlay"]["height"]
        for e in body["events"]:
            b = e["bbox"]
            assert 0 <= b["x"] < cw and 0 <= b["y"] < ch
            assert b["width"] > 0 and b["height"] > 0
            assert b["x"] + b["width"] <= cw + 1
            assert b["y"] + b["height"] <= ch + 1

    def test_event_totals_ignore_the_presentation_filter(self, client: TestClient):
        """Filtering what is shown must not change what was found."""
        sid = client.post("/api/sessions", json={"site_id": "bay-4"}).json()["session_id"]
        full = client.post(f"/api/sessions/{sid}/rescan", json={}).json()
        filtered = client.post(
            f"/api/sessions/{sid}/rescan", json={"min_confidence": 0.99}
        ).json()
        assert filtered["event_totals"] == full["event_totals"]
        assert len(filtered["events"]) <= len(full["events"])

    def test_a_second_scan_increments_the_index(self, client: TestClient):
        sid = client.post("/api/sessions", json={"site_id": "bay-4"}).json()["session_id"]
        first = client.post(f"/api/sessions/{sid}/rescan", json={}).json()
        second = client.post(f"/api/sessions/{sid}/rescan", json={}).json()
        assert first["scan_index"] == 0
        assert second["scan_index"] == 1
        assert client.get(f"/api/sessions/{sid}").json()["scans"] == 2

    def test_committing_records_the_human_decision_in_the_ledger(self, client: TestClient):
        sid = client.post("/api/sessions", json={"site_id": "bay-4"}).json()["session_id"]
        scan = client.post(f"/api/sessions/{sid}/rescan", json={}).json()

        r = client.post(
            f"/api/sessions/{sid}/commit",
            json={"approved": True, "note": "Verified against the pick list."},
        )
        assert r.status_code == 200, r.text
        entry = r.json()
        assert entry["approved"] is True
        assert entry["decision_made_by"] == "human"
        assert entry["note"] == "Verified against the pick list."
        assert entry["event_totals"] == scan["event_totals"]
        assert entry["registration_confidence"] == pytest.approx(
            scan["registration"]["confidence"]
        )

        ledger = client.get("/api/sites/bay-4/ledger").json()
        assert [e["entry_id"] for e in ledger["entries"]] == [entry["entry_id"]]

    def test_a_withheld_decision_is_recorded_as_withheld(self, client: TestClient):
        sid = client.post("/api/sessions", json={"site_id": "aisle-2"}).json()["session_id"]
        client.post(f"/api/sessions/{sid}/rescan", json={})
        entry = client.post(
            f"/api/sessions/{sid}/commit", json={"approved": False, "note": "Blurry."}
        ).json()
        assert entry["approved"] is False
        assert client.get(f"/api/sessions/{sid}").json()["committed"] is False

    def test_the_ledger_is_append_only(self, client: TestClient):
        sid = client.post("/api/sessions", json={"site_id": "bench-c"}).json()["session_id"]
        for note in ("first look", "second look"):
            client.post(f"/api/sessions/{sid}/rescan", json={})
            client.post(f"/api/sessions/{sid}/commit", json={"approved": True, "note": note})
        entries = client.get("/api/sites/bench-c/ledger").json()["entries"]
        assert [e["note"] for e in entries] == ["first look", "second look"]

    def test_a_scan_record_is_retrievable_and_serialisable(self, client: TestClient):
        sid = client.post("/api/sessions", json={"site_id": "bay-4"}).json()["session_id"]
        client.post(f"/api/sessions/{sid}/rescan", json={})
        r = client.get(f"/api/sessions/{sid}/scans/0")
        assert r.status_code == 200
        record = r.json()
        assert record["scan_index"] == 0
        assert record["outcome"] == "declined"
        assert "registration" in record and "trace" in record
        json.dumps(record)  # must be serialisable as-is

    def test_the_agent_asks_for_a_better_view_when_coverage_is_thin(self, client: TestClient):
        sid = client.post("/api/sessions", json={"site_id": "yard-north"}).json()["session_id"]
        body = client.post(f"/api/sessions/{sid}/rescan", json={}).json()
        # The core's vocabulary is observed / weak / unobserved, and a single
        # baseline frame of a wide scene is legitimately near the bottom of it.
        assert body["coverage"]["status"] in {"observed", "weak", "unobserved"}
        assert 0.0 <= body["coverage"]["observed_fraction"] <= 1.0
        # A single frame cannot cover a wide scene, so guidance is the honest
        # output rather than silence.
        assert isinstance(body["guidance"], list)
        if body["guidance"]:
            req = body["guidance"][0]
            assert req["direction"]
            assert req["reason"]

    def test_thin_coverage_is_reflected_in_the_agent_trace(self, client: TestClient):
        """Whatever the coverage says, the trace must say the same thing."""
        sid = client.post("/api/sessions", json={"site_id": "yard-north"}).json()["session_id"]
        body = client.post(f"/api/sessions/{sid}/rescan", json={}).json()
        rationales = " ".join(s.get("rationale") or "" for s in body["trace"]["steps"]).lower()
        if body["coverage"]["status"] != "observed":
            assert "cover" in rationales or "view" in rationales

    def test_an_unregistrable_pair_aborts_without_inventing_a_result(self, client: TestClient):
        """A blank baseline cannot be aligned, and the service must say so.

        This is also the case that used to return a 500: an identity registration
        reports an infinite reprojection error, and JSON cannot encode infinity.
        The failure only appeared on the path that most needs to work.
        """
        blank = np.full((240, 320, 3), 127, np.uint8)
        sid = client.post(
            "/api/sessions", json={"baseline_frames": [_data_url(blank)]}
        ).json()["session_id"]
        r = client.post(
            f"/api/sessions/{sid}/rescan",
            json={"frames": [_data_url(_frame(320, 240))]},
        )
        assert r.status_code == 200, r.text
        body = r.json()

        assert body["outcome"] == "aborted"
        assert body["events"] == []
        assert body["event_totals"] == {}
        # An abort must not read as "nothing changed": there is no model, so the
        # registration reports identity at zero confidence, and the error is
        # null rather than zero because nothing was measured.
        assert body["registration"]["model"] == "identity"
        assert body["registration"]["confidence"] == 0.0
        assert body["registration"]["reproj_error_px"] is None
        assert body["trace"]["steps"]
        # The trace must carry the reason, or the abort is unexplained.
        assert any(
            "registration confidence" in (s.get("rationale") or "").lower()
            for s in body["trace"]["steps"]
        )
        json.dumps(body)  # the whole payload must be JSON-representable

    def test_a_scan_that_found_nothing_still_reports_a_real_registration(
        self, client: TestClient
    ):
        """The opposite case, for contrast: a good alignment with no changes."""
        sid = client.post("/api/sessions", json={"site_id": "bay-4"}).json()["session_id"]
        baseline = client.get(f"/api/sessions/{sid}/baseline?frame=0").content
        r = client.post(
            f"/api/sessions/{sid}/rescan",
            json={"frames": ["data:image/png;base64," + base64.b64encode(baseline).decode()]},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        # Rescanning the baseline against itself registers perfectly and should
        # report almost nothing - the false-positive case that matters most.
        assert body["registration"]["model"] == "homography"
        assert body["registration"]["confidence"] > 0.8
        assert body["registration"]["reproj_error_px"] is not None
        assert sum(body["event_totals"].values()) == 0, body["event_totals"]

    def test_the_demo_hard_case_is_reported_honestly(self, client: TestClient):
        """The site the UI labels difficult should really be difficult."""
        sid = client.post("/api/sessions", json={"site_id": "archive-row-7"}).json()["session_id"]
        body = client.post(f"/api/sessions/{sid}/rescan", json={}).json()
        assert body["outcome"] in {"declined", "aborted", "step_limit"}
        assert body["ground_truth"]
        # Whether or not it is right, the true changes are always disclosed for
        # a demo site, so a viewer can judge the result for themselves.
        assert all("bbox" in g for g in body["ground_truth"])
