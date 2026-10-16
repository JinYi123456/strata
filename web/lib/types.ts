/**
 * The API contract, mirrored from the service's own schemas.
 *
 * These types are written by hand rather than generated, and that is a
 * deliberate trade: the service publishes an OpenAPI document at
 * `/api/openapi.json` and a generator could produce this file, but the generated
 * output would bury the two facts that matter most in a wall of structure. So
 * they are stated plainly here instead:
 *
 * 1. **Every box is `(x, y, width, height)` in canonical pixels.** Not source
 *    pixels. The canonical frame is the common coordinate system two visits are
 *    compared in, and it is larger than either capture.
 * 2. **`null` means "not measured", not zero.** An aborted registration has no
 *    reprojection error, which is a different claim from an error of zero.
 */

export type EventKind = "missing" | "added" | "moved" | "state_changed" | "unchanged";

export type AgentPhase = "observe" | "decide" | "act" | "gate" | "finish";

export type ScanOutcome =
  | "finalized"
  | "declined"
  | "aborted"
  | "step_limit"
  | "brain_failed"
  | "running";

export interface Health {
  status: "ok";
  service: string;
  api_version: string;
  vision_backend: string;
  executor: string;
  storage: string;
  opencv_version: string;
  model_weights_present: boolean;
  limits: {
    max_upload_bytes: number;
    max_frames_per_upload: number;
    max_dimension: number;
    max_concurrent_scans: number;
  };
}

export interface ImageRef {
  url: string;
  width: number;
  height: number;
}

export interface Box {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface SiteSummary {
  site_id: string;
  name: string;
  description: string;
  is_demo: boolean;
  baseline_frames: number;
  thumbnail: string | null;
}

export interface SessionState {
  session_id: string;
  site_id: string;
  name: string;
  created_at: string;
  canonical_size: [number, number];
  baseline_frames: number;
  baseline: ImageRef | null;
  coverage: Coverage | null;
  guidance: ImageRef | null;
  scans: number;
  committed: boolean;
}

export interface Coverage {
  status: "observed" | "weak" | "unobserved" | string;
  observed_fraction: number;
  frame_count: number;
  grid?: [number, number];
  view_requests?: ViewRequest[];
}

export interface ViewRequest {
  direction: string;
  target_cell: [number, number];
  deficit: number;
  priority: number;
  reason: string;
}

export interface ChangeEvent {
  event_id: string;
  kind: EventKind;
  confidence: number;
  bbox: Box;
  centroid: [number, number];
  displacement_px: number;
  appearance_delta: number;
  summary: string;
  evidence: Record<string, unknown>;
  /** The agent explicitly declined to resolve this one. */
  uncertain: boolean;
}

export interface TraceStep {
  index: number;
  phase: AgentPhase | string;
  action: string | null;
  rationale: string | null;
  decided_by: string | null;
  confidence: number | null;
  duration_ms: number | null;
  args: Record<string, unknown>;
  result: Record<string, unknown>;
}

export interface Trace {
  steps: TraceStep[];
  brains_used: string[];
  used_fallback: boolean;
  decision_count: number;
  action_count: number;
}

export interface Registration {
  model: string;
  confidence: number;
  inlier_count: number;
  match_count: number;
  inlier_ratio: number;
  reproj_error_px: number | null;
  notes: string[];
}

export interface GroundTruthEntry {
  kind: string;
  object_id: string;
  bbox: [number, number, number, number];
  bbox_space: "baseline" | "rescan";
  mutation: Record<string, unknown>;
}

export interface ScanResult {
  session_id: string;
  scan_index: number;
  outcome: ScanOutcome;
  feature_backend: string;
  registration: Registration;
  events: ChangeEvent[];
  event_totals: Record<string, number>;
  coverage: Coverage;
  guidance: ViewRequest[];
  trace: Trace;
  timings_ms: Record<string, number>;
  images: Partial<Record<PanelName, ImageRef>>;
  ground_truth: GroundTruthEntry[] | null;
}

export type PanelName = "baseline" | "rescan" | "overlay" | "side_by_side" | "guidance";

export interface LedgerEntry {
  entry_id: string;
  session_id: string;
  scan_index: number;
  recorded_at: string;
  outcome: string;
  approved: boolean;
  note: string;
  event_totals: Record<string, number>;
  registration_confidence: number;
  decision_made_by: string;
}

export interface Ledger {
  site_id: string;
  entries: LedgerEntry[];
}

export interface ApiErrorBody {
  error: string;
  detail: string;
  hint: string;
}
