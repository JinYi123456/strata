/**
 * The vocabulary the whole interface is built from.
 *
 * Change kinds get their own name, colour, and glyph here and nowhere else. The
 * glyph matters as much as the colour: a reader who cannot distinguish the hues
 * still needs to tell a departure from an arrival at a glance, and colour alone
 * cannot carry that.
 */

import type { EventKind } from "./types";

export interface KindStyle {
  /** How the interface refers to this kind. */
  label: string;
  /** The plain-language gloss, for when clarity beats character. */
  gloss: string;
  glyph: string;
  hex: string;
  /** Tailwind classes for text, border, and background at low opacity. */
  text: string;
  border: string;
  bg: string;
}

export const KIND: Record<EventKind, KindStyle> = {
  missing: {
    label: "Departed",
    gloss: "was here, now gone",
    glyph: "↓",
    hex: "#FF6B6B",
    text: "text-kind-missing",
    border: "border-kind-missing/40",
    bg: "bg-kind-missing/10",
  },
  added: {
    label: "Arrived",
    gloss: "not here before, now present",
    glyph: "↑",
    hex: "#4FE3C1",
    text: "text-kind-added",
    border: "border-kind-added/40",
    bg: "bg-kind-added/10",
  },
  moved: {
    label: "Relocated",
    gloss: "same thing, different place",
    glyph: "⇄",
    hex: "#B18CFF",
    text: "text-kind-moved",
    border: "border-kind-moved/40",
    bg: "bg-kind-moved/10",
  },
  state_changed: {
    label: "Altered",
    gloss: "same place, changed appearance",
    glyph: "◐",
    hex: "#63B3FF",
    text: "text-kind-state",
    border: "border-kind-state/40",
    bg: "bg-kind-state/10",
  },
  unchanged: {
    label: "Unchanged",
    gloss: "no difference found",
    glyph: "·",
    hex: "#93A3B4",
    text: "text-ink-muted",
    border: "border-ink-faint/30",
    bg: "bg-white/5",
  },
};

export function kindStyle(kind: string): KindStyle {
  return KIND[kind as EventKind] ?? KIND.unchanged;
}

/** The order kinds are listed in, chosen so departures read before arrivals. */
export const KIND_ORDER: EventKind[] = ["missing", "added", "moved", "state_changed"];

/**
 * How a scan outcome should be read.
 *
 * `declined` is the normal case, not a failure: the change set was prepared and
 * the write is waiting on a human. The interface says so plainly, because
 * "declined" as a bare word reads like something went wrong.
 */
export interface OutcomeStyle {
  label: string;
  detail: string;
  tone: "waiting" | "good" | "bad" | "idle";
}

export const OUTCOME: Record<string, OutcomeStyle> = {
  declined: {
    label: "Awaiting your decision",
    detail: "The change set is prepared. Nothing has been recorded yet.",
    tone: "waiting",
  },
  finalized: {
    label: "Recorded",
    detail: "The change set was approved and written to the site's history.",
    tone: "good",
  },
  aborted: {
    label: "Could not compare",
    detail:
      "The two visits could not be aligned, so no change set was produced. " +
      "Reporting differences here would report alignment error as change.",
    tone: "bad",
  },
  step_limit: {
    label: "Stopped at the step limit",
    detail: "The agent reached its iteration ceiling. What it found is reported as-is.",
    tone: "idle",
  },
  brain_failed: {
    label: "Decision layer failed",
    detail: "The optional model layer failed and the run stopped rather than guessing.",
    tone: "bad",
  },
  running: { label: "Running", detail: "", tone: "idle" },
};

export function outcomeStyle(outcome: string): OutcomeStyle {
  return OUTCOME[outcome] ?? OUTCOME.running!;
}

export const COVERAGE_LABEL: Record<string, string> = {
  observed: "Well covered",
  weak: "Partly covered",
  unobserved: "Barely covered",
};

export const PHASE_LABEL: Record<string, string> = {
  observe: "Observe",
  decide: "Decide",
  act: "Act",
  gate: "Approval",
  finish: "Finish",
};

export const PHASE_ORDER = ["observe", "decide", "act", "gate", "finish"] as const;

// --- Formatting -----------------------------------------------------------

export function pct(value: number, digits = 0): string {
  return `${(value * 100).toFixed(digits)}%`;
}

export function px(value: number | null, digits = 2): string {
  if (value === null || !Number.isFinite(value)) return "not measured";
  return `${value.toFixed(digits)} px`;
}

export function ms(value: number): string {
  if (value >= 1000) return `${(value / 1000).toFixed(2)} s`;
  return `${Math.round(value)} ms`;
}

export function confidenceWord(value: number): string {
  if (value >= 0.8) return "high";
  if (value >= 0.6) return "moderate";
  if (value >= 0.4) return "low";
  return "very low";
}

export function relativeTime(iso: string): string {
  const then = new Date(iso).getTime();
  if (!Number.isFinite(then)) return iso;
  const seconds = Math.max(0, (Date.now() - then) / 1000);
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)} min ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} h ago`;
  return `${Math.floor(seconds / 86400)} d ago`;
}

export function cx(...parts: Array<string | false | null | undefined>): string {
  return parts.filter(Boolean).join(" ");
}
