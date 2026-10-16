"use client";

/**
 * The human decision, given the weight it deserves.
 *
 * Nothing this system finds is recorded until someone says so. That is the whole
 * safety story, so the gate is a standing part of the interface rather than a
 * dialog that interrupts - the operator can read the change set, look at the
 * frames, and only then decide.
 *
 * The two buttons are not symmetrical. Approving writes to the site's history,
 * so it is the quieter, more deliberate control; withholding is always available
 * and never destructive. A note is offered but not required, because forcing
 * someone to justify "no" teaches them to type "no" faster.
 */

import { AnimatePresence, motion } from "framer-motion";
import { useState } from "react";
import { cx, outcomeStyle, relativeTime } from "@/lib/design";
import type { LedgerEntry } from "@/lib/types";
import { Button, Callout } from "./ui";

const TONE_RING: Record<string, string> = {
  waiting: "border-kind-uncertain/35 bg-kind-uncertain/[0.06]",
  good: "border-accent/35 bg-accent/[0.06]",
  bad: "border-kind-missing/35 bg-kind-missing/[0.06]",
  idle: "border-ground-edge bg-white/[0.015]",
};

export function ApprovalGate({
  outcome,
  eventCount,
  committed,
  entry,
  onDecide,
  busy,
  error,
}: {
  outcome: string;
  eventCount: number;
  committed: boolean;
  entry: LedgerEntry | null;
  onDecide: (approved: boolean, note: string) => void;
  busy: boolean;
  error: string | null;
}) {
  const [note, setNote] = useState("");
  const style = outcomeStyle(outcome);
  const decidable = outcome === "declined";

  // Once recorded, the gate becomes a receipt. Re-showing the buttons would
  // imply the decision is still open when it is not.
  if (committed && entry) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        className={cx("rounded-lg border px-4 py-3.5", TONE_RING.good)}
      >
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
          <span className="flex h-5 w-5 items-center justify-center rounded-full border border-accent/50 bg-accent/15 text-[11px] text-accent">
            ✓
          </span>
          <span className="text-sm font-semibold text-ink">Recorded</span>
          <span className="font-mono text-2xs text-ink-faint">
            {entry.entry_id} · {relativeTime(entry.recorded_at)} · by {entry.decision_made_by}
          </span>
        </div>
        {entry.note ? (
          <p className="mt-2 border-l-2 border-accent/30 pl-3 text-xs italic leading-relaxed text-ink-muted">
            “{entry.note}”
          </p>
        ) : null}
      </motion.div>
    );
  }

  return (
    <div className={cx("rounded-lg border px-4 py-3.5", TONE_RING[style.tone])}>
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold text-ink">{style.label}</span>
            {decidable ? (
              <span className="chip border-kind-uncertain/40 text-kind-uncertain">
                {eventCount} change{eventCount === 1 ? "" : "s"} pending
              </span>
            ) : null}
          </div>
          <p className="mt-1 max-w-xl text-xs leading-relaxed text-ink-muted">{style.detail}</p>
        </div>

        {decidable ? (
          <div className="flex w-full shrink-0 flex-col gap-2 sm:w-auto sm:flex-row sm:items-center">
            <label htmlFor="gate-note" className="sr-only">
              Note
            </label>
            <input
              id="gate-note"
              value={note}
              onChange={(e) => setNote(e.target.value)}
              placeholder="Note (optional)"
              maxLength={500}
              className="w-full rounded-md border border-ground-edge bg-ground px-3 py-2 text-xs
                text-ink placeholder:text-ink-faint focus:border-accent/50 sm:w-56"
            />
            <div className="flex gap-2">
              <Button
                variant="default"
                disabled={busy}
                onClick={() => onDecide(false, note)}
                title="Nothing will be written to the site's history"
              >
                Withhold
              </Button>
              <Button
                variant="primary"
                loading={busy}
                onClick={() => onDecide(true, note)}
                title="Write this change set into the site's history"
              >
                Record change set
              </Button>
            </div>
          </div>
        ) : null}
      </div>

      {!decidable && outcome !== "finalized" ? (
        <div className="mt-3">
          <Callout
            tone={style.tone === "bad" ? "bad" : "info"}
            title={
              outcome === "aborted"
                ? "No change set was produced"
                : "There is nothing to record for this scan"
            }
          >
            The trace below shows exactly what the agent did and why it stopped.
          </Callout>
        </div>
      ) : null}

      <AnimatePresence>
        {error ? (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="mt-3"
          >
            <Callout tone="bad" title="Could not record that">
              {error}
            </Callout>
          </motion.div>
        ) : null}
      </AnimatePresence>
    </div>
  );
}
