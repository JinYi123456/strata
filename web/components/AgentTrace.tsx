"use client";

/**
 * The agent's reasoning, as a timeline rather than a log.
 *
 * A log dump would be easier to write and useless to read. What a reviewer needs
 * to know is: what did it look at, what did it conclude, what did it decide to
 * do, and who decided. So steps are grouped by phase and each one leads with its
 * rationale in plain language, with the raw arguments available on demand.
 *
 * The `decided_by` badge is deliberately prominent. It answers the question that
 * matters about any system like this - was this a rule, a model, or a person? -
 * and it answers it per step, because a single run can involve all three.
 */

import { AnimatePresence, motion } from "framer-motion";
import { useState } from "react";
import { cx, ms as fmtMs, PHASE_LABEL, PHASE_ORDER } from "@/lib/design";
import type { Trace, TraceStep } from "@/lib/types";
import { Chip, Panel, PanelHead } from "./ui";

const PHASE_TONE: Record<string, string> = {
  observe: "border-kind-state/45 text-kind-state",
  decide: "border-accent/45 text-accent",
  act: "border-kind-moved/45 text-kind-moved",
  gate: "border-kind-uncertain/45 text-kind-uncertain",
  finish: "border-ink-faint/40 text-ink-muted",
};

/** Who or what produced a decision, in words. */
function decidedByLabel(value: string | null): { text: string; tone: string } {
  if (!value) return { text: "system", tone: "text-ink-faint border-ink-faint/30" };
  if (value === "human")
    return { text: "human approval", tone: "text-kind-uncertain border-kind-uncertain/40" };
  if (value === "loop") return { text: "loop control", tone: "text-ink-muted border-ink-faint/30" };
  if (value === "policy-rules")
    return { text: "deterministic rules", tone: "text-accent border-accent/40" };
  return { text: value, tone: "text-kind-moved border-kind-moved/40" };
}

export function AgentTrace({ trace }: { trace: Trace }) {
  const [openAll, setOpenAll] = useState(false);

  const groups: Array<{ phase: string; steps: TraceStep[] }> = PHASE_ORDER.map((phase) => ({
    phase: phase as string,
    steps: trace.steps.filter((s) => s.phase === phase),
  })).filter((g) => g.steps.length > 0);

  // Any phase the core adds later still shows up rather than disappearing.
  const known = new Set<string>(PHASE_ORDER);
  const extra = trace.steps.filter((s) => !known.has(s.phase));
  if (extra.length) groups.push({ phase: "other", steps: extra });

  return (
    // Capped, and scrolled internally. A run can produce twenty steps, and
    // letting the panel grow to fit them makes the row it sits in three times
    // taller than the panels beside it - which buries the coverage and alignment
    // panels that a reader needs to judge the very reasoning being shown.
    <Panel className="flex max-h-[34rem] min-h-0 flex-col">
      <PanelHead
        title="How it decided"
        hint={`${trace.decision_count} decision${
          trace.decision_count === 1 ? "" : "s"
        }, ${trace.action_count} action${trace.action_count === 1 ? "" : "s"}.`}
        right={
          <button
            type="button"
            onClick={() => setOpenAll((v) => !v)}
            className="font-mono text-2xs text-ink-faint underline-offset-2 hover:text-ink hover:underline"
          >
            {openAll ? "collapse" : "expand all"}
          </button>
        }
      />

      <div className="flex flex-wrap items-center gap-1.5 border-t border-ground-edge px-4 py-2.5">
        {trace.brains_used.map((brain) => {
          const d = decidedByLabel(brain);
          return (
            <Chip key={brain} className={d.tone} title="The decision layer used for this run">
              {d.text}
            </Chip>
          );
        })}
        {trace.used_fallback ? (
          <Chip
            className="border-kind-uncertain/40 text-kind-uncertain"
            title="A configured model layer failed and the deterministic rules took over. The error is recorded in the step below."
          >
            fell back to rules
          </Chip>
        ) : null}
        {!trace.used_fallback ? (
          <Chip className="border-ink-faint/30 text-ink-faint" title="No model call was needed">
            no model call
          </Chip>
        ) : null}
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto border-t border-ground-edge px-4 py-3">
        <ol className="relative space-y-4">
          {groups.map((group, gi) => (
            <li key={group.phase}>
              <div className="mb-2 flex items-center gap-2">
                <span className="label">{PHASE_LABEL[group.phase] ?? group.phase}</span>
                <span className="h-px flex-1 bg-ground-edge" />
              </div>
              <ol className="space-y-1.5 border-l border-ground-edge pl-4">
                {group.steps.map((step) => (
                  <Step
                    key={`${group.phase}-${step.index}`}
                    step={step}
                    defaultOpen={openAll}
                    forceOpen={openAll}
                    delay={gi * 0.02}
                  />
                ))}
              </ol>
            </li>
          ))}
        </ol>
      </div>
    </Panel>
  );
}

function Step({
  step,
  defaultOpen,
  forceOpen,
  delay,
}: {
  step: TraceStep;
  defaultOpen: boolean;
  forceOpen: boolean;
  delay: number;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const isOpen = forceOpen || open;
  const who = decidedByLabel(step.decided_by);
  const tone = PHASE_TONE[step.phase] ?? PHASE_TONE.finish!;
  const hasDetail = Object.keys(step.args ?? {}).length > 0 || Object.keys(step.result ?? {}).length > 0;

  return (
    <motion.li
      initial={{ opacity: 0, x: -4 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay, duration: 0.25 }}
      className="relative"
    >
      {/* A node on the rail, so the sequence reads top to bottom at a glance. */}
      <span
        className={cx(
          "absolute -left-[21px] top-1.5 h-2 w-2 rounded-full border bg-ground",
          tone,
        )}
        aria-hidden="true"
      />

      <div className="rounded-md border border-ground-edge/70 bg-white/[0.012] px-3 py-2">
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
          {step.action ? (
            <span className="font-mono text-2xs uppercase tracking-wider text-ink">
              {step.action.replace(/_/g, " ")}
            </span>
          ) : (
            <span className="font-mono text-2xs uppercase tracking-wider text-ink-faint">
              step {step.index}
            </span>
          )}
          <span className={cx("rounded border px-1.5 py-px font-mono text-[10px]", who.tone)}>
            {who.text}
          </span>
          {step.duration_ms !== null && step.duration_ms > 1 ? (
            <span className="tabular ml-auto font-mono text-2xs text-ink-faint">
              {fmtMs(step.duration_ms)}
            </span>
          ) : null}
        </div>

        {step.rationale ? (
          <p className="mt-1.5 text-xs leading-relaxed text-ink-muted">{step.rationale}</p>
        ) : null}

        {hasDetail ? (
          <>
            <button
              type="button"
              onClick={() => setOpen((v) => !v)}
              aria-expanded={isOpen}
              className="mt-1.5 font-mono text-2xs text-ink-faint underline-offset-2 hover:text-ink hover:underline"
            >
              {isOpen ? "hide detail" : "detail"}
            </button>
            <AnimatePresence initial={false}>
              {isOpen ? (
                <motion.pre
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: "auto", opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.2 }}
                  className="mt-1.5 overflow-x-auto rounded border border-ground-edge/60 bg-black/30
                    px-2.5 py-2 font-mono text-[10px] leading-relaxed text-ink-muted"
                >
                  {JSON.stringify({ args: step.args, result: step.result }, null, 2)}
                </motion.pre>
              ) : null}
            </AnimatePresence>
          </>
        ) : null}
      </div>
    </motion.li>
  );
}
