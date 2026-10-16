"use client";

/**
 * The change set: what was found, how sure, and what it was based on.
 *
 * This is the artefact the product exists to produce, so it is presented as a
 * record rather than a chart. Each row states the kind in words and a glyph,
 * gives a confidence with its plain-language reading, and can be opened to show
 * the measurements behind it.
 *
 * An event the agent declined to resolve is shown as unresolved rather than
 * quietly omitted or quietly included. "I looked and I am not sure" is a
 * finding, and a tool that hides it is worse than one that finds less.
 */

import { AnimatePresence, motion } from "framer-motion";
import { useState } from "react";
import { cx, kindStyle, px } from "@/lib/design";
import type { ChangeEvent } from "@/lib/types";
import { ConfidenceMeter, EmptyState, Panel, PanelHead } from "./ui";

export function ChangeLedger({
  events,
  totals,
  selectedId,
  onSelect,
  minConfidence,
  onMinConfidenceChange,
}: {
  events: ChangeEvent[];
  totals: Record<string, number>;
  selectedId: string | null;
  onSelect: (id: string | null) => void;
  minConfidence: number;
  onMinConfidenceChange: (v: number) => void;
}) {
  const totalFound = Object.values(totals).reduce((a, b) => a + b, 0);
  const hidden = totalFound - events.length;

  return (
    <Panel className="flex min-h-0 flex-col">
      <PanelHead
        title="Change set"
        hint={
          totalFound === 0
            ? "Nothing changed between these visits."
            : `${totalFound} change${totalFound === 1 ? "" : "s"} found.`
        }
        right={
          totalFound > 0 ? (
            <div className="flex items-center gap-2">
              <label
                htmlFor="min-confidence"
                className="label cursor-pointer whitespace-nowrap"
                title="Hide low-confidence rows from this list"
              >
                floor
              </label>
              <input
                id="min-confidence"
                type="range"
                min={0}
                max={0.9}
                step={0.05}
                value={minConfidence}
                onChange={(e) => onMinConfidenceChange(Number(e.target.value))}
                className="h-1 w-20 cursor-pointer appearance-none rounded-full bg-white/10
                  accent-accent [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:w-3
                  [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full
                  [&::-webkit-slider-thumb]:bg-accent"
              />
            </div>
          ) : null
        }
      />

      {/* The totals are what the system found; the list below may be filtered.
          Saying so explicitly stops a filtered view from reading as the truth. */}
      {hidden > 0 ? (
        <p className="border-t border-ground-edge px-4 py-2 font-mono text-2xs text-ink-faint">
          {hidden} below the floor, not listed. The totals above are unfiltered.
        </p>
      ) : null}

      <div className="min-h-0 flex-1 overflow-y-auto border-t border-ground-edge">
        {totalFound === 0 ? (
          <EmptyState
            title="No changes detected"
            body={
              <>
                Every region the system could compare matched the baseline. This is the
                outcome that matters most in the field: a quiet result has to mean a quiet
                site, not a blind detector.
              </>
            }
          />
        ) : events.length === 0 ? (
          <EmptyState
            title="All changes are below the floor"
            body={`${totalFound} were found, but none reach ${Math.round(
              minConfidence * 100,
            )}% confidence. Lower the floor to see them.`}
          />
        ) : (
          <ul className="divide-y divide-ground-edge">
            {events.map((event, i) => (
              <EventRow
                key={event.event_id}
                event={event}
                index={i}
                active={event.event_id === selectedId}
                onToggle={() => onSelect(event.event_id === selectedId ? null : event.event_id)}
              />
            ))}
          </ul>
        )}
      </div>
    </Panel>
  );
}

function EventRow({
  event,
  index,
  active,
  onToggle,
}: {
  event: ChangeEvent;
  index: number;
  active: boolean;
  onToggle: () => void;
}) {
  const [open, setOpen] = useState(false);
  const style = kindStyle(event.kind);

  return (
    <motion.li
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: Math.min(index * 0.035, 0.4), duration: 0.28 }}
      className={cx("transition-colors", active ? "bg-white/[0.035]" : "hover:bg-white/[0.02]")}
    >
      <div className="flex items-start gap-3 px-4 py-3">
        <button
          type="button"
          onClick={onToggle}
          className={cx(
            "mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded border",
            "font-mono text-sm leading-none transition-colors",
            style.border,
            style.bg,
            style.text,
          )}
          aria-label={`Locate ${style.label} on the frame`}
          title="Show this change on the frame"
        >
          {style.glyph}
        </button>

        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
            <span className={cx("text-sm font-semibold", style.text)}>{style.label}</span>
            <span className="text-2xs text-ink-faint">{style.gloss}</span>
            {event.uncertain ? (
              <span
                className="chip border-kind-uncertain/40 text-kind-uncertain"
                title="The agent recorded this as unresolved rather than assuming it"
              >
                unresolved
              </span>
            ) : null}
          </div>

          <p className="mt-1 text-xs leading-relaxed text-ink-muted">{event.summary}</p>

          <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1.5">
            <ConfidenceMeter value={event.confidence} />
            <span className="tabular font-mono text-2xs text-ink-faint">
              {event.bbox.width}×{event.bbox.height} px
            </span>
            {event.displacement_px > 0.5 ? (
              <span className="tabular font-mono text-2xs text-ink-faint">
                moved {px(event.displacement_px, 1)}
              </span>
            ) : null}
            <button
              type="button"
              onClick={() => setOpen((v) => !v)}
              aria-expanded={open}
              className="ml-auto font-mono text-2xs text-ink-faint underline-offset-2 hover:text-ink hover:underline"
            >
              {open ? "hide evidence" : "evidence"}
            </button>
          </div>
        </div>
      </div>

      <AnimatePresence initial={false}>
        {open ? (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.22, ease: [0.16, 1, 0.3, 1] }}
            className="overflow-hidden"
          >
            <Evidence event={event} />
          </motion.div>
        ) : null}
      </AnimatePresence>
    </motion.li>
  );
}

/**
 * The measurements behind a verdict.
 *
 * Shown verbatim and unformatted where possible: these are the numbers a
 * reviewer would ask for, and rounding them into prose would make them harder to
 * check rather than easier to read.
 */
function Evidence({ event }: { event: ChangeEvent }) {
  const entries = Object.entries(event.evidence ?? {});
  const b = event.bbox;

  return (
    <div className="border-t border-ground-edge/60 bg-black/20 px-4 py-3">
      <dl className="grid grid-cols-2 gap-x-6 gap-y-1.5 font-mono text-2xs sm:grid-cols-3">
        <Field label="event" value={event.event_id} />
        <Field label="centroid" value={`${event.centroid[0]}, ${event.centroid[1]}`} />
        <Field label="box" value={`x${b.x} y${b.y} w${b.width} h${b.height}`} />
        <Field label="displacement" value={px(event.displacement_px, 2)} />
        <Field label="appearance Δ" value={event.appearance_delta.toFixed(3)} />
        {entries.map(([k, v]) => (
          <Field key={k} label={k} value={renderValue(v)} />
        ))}
      </dl>
    </div>
  );
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0">
      <dt className="truncate text-ink-faint" title={label}>
        {label}
      </dt>
      <dd className="tabular truncate text-ink" title={value}>
        {value}
      </dd>
    </div>
  );
}

function renderValue(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(3);
  if (typeof v === "boolean") return v ? "true" : "false";
  if (typeof v === "string") return v;
  return JSON.stringify(v);
}
