"use client";

/**
 * The canonical frame, and the four ways of looking at it.
 *
 * The whole product rests on one idea: two visits can be placed in the same
 * coordinate system, and once they are, the comparison is meaningful. So the
 * canvas is the hero, and every mode is a way of interrogating the same aligned
 * pair rather than a separate picture.
 *
 * Two implementation details are worth stating.
 *
 * **Boxes are drawn in canonical pixels, for free.** The wrapper's aspect ratio
 * is set from the image's own dimensions and the overlay is an SVG whose
 * `viewBox` is the canonical frame. That means a box at canonical `(x, y, w, h)`
 * is drawn at exactly those coordinates with no arithmetic and no chance of a
 * scaling bug - the most common way a detection UI silently lies.
 *
 * **The wipe is only offered between the two aligned frames.** A before/after
 * slider is meaningless on unaligned images, so it is not a general-purpose
 * comparison widget here; it is the payoff for having registered the pair.
 */

import { AnimatePresence, motion } from "framer-motion";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { cx, kindStyle } from "@/lib/design";
import type { ChangeEvent, ImageRef, PanelName } from "@/lib/types";
import { EmptyState } from "./ui";

type Mode = "wipe" | "overlay" | "side_by_side" | "guidance" | "baseline";

const MODES: Array<{ id: Mode; label: string; hint: string }> = [
  { id: "wipe", label: "Compare", hint: "Scrub between the two aligned visits" },
  { id: "overlay", label: "Changes", hint: "Every change, drawn where it happened" },
  { id: "side_by_side", label: "Side by side", hint: "Baseline, rescan and diff together" },
  { id: "guidance", label: "Coverage", hint: "What the capture did not cover" },
  { id: "baseline", label: "Baseline", hint: "The recorded reference view" },
];

export function ImageViewer({
  images,
  events,
  selectedId,
  onSelect,
  busy = false,
  busyLabel,
}: {
  images: Partial<Record<PanelName, ImageRef>>;
  events: ChangeEvent[];
  selectedId: string | null;
  onSelect: (id: string | null) => void;
  busy?: boolean;
  busyLabel?: string;
}) {
  const [mode, setMode] = useState<Mode>("overlay");
  const overlay = images.overlay;
  const baseline = images.baseline;
  const rescan = images.rescan;

  // The wipe is the reason the alignment work exists, so it is offered first
  // whenever both aligned frames are present. Before a scan there is only the
  // baseline, and showing it is far better than showing an empty frame - the
  // operator should be able to see what was recorded.
  useEffect(() => {
    if (baseline && rescan) setMode("wipe");
    else if (overlay) setMode("overlay");
    else if (baseline) setMode("baseline");
  }, [baseline, rescan, overlay]);

  // Selecting a change from the ledger has to actually show it. Without this,
  // "locate" set a selection that was drawn nowhere, because the boxes only
  // exist in the overlay view - so the control appeared to do nothing at all.
  useEffect(() => {
    if (selectedId) setMode("overlay");
  }, [selectedId]);

  const available = useMemo(
    () =>
      MODES.filter((m) => {
        if (m.id === "wipe") return Boolean(baseline && rescan);
        return Boolean(images[m.id]);
      }),
    [images, baseline, rescan],
  );

  const frame = overlay ?? baseline;
  const size = frame ? { width: frame.width, height: frame.height } : null;

  if (busy) {
    return (
      <Frame size={size ?? undefined}>
        <div className="flex h-full flex-col items-center justify-center gap-4">
          <div className="relative h-16 w-16">
            <span className="absolute inset-0 rounded-full border border-accent/40 animate-pulseRing" />
            <span
              className="absolute inset-0 rounded-full border border-accent/30 animate-pulseRing"
              style={{ animationDelay: "0.7s" }}
            />
            <span className="absolute inset-0 flex items-center justify-center rounded-full border border-accent/50 bg-accent/10 text-accent">
              <ScanGlyph />
            </span>
          </div>
          <div className="text-center">
            <div className="text-sm font-medium text-ink">{busyLabel ?? "Scanning"}</div>
            <div className="mt-1 font-mono text-2xs text-ink-faint">
              registering · differencing · resolving events
            </div>
          </div>
        </div>
      </Frame>
    );
  }

  if (!size) {
    return (
      <Frame>
        <EmptyState
          title="No capture to show"
          body="This session has no baseline frame stored, so there is nothing to display yet."
        />
      </Frame>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-1.5">
        {available.map((m) => (
          <button
            key={m.id}
            type="button"
            onClick={() => setMode(m.id)}
            title={m.hint}
            aria-pressed={mode === m.id}
            className={cx(
              "relative rounded-md px-2.5 py-1.5 text-xs font-medium transition-colors",
              mode === m.id
                ? "text-ink"
                : "text-ink-muted hover:bg-white/[0.04] hover:text-ink",
            )}
          >
            {mode === m.id ? (
              <motion.span
                layoutId="viewer-tab"
                className="absolute inset-0 rounded-md border border-accent/40 bg-accent/10"
                transition={{ type: "spring", stiffness: 420, damping: 34 }}
              />
            ) : null}
            <span className="relative">{m.label}</span>
          </button>
        ))}
        <span className="ml-auto hidden font-mono text-2xs text-ink-faint sm:block">
          {size.width} × {size.height} canonical
        </span>
      </div>

      <Frame size={size}>
        <AnimatePresence mode="wait" initial={false}>
          <motion.div
            key={mode}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.18 }}
            className="absolute inset-0"
          >
            {mode === "wipe" && baseline && rescan ? (
              <Wipe baseline={baseline} rescan={rescan} />
            ) : null}

            {mode === "overlay" && overlay ? (
              <div className="relative h-full w-full">
                <img
                  src={overlay.url}
                  alt="Change overlay: the baseline frame with detected changes marked"
                  className="absolute inset-0 h-full w-full"
                />
                <BoxLayer
                  size={size}
                  events={events}
                  selectedId={selectedId}
                  onSelect={onSelect}
                />
              </div>
            ) : null}

            {mode === "baseline" && images.baseline ? (
              <img
                src={images.baseline.url}
                alt="The recorded baseline frame"
                className="absolute inset-0 h-full w-full"
              />
            ) : null}

            {mode === "side_by_side" && images.side_by_side ? (
              <img
                src={images.side_by_side.url}
                alt="Baseline, rescan and diff shown side by side"
                className="absolute inset-0 h-full w-full object-contain"
              />
            ) : null}

            {mode === "guidance" && images.guidance ? (
              <img
                src={images.guidance.url}
                alt="Coverage overlay with the recommended next capture"
                className="absolute inset-0 h-full w-full"
              />
            ) : null}
          </motion.div>
        </AnimatePresence>
      </Frame>

      {mode === "wipe" ? (
        <p className="text-2xs leading-relaxed text-ink-faint">
          Both visits are drawn in the same frame, so the divider compares like with like.
          Without registration this slider would only compare two differently-framed photographs.
        </p>
      ) : null}
      {mode === "overlay" && events.length > 0 ? (
        <p className="text-2xs leading-relaxed text-ink-faint">
          Click a marker to inspect that change. Boxes are drawn in canonical pixels.
        </p>
      ) : null}
    </div>
  );
}

function Frame({
  children,
  size,
}: {
  children: React.ReactNode;
  size?: { width: number; height: number };
}) {
  // The frame takes the image's own aspect ratio rather than a fixed one, so
  // nothing is stretched and the SVG overlay maps to the pixels exactly.
  return (
    <div
      className="relative w-full overflow-hidden rounded-lg border border-ground-edge bg-ground"
      style={{ aspectRatio: size ? `${size.width} / ${size.height}` : "4 / 3" }}
    >
      {children}
    </div>
  );
}

// --- The aligned before/after ---------------------------------------------

function Wipe({ baseline, rescan }: { baseline: ImageRef; rescan: ImageRef }) {
  const [split, setSplit] = useState(50);
  const ref = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  const setFromClientX = useCallback((clientX: number) => {
    const el = ref.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const next = ((clientX - rect.left) / rect.width) * 100;
    setSplit(Math.max(0, Math.min(100, next)));
  }, []);

  useEffect(() => {
    const move = (e: PointerEvent) => {
      if (!dragging.current) return;
      setFromClientX(e.clientX);
    };
    const up = () => {
      dragging.current = false;
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    return () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
  }, [setFromClientX]);

  return (
    <div
      ref={ref}
      className="relative h-full w-full cursor-ew-resize touch-none select-none"
      onPointerDown={(e) => {
        dragging.current = true;
        setFromClientX(e.clientX);
      }}
      role="slider"
      aria-label="Compare baseline and rescan"
      aria-valuenow={Math.round(split)}
      aria-valuemin={0}
      aria-valuemax={100}
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "ArrowLeft") setSplit((s) => Math.max(0, s - 2));
        if (e.key === "ArrowRight") setSplit((s) => Math.min(100, s + 2));
      }}
    >
      <img
        src={rescan.url}
        alt="Rescan"
        className="absolute inset-0 h-full w-full"
        draggable={false}
      />
      <div className="absolute inset-0" style={{ clipPath: `inset(0 ${100 - split}% 0 0)` }}>
        <img
          src={baseline.url}
          alt="Baseline"
          className="absolute inset-0 h-full w-full"
          draggable={false}
        />
      </div>

      {/* Labels, so it is never ambiguous which side is which. */}
      <span className="pointer-events-none absolute left-3 top-3 rounded bg-black/65 px-2 py-1 font-mono text-2xs uppercase tracking-wider text-ink backdrop-blur-sm">
        Baseline
      </span>
      <span className="pointer-events-none absolute right-3 top-3 rounded bg-black/65 px-2 py-1 font-mono text-2xs uppercase tracking-wider text-accent backdrop-blur-sm">
        Rescan
      </span>

      <div
        className="pointer-events-none absolute inset-y-0 w-px bg-accent/90"
        style={{ left: `${split}%` }}
      >
        <span className="absolute top-1/2 flex h-8 w-8 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full border border-accent/70 bg-ground/90 shadow-glow backdrop-blur-sm">
          <svg viewBox="0 0 24 24" className="h-4 w-4 text-accent" fill="none" aria-hidden="true">
            <path
              d="M9 7 5 12l4 5M15 7l4 5-4 5"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </span>
      </div>

      <span className="sr-only">
        Showing {Math.round(split)} percent of the baseline frame
      </span>
    </div>
  );
}

// --- Interactive boxes ----------------------------------------------------

function BoxLayer({
  size,
  events,
  selectedId,
  onSelect,
}: {
  size: { width: number; height: number };
  events: ChangeEvent[];
  selectedId: string | null;
  onSelect: (id: string | null) => void;
}) {
  return (
    <svg
      viewBox={`0 0 ${size.width} ${size.height}`}
      preserveAspectRatio="none"
      className="absolute inset-0 h-full w-full"
      onClick={(e) => {
        if (e.target === e.currentTarget) onSelect(null);
      }}
    >
      {events.map((event, i) => {
        const style = kindStyle(event.kind);
        const active = event.event_id === selectedId;
        const { x, y, width, height } = event.bbox;
        const label = style.glyph;

        return (
          <g
            key={event.event_id}
            className="cursor-pointer"
            onClick={(e) => {
              e.stopPropagation();
              onSelect(active ? null : event.event_id);
            }}
          >
            <motion.rect
              x={x}
              y={y}
              width={width}
              height={height}
              fill={active ? style.hex : "transparent"}
              fillOpacity={active ? 0.16 : 0}
              stroke={style.hex}
              strokeWidth={active ? 7 : 4}
              strokeDasharray={event.uncertain ? "14 10" : undefined}
              initial={{ opacity: 0, scale: 0.94 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.04 * i, duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
              style={{ transformOrigin: `${x + width / 2}px ${y + height / 2}px` }}
            />
            {/* A marker pinned to the top-left corner, carrying the kind glyph.
                Reading the glyph is faster than reading a label, and it works
                without relying on colour. */}
            <g transform={`translate(${x}, ${y})`}>
              <rect
                x={-34}
                y={-34}
                width={68}
                height={68}
                rx={10}
                fill="#080B0F"
                fillOpacity={0.85}
                stroke={style.hex}
                strokeWidth={3}
              />
              <text
                x={0}
                y={0}
                textAnchor="middle"
                dominantBaseline="central"
                fill={style.hex}
                fontSize={44}
                fontFamily="ui-monospace, monospace"
              >
                {label}
              </text>
            </g>
          </g>
        );
      })}
    </svg>
  );
}

function ScanGlyph() {
  return (
    <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" aria-hidden="true">
      <path
        d="M4 8V5.5A1.5 1.5 0 0 1 5.5 4H8M16 4h2.5A1.5 1.5 0 0 1 20 5.5V8M20 16v2.5a1.5 1.5 0 0 1-1.5 1.5H16M8 20H5.5A1.5 1.5 0 0 1 4 18.5V16M4 12h16"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
    </svg>
  );
}
