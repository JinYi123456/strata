"use client";

/**
 * Two panels about how much to trust the scan, rather than what it found.
 *
 * **Coverage** answers "did we even look?". A confident "nothing changed" over
 * an area nobody photographed is the most dangerous output this system could
 * produce, so the guidance is given the same prominence as the change set.
 *
 * **Registration** answers "are these two frames actually aligned?". Every
 * number below the diff is meaningless if the alignment is wrong, so the
 * evidence for the alignment is shown rather than assumed - including the notes
 * the geometry stage wrote about its own choices.
 */

import { motion } from "framer-motion";
import { COVERAGE_LABEL, cx, pct, px } from "@/lib/design";
import type { Coverage, Registration, ViewRequest } from "@/lib/types";
import { Callout, Chip, ConfidenceMeter, Panel, PanelHead, Stat } from "./ui";

// --- Coverage and guidance -------------------------------------------------

const COVERAGE_TONE: Record<string, { bar: string; text: string }> = {
  observed: { bar: "bg-accent", text: "text-accent" },
  weak: { bar: "bg-kind-uncertain", text: "text-kind-uncertain" },
  unobserved: { bar: "bg-kind-missing", text: "text-kind-missing" },
};

export function CoveragePanel({
  coverage,
  guidance,
}: {
  coverage: Coverage | null;
  guidance: ViewRequest[];
}) {
  if (!coverage || typeof coverage.observed_fraction !== "number") {
    return (
      <Panel>
        <PanelHead title="Coverage" hint="How much of the site these captures actually saw." />
        <div className="border-t border-ground-edge px-4 py-4 text-xs text-ink-faint">
          Not measured for this scan.
        </div>
      </Panel>
    );
  }

  const tone = COVERAGE_TONE[coverage.status] ?? COVERAGE_TONE.weak!;
  const fraction = Math.max(0, Math.min(1, coverage.observed_fraction));

  return (
    <Panel>
      <PanelHead
        title="Coverage"
        hint="How much of the site these captures actually saw."
        right={
          <span className={cx("font-mono text-2xs uppercase tracking-wider", tone.text)}>
            {COVERAGE_LABEL[coverage.status] ?? coverage.status}
          </span>
        }
      />
      <div className="border-t border-ground-edge px-4 py-3.5">
        <div className="flex items-baseline justify-between">
          <span className="tabular font-mono text-2xl text-ink">{pct(fraction)}</span>
          <span className="font-mono text-2xs text-ink-faint">
            {coverage.frame_count} frame{coverage.frame_count === 1 ? "" : "s"}
            {coverage.grid ? ` · ${coverage.grid[0]}×${coverage.grid[1]} grid` : ""}
          </span>
        </div>
        <div className="mt-2.5 h-1.5 overflow-hidden rounded-full bg-white/[0.07]">
          <motion.div
            className={cx("h-full rounded-full", tone.bar)}
            initial={{ width: 0 }}
            animate={{ width: `${fraction * 100}%` }}
            transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
          />
        </div>

        {coverage.status !== "observed" ? (
          <p className="mt-3 text-2xs leading-relaxed text-ink-muted">
            Parts of the frame were seen too thinly to compare. Changes there are
            deliberately not reported, because the system cannot distinguish a real
            change from an alignment artefact in a region it barely saw.
          </p>
        ) : null}
      </div>

      <div className="border-t border-ground-edge px-4 py-3.5">
        <div className="label mb-2.5">Recommended next capture</div>
        {guidance.length === 0 ? (
          <p className="text-xs text-ink-faint">
            Nothing to ask for. Coverage is enough to compare this site.
          </p>
        ) : (
          <ol className="space-y-2">
            {guidance.map((req, i) => (
              <GuidanceCard key={`${req.target_cell.join("-")}-${i}`} request={req} index={i} />
            ))}
          </ol>
        )}
      </div>
    </Panel>
  );
}

function GuidanceCard({ request, index }: { request: ViewRequest; index: number }) {
  return (
    <motion.li
      initial={{ opacity: 0, y: 5 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.06 * index, duration: 0.28 }}
      className="flex items-start gap-3 rounded-md border border-ground-edge bg-white/[0.015] px-3 py-2.5"
    >
      {/* The instruction is the point, so it leads; the reason follows it. */}
      <span className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded border border-accent/40 bg-accent/10 text-accent">
        <DirectionArrow direction={request.direction} />
      </span>
      <div className="min-w-0">
        <div className="text-xs font-semibold capitalize text-ink">{request.direction}</div>
        <div className="mt-0.5 text-2xs leading-relaxed text-ink-muted">{request.reason}</div>
        <div className="mt-1 flex items-center gap-3 font-mono text-2xs text-ink-faint">
          <span>cell {request.target_cell.join(",")}</span>
          <span>gap {pct(request.deficit)}</span>
        </div>
      </div>
    </motion.li>
  );
}

function DirectionArrow({ direction }: { direction: string }) {
  const d = direction.toLowerCase();
  const rotation = d.includes("left")
    ? 180
    : d.includes("right")
      ? 0
      : d.includes("up") || d.includes("closer") || d.includes("step in")
        ? -90
        : d.includes("down") || d.includes("back")
          ? 90
          : 0;
  return (
    <svg
      viewBox="0 0 24 24"
      className="h-3.5 w-3.5"
      fill="none"
      aria-hidden="true"
      style={{ transform: `rotate(${rotation}deg)` }}
    >
      <path
        d="M5 12h14M13 6l6 6-6 6"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

// --- Registration ----------------------------------------------------------

export function RegistrationPanel({ registration }: { registration: Registration }) {
  const ok = registration.model !== "identity" && registration.inlier_count > 0;

  return (
    <Panel>
      <PanelHead
        title="Alignment"
        hint="The registration every comparison above depends on."
        right={
          <Chip
            className={ok ? "border-accent/40 text-accent" : "border-kind-missing/40 text-kind-missing"}
            title={ok ? "The two visits were placed in one frame" : "No usable alignment was found"}
          >
            {registration.model}
          </Chip>
        }
      />

      {!ok ? (
        <div className="border-t border-ground-edge px-4 py-3.5">
          <Callout tone="bad" title="These visits could not be aligned">
            No usable transform was found between them. Any differences reported from an
            unaligned pair would be alignment error, so none are. Record a fresh baseline
            with more overlapping detail, or capture more of the scene in each visit.
          </Callout>
        </div>
      ) : null}

      <div className="grid grid-cols-2 gap-4 border-t border-ground-edge px-4 py-3.5 sm:grid-cols-4">
        <Stat
          label="inliers"
          value={registration.inlier_count}
          sub={`of ${registration.match_count} matches`}
        />
        <Stat label="inlier ratio" value={pct(registration.inlier_ratio)} />
        <Stat
          label="reproj error"
          value={px(registration.reproj_error_px)}
          sub={registration.reproj_error_px === null ? "no model to measure" : "mean, inliers"}
        />
        <div className="min-w-0">
          <div className="label">confidence</div>
          <div className="mt-1.5">
            <ConfidenceMeter value={registration.confidence} />
          </div>
        </div>
      </div>

      {registration.notes.length > 0 ? (
        <div className="border-t border-ground-edge px-4 py-3">
          <div className="label mb-1.5">What the geometry stage recorded</div>
          <ul className="space-y-1">
            {registration.notes.map((note, i) => (
              <li key={i} className="flex gap-2 text-2xs leading-relaxed text-ink-muted">
                <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-ink-faint" />
                <span>{note}</span>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </Panel>
  );
}
