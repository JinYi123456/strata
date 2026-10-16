"use client";

/**
 * A site's history: every scan, including the ones that changed nothing and the
 * ones a person declined.
 *
 * A ledger that only kept successes would be a highlight reel. What makes it
 * useful is that it is complete - "we looked on Tuesday and found nothing" is a
 * fact about the site, and so is "we looked on Wednesday and did not trust what
 * we saw".
 */

import { motion } from "framer-motion";
import { cx, kindStyle, KIND_ORDER, relativeTime } from "@/lib/design";
import type { LedgerEntry } from "@/lib/types";
import { EmptyState, Panel, PanelHead, Skeleton } from "./ui";

export function LedgerHistory({
  entries,
  loading,
  onRefresh,
}: {
  entries: LedgerEntry[];
  loading: boolean;
  onRefresh: () => void;
}) {
  return (
    <Panel className="flex min-h-0 flex-col">
      <PanelHead
        title="History"
        hint="Every scan of this site, in the order it happened."
        right={
          <button
            type="button"
            onClick={onRefresh}
            className="font-mono text-2xs text-ink-faint underline-offset-2 hover:text-ink hover:underline"
          >
            refresh
          </button>
        }
      />

      <div className="min-h-0 flex-1 overflow-y-auto border-t border-ground-edge">
        {loading ? (
          <div className="space-y-2 p-4">
            <Skeleton className="h-12 w-full" />
            <Skeleton className="h-12 w-full" />
          </div>
        ) : entries.length === 0 ? (
          <EmptyState
            title="Nothing recorded yet"
            body="Approved change sets appear here, newest last. A scan you withhold is not recorded."
          />
        ) : (
          <ol className="divide-y divide-ground-edge">
            {entries.map((entry, i) => (
              <motion.li
                key={entry.entry_id}
                initial={{ opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: Math.min(i * 0.04, 0.3) }}
                className="px-4 py-3"
              >
                <div className="flex items-center gap-2">
                  <span
                    className={cx(
                      "h-1.5 w-1.5 shrink-0 rounded-full",
                      entry.approved ? "bg-accent" : "bg-ink-faint",
                    )}
                    aria-hidden="true"
                  />
                  <span className="text-xs font-medium text-ink">
                    {entry.approved ? "Recorded" : "Withheld"}
                  </span>
                  <span className="ml-auto font-mono text-2xs text-ink-faint">
                    {relativeTime(entry.recorded_at)}
                  </span>
                </div>

                <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 pl-3.5">
                  <span className="font-mono text-2xs text-ink-faint">
                    scan #{entry.scan_index} · {entry.outcome}
                  </span>
                  <span className="font-mono text-2xs text-ink-faint">
                    alignment {Math.round(entry.registration_confidence * 100)}%
                  </span>
                  <EventTotals totals={entry.event_totals} />
                </div>

                {entry.note ? (
                  <p className="mt-1.5 border-l border-ground-edge pl-3 text-2xs italic leading-relaxed text-ink-muted">
                    {entry.note}
                  </p>
                ) : null}
              </motion.li>
            ))}
          </ol>
        )}
      </div>
    </Panel>
  );
}

export function EventTotals({ totals }: { totals: Record<string, number> }) {
  const shown = KIND_ORDER.filter((k) => (totals[k] ?? 0) > 0);
  const sum = Object.values(totals).reduce((a, b) => a + b, 0);

  if (sum === 0) {
    return <span className="font-mono text-2xs text-ink-faint">no changes</span>;
  }
  return (
    <span className="flex items-center gap-2">
      {shown.map((k) => {
        const style = kindStyle(k);
        return (
          <span
            key={k}
            className={cx("font-mono text-2xs", style.text)}
            title={`${style.label}: ${totals[k]}`}
          >
            {style.glyph}
            <span className="tabular ml-1">{totals[k]}</span>
          </span>
        );
      })}
    </span>
  );
}
