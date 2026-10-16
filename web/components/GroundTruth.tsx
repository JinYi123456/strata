"use client";

/**
 * Detected changes against the changes that were actually made.
 *
 * Only ever rendered for a sample site, where the true answer is known. It is
 * included because a system that reports on the world should be judged on
 * whether it is right, and the fastest way to establish that is to hand over the
 * answer key rather than to describe the results approvingly.
 *
 * Matching is by overlap, and it is deliberately generous: a detection counts as
 * finding a change if its box overlaps the true box at all. Being strict here
 * would flatter the system by inflating the number of apparent misses, which is
 * the opposite of the point.
 */

import { cx, kindStyle } from "@/lib/design";
import type { ChangeEvent, GroundTruthEntry } from "@/lib/types";
import { Panel, PanelHead, Stat } from "./ui";

interface Box {
  x: number;
  y: number;
  width: number;
  height: number;
}

function overlaps(a: Box, b: Box): boolean {
  return a.x < b.x + b.width && b.x < a.x + a.width && a.y < b.y + b.height && b.y < a.y + a.height;
}

/** A destination box is the second half of a move, not a change in its own right. */
function isDestination(entry: GroundTruthEntry): boolean {
  return entry.kind === "moved_destination";
}

export function GroundTruth({
  truth,
  events,
}: {
  truth: GroundTruthEntry[];
  events: ChangeEvent[];
}) {
  const primary = truth.filter((t) => !isDestination(t));
  const matched = new Set<string>();
  const matchedTruth = new Set<number>();

  primary.forEach((entry, index) => {
    const box: Box = {
      x: entry.bbox[0],
      y: entry.bbox[1],
      width: entry.bbox[2],
      height: entry.bbox[3],
    };
    // The space differs: an insertion's true box is in rescan pixels and a
    // removal's is in baseline pixels, while detections are in canonical pixels.
    // The service converts before serving, so both are canonical here.
    const hit = events.find((e) => overlaps(box, e.bbox));
    if (hit) {
      matched.add(hit.event_id);
      matchedTruth.add(index);
    }
  });

  const found = matchedTruth.size;
  const spurious = events.filter((e) => !matched.has(e.event_id)).length;
  const missed = primary.length - found;
  const exact = found === primary.length && spurious === 0;

  return (
    <Panel>
      <PanelHead
        title="Against the answer key"
        hint="These frames were generated with known changes, so the result can be checked rather than taken on trust."
        right={
          <span
            className={cx(
              "font-mono text-2xs uppercase tracking-wider",
              exact ? "text-accent" : "text-kind-uncertain",
            )}
          >
            {exact ? "exact" : "imperfect"}
          </span>
        }
      />

      <div className="grid grid-cols-3 gap-4 border-t border-ground-edge px-4 py-3.5">
        <Stat
          label="found"
          value={`${found}/${primary.length}`}
          sub="true changes detected"
          tone={found === primary.length ? "accent" : "warn"}
        />
        <Stat
          label="missed"
          value={missed}
          sub={missed === 0 ? "nothing overlooked" : "changes not reported"}
          tone={missed === 0 ? "default" : "warn"}
        />
        <Stat
          label="extra"
          value={spurious}
          sub={spurious === 0 ? "nothing invented" : "reported but not real"}
          tone={spurious === 0 ? "default" : "warn"}
        />
      </div>

      <div className="border-t border-ground-edge px-4 py-3">
        <div className="label mb-2">The changes that were made</div>
        <ul className="space-y-1.5">
          {primary.map((entry, i) => {
            const style = kindStyle(entry.kind);
            const hit = matchedTruth.has(i);
            return (
              <li key={`${entry.object_id}-${i}`} className="flex items-center gap-2.5">
                <span
                  className={cx(
                    "flex h-5 w-5 shrink-0 items-center justify-center rounded border font-mono text-xs",
                    hit
                      ? cx(style.border, style.bg, style.text)
                      : "border-kind-missing/40 bg-kind-missing/10 text-kind-missing",
                  )}
                  title={hit ? "Detected" : "Not detected"}
                >
                  {hit ? style.glyph : "×"}
                </span>
                <span className="text-xs text-ink-muted">
                  <span className={style.text}>{style.label}</span>{" "}
                  <span className="font-mono text-2xs text-ink-faint">{entry.object_id}</span>
                </span>
                <span className="ml-auto font-mono text-2xs text-ink-faint">
                  {hit ? "detected" : "missed"}
                </span>
              </li>
            );
          })}
        </ul>
      </div>
    </Panel>
  );
}
