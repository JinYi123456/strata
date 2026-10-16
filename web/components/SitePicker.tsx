"use client";

/**
 * Choosing a site to work with.
 *
 * Presented as a set of places rather than a table of records, because that is
 * what they are. Each card carries its own preview frame, so the choice is made
 * by looking at the site rather than by reading its identifier.
 *
 * The difficult site is labelled as difficult in the picker itself. A demo that
 * only offers its best cases is a demo that has been curated to mislead, and the
 * honest thing is to hand the weak spot over with a note attached.
 */

import { motion } from "framer-motion";
import { useState } from "react";
import { cx } from "@/lib/design";
import type { SiteSummary } from "@/lib/types";
import { Button, Skeleton } from "./ui";

export function SitePicker({
  sites,
  loading,
  onOpen,
  onCreate,
  busySite,
}: {
  sites: SiteSummary[];
  loading: boolean;
  onOpen: (site: SiteSummary) => void;
  onCreate: (name: string, files: File[]) => void;
  busySite: string | null;
}) {
  return (
    <div className="mx-auto w-full max-w-6xl px-6 py-10">
      <header className="mb-8">
        <h1 className="text-2xl font-semibold tracking-tight text-ink">Sites</h1>
        <p className="mt-1.5 max-w-2xl text-sm leading-relaxed text-ink-muted">
          Record a place once. Re-scan it later. Get a typed account of what changed,
          with the evidence and the reasoning shown alongside it.
        </p>
      </header>

      {loading ? (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-64 w-full" />
          ))}
        </div>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {sites.map((site, i) => (
            <SiteCard
              key={site.site_id}
              site={site}
              index={i}
              busy={busySite === site.site_id}
              onOpen={() => onOpen(site)}
            />
          ))}
          <NewSiteCard onCreate={onCreate} />
        </div>
      )}
    </div>
  );
}

function SiteCard({
  site,
  index,
  busy,
  onOpen,
}: {
  site: SiteSummary;
  index: number;
  busy: boolean;
  onOpen: () => void;
}) {
  const hard = /difficult|limitation|adjacent|clutter/i.test(site.description);

  return (
    <motion.button
      type="button"
      onClick={onOpen}
      disabled={busy}
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: Math.min(index * 0.05, 0.4), duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
      whileHover={{ y: -3 }}
      className={cx(
        "group relative flex flex-col overflow-hidden rounded-lg border border-ground-edge",
        "bg-ground-panel/70 text-left transition-colors hover:border-ink-faint/40",
        "disabled:cursor-wait",
      )}
    >
      <div className="relative aspect-[16/10] w-full overflow-hidden bg-ground">
        {site.thumbnail ? (
          <img
            src={site.thumbnail}
            alt={`Preview of ${site.name}`}
            className="h-full w-full object-cover transition-transform duration-500
              group-hover:scale-[1.03]"
          />
        ) : (
          <div className="flex h-full items-center justify-center text-ink-faint">
            <svg viewBox="0 0 24 24" className="h-6 w-6" fill="none" aria-hidden="true">
              <path
                d="M4 16l4.5-5 3.5 4 3-3 5 5M4 5h16v14H4z"
                stroke="currentColor"
                strokeWidth="1.4"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </div>
        )}

        {/* A scan line sweeps the preview while a session is opening, so the wait
            reads as work being done rather than as a stall. */}
        {busy ? (
          <div className="absolute inset-0 overflow-hidden bg-ground/60 backdrop-blur-[1px]">
            <div className="absolute inset-x-0 h-16 animate-scanline bg-gradient-to-b from-transparent via-accent/25 to-transparent" />
          </div>
        ) : null}

        <div className="absolute left-3 top-3 flex gap-1.5">
          {site.is_demo ? (
            <span className="rounded bg-black/65 px-1.5 py-0.5 font-mono text-2xs uppercase tracking-wider text-ink-muted backdrop-blur-sm">
              sample
            </span>
          ) : (
            <span className="rounded bg-black/65 px-1.5 py-0.5 font-mono text-2xs uppercase tracking-wider text-accent backdrop-blur-sm">
              yours
            </span>
          )}
          {hard ? (
            <span
              className="rounded bg-black/65 px-1.5 py-0.5 font-mono text-2xs uppercase
                tracking-wider text-kind-uncertain backdrop-blur-sm"
              title="A scene the system is known to find difficult"
            >
              hard case
            </span>
          ) : null}
        </div>
      </div>

      <div className="flex flex-1 flex-col p-4">
        <h2 className="text-sm font-semibold tracking-tight text-ink">{site.name}</h2>
        <p className="mt-1.5 flex-1 text-xs leading-relaxed text-ink-muted">{site.description}</p>
        <div className="mt-3 flex items-center justify-between">
          <span className="font-mono text-2xs text-ink-faint">
            {site.baseline_frames} baseline frame{site.baseline_frames === 1 ? "" : "s"}
          </span>
          <span className="flex items-center gap-1 font-mono text-2xs text-accent opacity-0 transition-opacity group-hover:opacity-100">
            open
            <svg viewBox="0 0 24 24" className="h-3 w-3" fill="none" aria-hidden="true">
              <path
                d="M5 12h14M13 6l6 6-6 6"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </span>
        </div>
      </div>
    </motion.button>
  );
}

/**
 * Uploading your own captures.
 *
 * Kept deliberately plain: the interesting part of this product is what happens
 * after the images are in, and dressing up a file input would suggest otherwise.
 */
function NewSiteCard({ onCreate }: { onCreate: (name: string, files: File[]) => void }) {
  const [name, setName] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [open, setOpen] = useState(false);

  if (!open) {
    return (
      <motion.button
        type="button"
        onClick={() => setOpen(true)}
        initial={{ opacity: 0, y: 14 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3, duration: 0.4 }}
        whileHover={{ y: -3 }}
        className="flex min-h-[16rem] flex-col items-center justify-center gap-2.5 rounded-lg
          border border-dashed border-ground-edge bg-transparent p-6 text-ink-faint
          transition-colors hover:border-accent/40 hover:text-accent"
      >
        <svg viewBox="0 0 24 24" className="h-6 w-6" fill="none" aria-hidden="true">
          <path
            d="M12 5v14M5 12h14"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinecap="round"
          />
        </svg>
        <span className="text-sm font-medium">Record your own site</span>
        <span className="max-w-[16rem] text-center text-2xs leading-relaxed">
          Upload a baseline capture. You can scan it against a second visit straight away.
        </span>
      </motion.button>
    );
  }

  return (
    <div className="flex min-h-[16rem] flex-col gap-3 rounded-lg border border-ground-edge bg-ground-panel/70 p-4">
      <h3 className="text-sm font-semibold text-ink">New site</h3>

      <label className="flex flex-col gap-1.5">
        <span className="label">Name</span>
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="North rack, bay 3"
          className="rounded-md border border-ground-edge bg-ground px-3 py-2 text-xs text-ink
            placeholder:text-ink-faint focus:border-accent/50"
        />
      </label>

      <label className="flex flex-col gap-1.5">
        <span className="label">Baseline frame</span>
        <input
          type="file"
          accept="image/png,image/jpeg"
          onChange={(e) => setFiles(Array.from(e.target.files ?? []))}
          className="w-full cursor-pointer rounded-md border border-ground-edge bg-ground text-xs
            text-ink-muted file:mr-3 file:cursor-pointer file:border-0 file:bg-ground-raised
            file:px-3 file:py-2 file:text-xs file:text-ink hover:file:bg-ground-hover"
        />
      </label>

      <div className="mt-auto flex gap-2">
        <Button onClick={() => setOpen(false)}>Cancel</Button>
        <Button
          variant="primary"
          disabled={files.length === 0}
          onClick={() => onCreate(name.trim() || "Untitled site", files)}
          className="flex-1"
        >
          Create
        </Button>
      </div>
    </div>
  );
}
