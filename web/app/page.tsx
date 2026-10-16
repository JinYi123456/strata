"use client";

/**
 * The workspace: one site, one baseline, and whatever the latest rescan found.
 *
 * The layout answers the three questions in the order they are asked.
 *
 * 1. *What changed?* The frame and the change set sit side by side, because the
 *    list and the picture are two views of the same answer and reading either
 *    alone is worse than reading both.
 * 2. *Should I trust it?* Alignment, coverage, and the reasoning sit below,
 *    together, because they are the same kind of information - the basis for the
 *    result rather than the result.
 * 3. *What has been recorded?* The history sits last, since it is the record
 *    rather than the working surface.
 */

import { AnimatePresence, motion } from "framer-motion";
import { useCallback, useEffect, useRef, useState } from "react";
import { AgentTrace } from "@/components/AgentTrace";
import { ApprovalGate } from "@/components/ApprovalGate";
import { ChangeLedger } from "@/components/ChangeLedger";
import { GroundTruth } from "@/components/GroundTruth";
import { ImageViewer } from "@/components/ImageViewer";
import { LedgerHistory } from "@/components/LedgerHistory";
import { CoveragePanel, RegistrationPanel } from "@/components/ScanDetails";
import { SitePicker } from "@/components/SitePicker";
import { Button, Callout, Panel, PanelHead, Skeleton } from "@/components/ui";
import { ApiError, api, readAsDataUrl } from "@/lib/api";
import { cx, outcomeStyle } from "@/lib/design";
import type { LedgerEntry, ScanResult, SessionState, SiteSummary } from "@/lib/types";

export default function Page() {
  const [sites, setSites] = useState<SiteSummary[]>([]);
  const [sitesLoading, setSitesLoading] = useState(true);
  const [fatal, setFatal] = useState<string | null>(null);

  const [session, setSession] = useState<SessionState | null>(null);
  const [openingSite, setOpeningSite] = useState<string | null>(null);

  const [scan, setScan] = useState<ScanResult | null>(null);
  const [scanning, setScanning] = useState(false);
  const [scanError, setScanError] = useState<string | null>(null);

  const [selected, setSelected] = useState<string | null>(null);
  const [minConfidence, setMinConfidence] = useState(0);

  const [entry, setEntry] = useState<LedgerEntry | null>(null);
  const [committing, setCommitting] = useState(false);
  const [commitError, setCommitError] = useState<string | null>(null);

  const [history, setHistory] = useState<LedgerEntry[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);

  // --- Loading -----------------------------------------------------------

  useEffect(() => {
    let alive = true;
    api
      .sites()
      .then((s) => alive && setSites(s))
      .catch((e: unknown) => alive && setFatal(messageOf(e)))
      .finally(() => alive && setSitesLoading(false));
    return () => {
      alive = false;
    };
  }, []);

  const loadHistory = useCallback(async (siteId: string) => {
    setHistoryLoading(true);
    try {
      const body = await api.ledger(siteId);
      setHistory(body.entries);
    } catch {
      // History is supplementary; failing to load it must not break the scan.
      setHistory([]);
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  // --- Actions -----------------------------------------------------------

  const openSite = useCallback(
    async (site: SiteSummary) => {
      setOpeningSite(site.site_id);
      setFatal(null);
      try {
        const created = await api.createSession({ site_id: site.site_id });
        setSession(created);
        setScan(null);
        setSelected(null);
        setEntry(null);
        setCommitError(null);
        setScanError(null);
        window.history.replaceState(null, "", `?session=${created.session_id}`);
        void loadHistory(created.site_id);
      } catch (e: unknown) {
        setFatal(messageOf(e));
      } finally {
        setOpeningSite(null);
      }
    },
    [loadHistory],
  );

  const createSite = useCallback(
    async (name: string, files: File[]) => {
      setOpeningSite("new");
      setFatal(null);
      try {
        const frames = await Promise.all(files.slice(0, 8).map(readAsDataUrl));
        const created = await api.createSession({ name, baseline_frames: frames });
        setSession(created);
        setScan(null);
        setSelected(null);
        setEntry(null);
        window.history.replaceState(null, "", `?session=${created.session_id}`);
        setSites((prev) => [
          ...prev,
          {
            site_id: created.site_id,
            name: created.name,
            description: "Uploaded captures.",
            is_demo: false,
            baseline_frames: created.baseline_frames,
            thumbnail: null,
          },
        ]);
      } catch (e: unknown) {
        setFatal(messageOf(e));
      } finally {
        setOpeningSite(null);
      }
    },
    [],
  );

  const runScan = useCallback(async () => {
    if (!session) return;
    setScanning(true);
    setScanError(null);
    setSelected(null);
    setEntry(null);
    setCommitError(null);
    try {
      const result = await api.rescan(session.session_id, { min_confidence: 0 });
      setScan(result);
      setSession((s) => (s ? { ...s, scans: result.scan_index + 1 } : s));
    } catch (e: unknown) {
      setScanError(messageOf(e));
    } finally {
      setScanning(false);
    }
  }, [session]);

  const decide = useCallback(
    async (approved: boolean, note: string) => {
      if (!session) return;
      setCommitting(true);
      setCommitError(null);
      try {
        const recorded = await api.commit(session.session_id, { approved, note });
        setEntry(recorded);
        setSession((s) => (s ? { ...s, committed: approved } : s));
        void loadHistory(session.site_id);
      } catch (e: unknown) {
        setCommitError(messageOf(e));
      } finally {
        setCommitting(false);
      }
    },
    [session, loadHistory],
  );

  const closeSession = useCallback(() => {
    setSession(null);
    setScan(null);
    setSelected(null);
    setEntry(null);
    setHistory([]);
    window.history.replaceState(null, "", window.location.pathname);
  }, []);

  // Escape clears the selection, which is the fastest way out of a focused view.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setSelected(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // --- Render ------------------------------------------------------------

  if (!session) {
    return (
      <main id="main">
        <TopBar onHome={null} />
        {fatal ? (
          <div className="mx-auto max-w-3xl px-6 pt-8">
            <Callout tone="bad" title="Could not load sites">
              {fatal}
            </Callout>
          </div>
        ) : null}
        <SitePicker
          sites={sites}
          loading={sitesLoading}
          onOpen={openSite}
          onCreate={createSite}
          busySite={openingSite}
        />
      </main>
    );
  }

  const events = (scan?.events ?? []).filter((e) => e.confidence >= minConfidence);
  const totals = scan?.event_totals ?? {};
  const found = Object.values(totals).reduce((a, b) => a + b, 0);
  const outcome = scan?.outcome ?? "";
  const style = outcomeStyle(outcome || "running");

  return (
    <main id="main" className="min-h-screen">
      <TopBar
        onHome={closeSession}
        sessionName={session.name}
        sessionId={session.session_id}
        right={
          <Button variant="primary" onClick={runScan} loading={scanning}>
            {scan ? "Scan again" : "Run rescan"}
          </Button>
        }
      />

      <div className="mx-auto max-w-[1600px] px-6 pb-20 pt-6">
        <SessionSummary session={session} scan={scan} found={found} outcome={outcome} />

        {scanError ? (
          <div className="mt-5">
            <Callout tone="bad" title="The scan did not complete">
              {scanError}
            </Callout>
          </div>
        ) : null}

        {!scan && !scanning ? (
          <FirstScanPrompt session={session} onScan={runScan} />
        ) : null}

        <div className="mt-5 grid gap-5 lg:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)]">
          {/* Left: the frame, then the decision about it. */}
          <div className="flex flex-col gap-5">
            <Panel className="p-4">
              <ImageViewer
                // Before the first scan there are no panels from a scan, so the
                // recorded baseline stands in. Showing the reference view is far
                // more useful than showing nothing.
                images={scan?.images ?? (session.baseline ? { baseline: session.baseline } : {})}
                events={scan?.events ?? []}
                selectedId={selected}
                onSelect={setSelected}
                busy={scanning}
                busyLabel={`Scanning ${session.name}`}
              />
            </Panel>

            {scan ? (
              <ApprovalGate
                outcome={outcome}
                eventCount={found}
                committed={session.committed && entry !== null}
                entry={entry}
                onDecide={decide}
                busy={committing}
                error={commitError}
              />
            ) : null}
          </div>

          {/* Right: what was found. */}
          <div className="flex flex-col gap-5">
            {scanning && !scan ? (
              <Panel className="p-4">
                <PanelHead title="Change set" hint="Resolving events…" />
                <div className="mt-3 space-y-2">
                  <Skeleton className="h-16 w-full" />
                  <Skeleton className="h-16 w-full" />
                  <Skeleton className="h-16 w-full" />
                </div>
              </Panel>
            ) : scan ? (
              <div className="lg:sticky lg:top-[4.75rem] lg:max-h-[calc(100vh-6rem)]">
                <ChangeLedger
                  events={events}
                  totals={totals}
                  selectedId={selected}
                  onSelect={setSelected}
                  minConfidence={minConfidence}
                  onMinConfidenceChange={setMinConfidence}
                />
              </div>
            ) : (
              <Panel>
                <PanelHead title="Change set" />
                <div className="border-t border-ground-edge px-4 py-8 text-center text-xs text-ink-faint">
                  Nothing scanned yet.
                </div>
              </Panel>
            )}
          </div>
        </div>

        {/* The basis for the result. */}
        <AnimatePresence>
          {scan ? (
            <motion.section
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.35, delay: 0.05 }}
              className="mt-5 grid gap-5 lg:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)_minmax(0,1fr)]"
            >
              <AgentTrace trace={scan.trace} />
              <CoveragePanel coverage={scan.coverage} guidance={scan.guidance} />
              <div className="flex flex-col gap-5">
                <RegistrationPanel registration={scan.registration} />
                {scan.ground_truth && scan.ground_truth.length > 0 ? (
                  <GroundTruth truth={scan.ground_truth} events={scan.events} />
                ) : null}
              </div>
            </motion.section>
          ) : null}
        </AnimatePresence>

        <div className="mt-5">
          <LedgerHistory
            entries={history}
            loading={historyLoading}
            onRefresh={() => void loadHistory(session.site_id)}
          />
        </div>

        <footer className="mt-10 flex flex-wrap items-center justify-between gap-3 border-t border-ground-edge pt-5 font-mono text-2xs text-ink-faint">
          <span>
            {session.canonical_size[0]} × {session.canonical_size[1]} canonical ·{" "}
            {scan?.feature_backend ?? "—"}
          </span>
          <span className={cx(style.tone === "waiting" && "text-kind-uncertain")}>
            {style.label}
          </span>
        </footer>
      </div>
    </main>
  );
}

// --- Chrome ---------------------------------------------------------------

function TopBar({
  onHome,
  sessionName,
  sessionId,
  right,
}: {
  onHome: (() => void) | null;
  sessionName?: string;
  sessionId?: string;
  right?: React.ReactNode;
}) {
  return (
    <header className="sticky top-0 z-30 border-b border-ground-edge bg-ground/85 backdrop-blur-md">
      <div className="mx-auto flex h-[3.75rem] max-w-[1600px] items-center gap-4 px-6">
        <button
          type="button"
          onClick={onHome ?? undefined}
          disabled={!onHome}
          className={cx(
            "flex items-center gap-2.5",
            onHome && "transition-opacity hover:opacity-80",
          )}
        >
          <Mark />
          <span className="text-sm font-semibold tracking-[0.18em] text-ink">STRATA</span>
        </button>

        {sessionName ? (
          <>
            <span className="text-ink-faint">/</span>
            <div className="min-w-0">
              <div className="truncate text-sm text-ink">{sessionName}</div>
            </div>
            {sessionId ? (
              <span className="hidden font-mono text-2xs text-ink-faint lg:block">{sessionId}</span>
            ) : null}
          </>
        ) : (
          <span className="hidden text-xs text-ink-faint sm:block">
            version control for the physical world
          </span>
        )}

        <div className="ml-auto flex items-center gap-2">{right}</div>
      </div>
    </header>
  );
}

function Mark() {
  return (
    <svg viewBox="0 0 24 24" className="h-5 w-5 text-accent" fill="none" aria-hidden="true">
      {/* Three strata, the shallowest offset: a section through a site. */}
      <path
        d="M3 8.5 12 4l9 4.5-9 4.5z"
        stroke="currentColor"
        strokeWidth="1.3"
        strokeLinejoin="round"
      />
      <path d="M3 13.5 12 18l9-4.5" stroke="currentColor" strokeWidth="1.3" strokeLinejoin="round" />
      <path
        d="M3 17.5 12 22l9-4.5"
        stroke="currentColor"
        strokeWidth="1.3"
        strokeLinejoin="round"
        strokeOpacity="0.5"
      />
    </svg>
  );
}

function SessionSummary({
  session,
  scan,
  found,
  outcome,
}: {
  session: SessionState;
  scan: ScanResult | null;
  found: number;
  outcome: string;
}) {
  const style = outcomeStyle(outcome || "running");
  return (
    <div className="flex flex-wrap items-end justify-between gap-4">
      <div className="min-w-0">
        <h1 className="text-xl font-semibold tracking-tight text-ink">{session.name}</h1>
        <p className="mt-1 font-mono text-2xs text-ink-faint">
          opened {new Date(session.created_at).toLocaleDateString()} · {session.baseline_frames}{" "}
          baseline frame{session.baseline_frames === 1 ? "" : "s"} · {session.scans} scan
          {session.scans === 1 ? "" : "s"}
        </p>
      </div>
      {scan ? (
        <div className="flex flex-wrap items-center gap-2">
          <span className="chip border-ground-edge text-ink-muted">{found} changes</span>
          <span
            className={cx(
              "chip",
              style.tone === "bad"
                ? "border-kind-missing/40 text-kind-missing"
                : style.tone === "waiting"
                  ? "border-kind-uncertain/40 text-kind-uncertain"
                  : style.tone === "good"
                    ? "border-accent/40 text-accent"
                    : "border-ground-edge text-ink-muted",
            )}
          >
            {style.label}
          </span>
        </div>
      ) : null}
    </div>
  );
}

function FirstScanPrompt({ session, onScan }: { session: SessionState; onScan: () => void }) {
  return (
    <Panel className="mt-5">
      <div className="flex flex-wrap items-center justify-between gap-4 px-5 py-4">
        <div className="min-w-0">
          <h2 className="text-sm font-semibold text-ink">Baseline recorded</h2>
          <p className="mt-1 max-w-xl text-xs leading-relaxed text-ink-muted">
            {session.name} has been recorded as the reference. Run a rescan to register a second
            visit against it and see what changed. The first scan also measures how much of the
            site these captures actually cover.
          </p>
        </div>
        <Button variant="primary" onClick={onScan}>
          Run rescan
        </Button>
      </div>
    </Panel>
  );
}

// --- Helpers --------------------------------------------------------------

function messageOf(e: unknown): string {
  if (e instanceof ApiError) return e.hint ? `${e.message} ${e.hint}` : e.message;
  if (e instanceof Error) return e.message;
  return String(e);
}
