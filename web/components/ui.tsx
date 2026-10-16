"use client";

/**
 * The primitives every panel is assembled from.
 *
 * Small on purpose: a shared component earns its place here only when getting it
 * wrong in one place would be a bug. The confidence meter is the clearest case -
 * it is the interface's single way of expressing "how sure is the system", and
 * having one implementation means the words and the bar can never disagree.
 */

import { motion } from "framer-motion";
import type { ReactNode } from "react";
import { confidenceWord, cx, pct } from "@/lib/design";

// --- Layout ---------------------------------------------------------------

export function Panel({
  children,
  className,
  as: Tag = "section",
}: {
  children: ReactNode;
  className?: string;
  as?: "section" | "div" | "aside";
}) {
  return <Tag className={cx("panel", className)}>{children}</Tag>;
}

export function PanelHead({
  title,
  hint,
  right,
  icon,
}: {
  title: string;
  hint?: string;
  right?: ReactNode;
  icon?: ReactNode;
}) {
  return (
    <header className="flex items-start justify-between gap-4 px-4 py-3">
      <div className="min-w-0">
        <h2 className="flex items-center gap-2 text-sm font-semibold tracking-tight text-ink">
          {icon}
          {title}
        </h2>
        {hint ? <p className="mt-0.5 text-xs leading-relaxed text-ink-muted">{hint}</p> : null}
      </div>
      {right ? <div className="shrink-0">{right}</div> : null}
    </header>
  );
}

// --- Controls -------------------------------------------------------------

type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "default" | "primary" | "danger";
  loading?: boolean;
};

export function Button({
  variant = "default",
  loading = false,
  className,
  children,
  disabled,
  ...rest
}: ButtonProps) {
  const variantClass =
    variant === "primary" ? "btn-primary" : variant === "danger" ? "btn-danger" : "";
  return (
    <button
      {...rest}
      disabled={disabled || loading}
      className={cx("btn", variantClass, className)}
      aria-busy={loading || undefined}
    >
      {loading ? <Spinner /> : null}
      {children}
    </button>
  );
}

export function Spinner({ className }: { className?: string }) {
  return (
    <svg
      className={cx("h-3.5 w-3.5 animate-spin", className)}
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
    >
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeOpacity="0.25" strokeWidth="3" />
      <path
        d="M21 12a9 9 0 0 0-9-9"
        stroke="currentColor"
        strokeWidth="3"
        strokeLinecap="round"
      />
    </svg>
  );
}

export function Chip({
  children,
  className,
  title,
}: {
  children: ReactNode;
  className?: string;
  title?: string;
}) {
  return (
    <span className={cx("chip", className)} title={title}>
      {children}
    </span>
  );
}

// --- Data display ---------------------------------------------------------

/**
 * The one way this interface expresses confidence.
 *
 * The bar and the word come from the same number in the same render, so a
 * screen reader hears "high" exactly when the bar is long. A bare percentage
 * invites a precision the system does not have.
 */
export function ConfidenceMeter({
  value,
  showWord = true,
  className,
}: {
  value: number;
  showWord?: boolean;
  className?: string;
}) {
  const clamped = Math.max(0, Math.min(1, value));
  const tone =
    clamped >= 0.8
      ? "bg-accent"
      : clamped >= 0.6
        ? "bg-kind-state"
        : clamped >= 0.4
          ? "bg-kind-uncertain"
          : "bg-kind-missing";

  return (
    <div className={cx("flex items-center gap-2", className)}>
      <div
        className="h-1 w-14 shrink-0 overflow-hidden rounded-full bg-white/10"
        role="meter"
        aria-valuenow={Math.round(clamped * 100)}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={`Confidence ${pct(clamped)}`}
      >
        <motion.div
          className={cx("h-full rounded-full", tone)}
          initial={{ width: 0 }}
          animate={{ width: `${clamped * 100}%` }}
          transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
        />
      </div>
      <span className="tabular font-mono text-2xs text-ink-muted">
        {pct(clamped)}
        {showWord ? <span className="text-ink-faint"> · {confidenceWord(clamped)}</span> : null}
      </span>
    </div>
  );
}

export function Stat({
  label,
  value,
  sub,
  tone = "default",
}: {
  label: string;
  value: ReactNode;
  sub?: ReactNode;
  tone?: "default" | "accent" | "warn";
}) {
  return (
    <div className="min-w-0">
      <div className="label">{label}</div>
      <div
        className={cx(
          "tabular mt-1 truncate font-mono text-lg leading-none",
          tone === "accent" ? "text-accent" : tone === "warn" ? "text-kind-uncertain" : "text-ink",
        )}
      >
        {value}
      </div>
      {sub ? <div className="mt-1 text-2xs text-ink-faint">{sub}</div> : null}
    </div>
  );
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={cx("skeleton", className)} aria-hidden="true" />;
}

export function EmptyState({
  title,
  body,
  action,
  icon,
}: {
  title: string;
  body: ReactNode;
  action?: ReactNode;
  icon?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center px-6 py-14 text-center">
      {icon ? (
        <div className="mb-4 flex h-11 w-11 items-center justify-center rounded-full border border-ground-edge bg-ground-raised text-ink-faint">
          {icon}
        </div>
      ) : null}
      <h3 className="text-sm font-semibold text-ink">{title}</h3>
      <p className="mt-1.5 max-w-md text-xs leading-relaxed text-ink-muted">{body}</p>
      {action ? <div className="mt-5">{action}</div> : null}
    </div>
  );
}

export function Callout({
  tone,
  title,
  children,
}: {
  tone: "info" | "warn" | "bad" | "good";
  title: string;
  children?: ReactNode;
}) {
  const tones = {
    info: "border-ground-edge bg-white/[0.02] text-ink-muted",
    warn: "border-kind-uncertain/35 bg-kind-uncertain/[0.07] text-kind-uncertain",
    bad: "border-kind-missing/35 bg-kind-missing/[0.07] text-kind-missing",
    good: "border-accent/35 bg-accent/[0.07] text-accent",
  } as const;
  const glyphs = { info: "i", warn: "!", bad: "×", good: "✓" } as const;

  return (
    <div className={cx("rounded-md border px-3.5 py-3", tones[tone])} role="status">
      <div className="flex items-start gap-2.5">
        <span
          className="mt-px flex h-4 w-4 shrink-0 items-center justify-center rounded-full
            border border-current font-mono text-[10px] leading-none"
          aria-hidden="true"
        >
          {glyphs[tone]}
        </span>
        <div className="min-w-0">
          <div className="text-xs font-semibold">{title}</div>
          {children ? <div className="mt-1 text-xs leading-relaxed opacity-90">{children}</div> : null}
        </div>
      </div>
    </div>
  );
}
