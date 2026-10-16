/**
 * The API client.
 *
 * One rule: a failed request always surfaces as an `ApiError` carrying the
 * server's own message. The service returns structured failures for every case
 * it knows about, and a client that shows "Something went wrong" instead of
 * "the image is 40 MB, the limit is 12 MB" throws away the work that went into
 * producing a useful message.
 */

import type {
  ApiErrorBody,
  Health,
  Ledger,
  ScanResult,
  SessionState,
  SiteSummary,
} from "./types";

/**
 * Where the API lives.
 *
 * In production the service serves this bundle itself, so the API is on the same
 * origin and the base is empty - which also means the app can be published as a
 * single URL with no CORS involved. In development the two run as separate
 * processes, so the base points at the local service instead.
 *
 * `NEXT_PUBLIC_API_BASE` overrides both, for the case where the front end and
 * the service are deployed separately.
 */
const API_ORIGIN =
  process.env.NEXT_PUBLIC_API_BASE ??
  (process.env.NODE_ENV === "development" ? "http://127.0.0.1:8000" : "");

const BASE = `${API_ORIGIN}/api`;

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly hint: string;

  constructor(status: number, body: ApiErrorBody | null, fallback: string) {
    const detail = body?.detail || fallback;
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.code = body?.error ?? "unknown";
    this.hint = body?.hint ?? "";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    });
  } catch {
    // A network failure is not the same as a rejection, and saying so is the
    // difference between "the service is down" and "your input was wrong".
    throw new ApiError(0, null, "Could not reach the service. Is it running?");
  }

  if (!res.ok) {
    let body: ApiErrorBody | null = null;
    try {
      body = (await res.json()) as ApiErrorBody;
    } catch {
      // Non-JSON error body; fall through to the status text.
    }
    throw new ApiError(res.status, body, `Request failed with status ${res.status}`);
  }

  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export const api = {
  health: () => request<Health>("/health"),

  sites: async (): Promise<SiteSummary[]> => {
    const body = await request<{ sites: SiteSummary[] }>("/sites");
    return body.sites;
  },

  createSession: (input: { site_id?: string; name?: string; baseline_frames?: string[] }) =>
    request<SessionState>("/sessions", {
      method: "POST",
      body: JSON.stringify(input),
    }),

  session: (id: string) => request<SessionState>(`/sessions/${id}`),

  rescan: (
    id: string,
    input: { frames?: string[]; use_site_rescan?: boolean; min_confidence?: number } = {},
  ) =>
    request<ScanResult>(`/sessions/${id}/rescan`, {
      method: "POST",
      body: JSON.stringify(input),
    }),

  commit: (id: string, input: { approved: boolean; note?: string }) =>
    request<import("./types").LedgerEntry>(`/sessions/${id}/commit`, {
      method: "POST",
      body: JSON.stringify(input),
    }),

  ledger: (siteId: string) => request<Ledger>(`/sites/${encodeURIComponent(siteId)}/ledger`),

  scanRecord: (id: string, index: number) =>
    request<Record<string, unknown>>(`/sessions/${id}/scans/${index}`),
};

/** Absolute URL for an API-relative path, safe on any origin. */
export function absolute(url: string): string {
  if (typeof window === "undefined") return url;
  return new URL(url, window.location.origin).toString();
}

/** Read a file as a base64 data URL, for the upload path. */
export function readAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(new Error(`Could not read ${file.name}`));
    reader.readAsDataURL(file);
  });
}
