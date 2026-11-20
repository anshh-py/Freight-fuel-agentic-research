import type {
  ApiResponse,
  FeatureCatalog,
  HealthResponse,
  ReplayCatalog,
  ReplayResponse,
  SessionState,
} from "./types";

// The Vite dev server proxies /api and /health to the FastAPI backend. In
// production, set VITE_API_BASE to the full backend origin.
const API_BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");

export class ApiError extends Error {
  status: number;
  code?: string;
  constructor(status: number, message: string, code?: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });

  if (res.status === 204) {
    return undefined as T;
  }

  let body: unknown = null;
  try {
    body = await res.json();
  } catch {
    body = null;
  }

  if (!res.ok) {
    const detail =
      body && typeof body === "object" && "detail" in body
        ? String((body as { detail: unknown }).detail)
        : `HTTP ${res.status}`;
    throw new ApiError(res.status, detail);
  }

  return body as T;
}

export const api = {
  health: () => request<HealthResponse>("/health"),

  features: () => request<FeatureCatalog>("/api/v1/features"),

  createSession: (featureSet: string) =>
    request<{ session_id: string; feature_set: string }>("/api/v1/sessions", {
      method: "POST",
      body: JSON.stringify({ feature_set: featureSet }),
    }),

  getSession: (sessionId: string) =>
    request<SessionState>(`/api/v1/sessions/${sessionId}`),

  deleteSession: (sessionId: string) =>
    request<void>(`/api/v1/sessions/${sessionId}`, { method: "DELETE" }),

  sendMessage: (
    sessionId: string,
    query: string,
    structuredInputs: Record<string, string | number> | null = null,
  ) =>
    request<ApiResponse>(`/api/v1/sessions/${sessionId}/messages`, {
      method: "POST",
      body: JSON.stringify({
        query,
        structured_inputs: structuredInputs ?? null,
      }),
    }),

  replayCatalog: () =>
    request<ReplayCatalog>("/api/v1/replay/catalog"),

  replay: (tripId = "trip_000001", sample = 200) =>
    request<ReplayResponse>(
      `/api/v1/replay?trip_id=${encodeURIComponent(tripId)}&sample=${sample}`,
    ),
};
