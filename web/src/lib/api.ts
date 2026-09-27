/**
 * Data access for the IRVision frontend.
 *
 * Two modes:
 *  - live: NEXT_PUBLIC_API_URL points to the FastAPI backend (irvision/api/server.py)
 *  - demo: no backend; real results pre-rendered by scripts/export_web_demo.py into /public/demo
 * If the live backend is unreachable, calls fall back to demo data.
 */

export type StageStatus = "done" | "off" | "skipped";

export interface Stage {
  key: string;
  label: string;
  description: string;
  status: StageStatus;
  ms: number;
}

export interface Detection {
  class_name: string;
  confidence: number;
  box: [number, number, number, number];
}

export interface Payload {
  source: { kind?: string; id?: string; title?: string };
  shape: [number, number];
  output_shape: [number, number];
  scale: number;
  colorizer: string;
  stages: Stage[];
  images: Partial<Record<
    "input" | "enhanced" | "colorized" | "detections" | "super_resolved" | "reference" | "semantic" | "reference_semantic",
    string
  >>;
  metrics: {
    total_ms: number;
    psnr?: number;
    ssim?: number;
    semantic?: { agreement: number; miou: number; mean_dice: number; iou: Record<string, number | null> };
    detection?: { tp: number; n_pred: number; n_ref: number; precision: number | null; recall: number | null };
  };
  detections: Detection[] | null;
  reference_detections: Detection[] | null;
  legend: { name: string; color: string }[];
  warnings: string[];
}

export interface Example {
  id: string;
  title: string;
  description: string;
  thumbnail: string;
  ir_thumbnail: string;
}

export interface Report {
  colorization?: {
    patches: number;
    rows: { method: string; all: [number, number]; holdout: [number, number] | null }[];
    end_to_end: { psnr: number; ssim: number; time_ms: Record<string, number> }[];
  };
  semantic?: {
    ceiling: number;
    rows: { method: string; miou: number; agreement: number }[];
    per_class: Record<string, number | null>;
  };
  advanced?: {
    super_resolution: { method: string; psnr: number; ssim: number; semantic_miou: number }[];
    detection: { true_rgb: Record<string, number>; unet_rgb: Record<string, number>; matched: number };
  };
}

export interface Options {
  superResolution: boolean;
  detection: boolean;
}

const BUILD_API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "").replace(/\/$/, "");
const STORAGE_KEY = "irvision_api";

/**
 * Model server URL, in order of priority:
 *  1. `?api=https://...` in the page URL (remembered in this browser; `?api=off` forgets it)
 *  2. a URL remembered from an earlier visit
 *  3. NEXT_PUBLIC_API_URL at build time
 * This lets the site use a server whose address changes (e.g. a Cloudflare quick tunnel)
 * without redeploying.
 */
export function apiUrl(): string {
  if (typeof window === "undefined") return BUILD_API_URL;
  try {
    const param = new URLSearchParams(window.location.search).get("api");
    if (param !== null) {
      if (param === "" || param === "off") window.localStorage.removeItem(STORAGE_KEY);
      else window.localStorage.setItem(STORAGE_KEY, param.replace(/\/$/, ""));
    }
    return window.localStorage.getItem(STORAGE_KEY) ?? BUILD_API_URL;
  } catch {
    return BUILD_API_URL;
  }
}

export type Mode = "live" | "demo";

export function configuredMode(): Mode {
  return apiUrl() ? "live" : "demo";
}

async function getJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* keep status text */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export async function checkBackend(): Promise<boolean> {
  if (!apiUrl()) return false;
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 8000);
    const res = await fetch(`${apiUrl()}/api/health`, { signal: controller.signal });
    clearTimeout(timer);
    return res.ok;
  } catch {
    return false;
  }
}

const variant = (o: Options) => `sr${o.superResolution ? 1 : 0}_det${o.detection ? 1 : 0}`;

export async function fetchExamples(mode: Mode): Promise<Example[]> {
  if (mode === "live") {
    try {
      return await getJson<Example[]>(`${apiUrl()}/api/examples`);
    } catch {
      /* fall back to demo data */
    }
  }
  return getJson<Example[]>("/demo/examples.json");
}

export async function runExample(mode: Mode, id: string, options: Options): Promise<Payload> {
  if (mode === "live") {
    const qs = `super_resolution=${options.superResolution}&detection=${options.detection}`;
    return getJson<Payload>(`${apiUrl()}/api/process/example/${encodeURIComponent(id)}?${qs}`, { method: "POST" });
  }
  return getJson<Payload>(`/demo/${id}/${variant(options)}/payload.json`);
}

export async function runUpload(image: File, reference: File | null, options: Options): Promise<Payload> {
  if (!apiUrl()) throw new Error("Uploading your own image needs the live model server.");
  const form = new FormData();
  form.append("image", image);
  if (reference) form.append("reference", reference);
  form.append("super_resolution", String(options.superResolution));
  form.append("detection", String(options.detection));
  return getJson<Payload>(`${apiUrl()}/api/process`, { method: "POST", body: form });
}

export async function fetchReport(mode: Mode): Promise<Report> {
  if (mode === "live") {
    try {
      return await getJson<Report>(`${apiUrl()}/api/report`);
    } catch {
      /* fall back to demo data */
    }
  }
  return getJson<Report>("/demo/report.json");
}
