export const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";

function isInvalidDatasetToken(value) {
  const token = String(value || "").trim().toLowerCase();
  return token === "" || token === "all" || token === "undefined" || token === "null" || token === "none" || token === "nan";
}

function buildUrl(path, query) {
  const url = new URL(`${API_BASE}${path}`);
  const params = query && typeof query === "object" ? query : {};
  const skipDatasetInjection = path === "/api/datasets";

  if (typeof window !== "undefined") {
    const stored = window.localStorage.getItem("ids:selectedDataset") || "";
    if (!skipDatasetInjection && !isInvalidDatasetToken(stored) && params.dataset === undefined && !url.searchParams.get("dataset")) {
      url.searchParams.set("dataset", stored);
    }
  }

  Object.entries(params).forEach(([k, v]) => {
    if (v === undefined || v === null || v === "") {
      return;
    }
    if (k === "dataset" && isInvalidDatasetToken(v)) {
      return;
    }
    url.searchParams.set(k, String(v));
  });

  return url.toString();
}

export async function fetchApi(path, query) {
  let res;
  try {
    res = await fetch(buildUrl(path, query), { cache: "no-store" });
  } catch (err) {
    throw new Error(`Cannot connect to backend ${API_BASE}. Start Python API with: poetry run python web_main.py --serve-only`);
  }

  if (!res.ok) {
    throw new Error(`API request failed (${res.status}): ${path}`);
  }
  return await res.json();
}

export function num(v, digits = 4) {
  const parsed = Number(v);
  if (Number.isNaN(parsed)) {
    return String(v);
  }
  return parsed.toFixed(digits);
}
