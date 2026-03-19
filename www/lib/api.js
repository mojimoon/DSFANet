export const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";

export async function fetchApi(path) {
  let res;
  try {
    res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
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
