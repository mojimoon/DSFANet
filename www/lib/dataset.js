"use client";

const STORAGE_KEY = "ids:selectedDataset";

function normalizeDatasetValue(value) {
  const next = String(value || "").trim();
  const low = next.toLowerCase();
  if (!next || low === "all" || low === "undefined" || low === "null" || low === "none" || low === "nan") {
    return "";
  }
  return next;
}

export function getStoredDataset() {
  if (typeof window === "undefined") {
    return "";
  }
  return normalizeDatasetValue(window.localStorage.getItem(STORAGE_KEY) || "");
}

export function setStoredDataset(value) {
  if (typeof window === "undefined") {
    return;
  }
  const next = normalizeDatasetValue(value);
  if (!next) {
    window.localStorage.removeItem(STORAGE_KEY);
    return;
  }
  window.localStorage.setItem(STORAGE_KEY, next);
}

export function buildHrefWithDataset(path, dataset) {
  const normalized = normalizeDatasetValue(dataset);
  if (!normalized) {
    return path;
  }
  const sep = path.includes("?") ? "&" : "?";
  return `${path}${sep}dataset=${encodeURIComponent(normalized)}`;
}

export function getDatasetFromQuery() {
  if (typeof window === "undefined") {
    return "";
  }
  const params = new URLSearchParams(window.location.search);
  return normalizeDatasetValue(params.get("dataset") || "");
}
