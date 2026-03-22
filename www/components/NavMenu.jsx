"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { Beaker, Gauge, Layers, TrendingUpDown, Radar, RefreshCcw, ShieldAlert, Telescope } from "lucide-react";
import { fetchApi } from "@/lib/api";
import { buildHrefWithDataset, getDatasetFromQuery, getStoredDataset, setStoredDataset } from "@/lib/dataset";
import SidebarRetrainControl from "@/components/SidebarRetrainControl";

const links = [
  ["/", "Overview", Gauge],
  ["/dataset", "Dataset", Layers],
  ["/benchmarks", "Benchmarks", Telescope],
  ["/attacks", "Attacks", TrendingUpDown],
  ["/retrain-strategy", "Retrain Strategy", RefreshCcw],
  ["/models", "Models", Radar],
  ["/alerts", "Alerts", ShieldAlert],
  ["/experiments", "Experiments", Beaker],
];

export default function NavMenu() {
  const pathname = usePathname();
  const router = useRouter();
  const [dataset, setDataset] = useState("");
  const [datasetOptions, setDatasetOptions] = useState([]);

  useEffect(() => {
    const qDataset = getDatasetFromQuery();
    const stored = getStoredDataset();
    const chosen = qDataset || stored || "";
    if (chosen) {
      setDataset(chosen);
      setStoredDataset(chosen);
    }
  }, []);

  const onDatasetChange = (nextValue) => {
    const next = String(nextValue || "");
    // console.log("Switching dataset to", next);
    const currentQueryDataset = getDatasetFromQuery();
    if (next === dataset && next === currentQueryDataset) {
      return;
    }

    setDataset(next);
    setStoredDataset(next);
    const params = new URLSearchParams(typeof window !== "undefined" ? window.location.search : "");
    if (next) {
      params.set("dataset", next);
    } else {
      params.delete("dataset");
    }
    const query = params.toString();
    const target = query ? `${pathname}?${query}` : pathname;
    router.replace(target, { scroll: false });
    router.refresh();
  };

  useEffect(() => {
    fetchApi("/api/datasets")
      .then((rows) => {
        const list = Array.isArray(rows) ? rows : [];
        setDatasetOptions(list);
        if (!list.length) {
          return;
        }

        const fallback = String(list[0].dataset || "");
        const currentDataset = getDatasetFromQuery() || getStoredDataset() || dataset || "";
        const hasCurrent = list.some((row) => String(row.dataset || "") === String(currentDataset));
        if (!hasCurrent && fallback) {
          onDatasetChange(fallback);
        }
      })
      .catch(() => setDatasetOptions([]));
  }, []);

  const selectorHint = useMemo(() => {
    const current = datasetOptions.find((x) => String(x.dataset) === String(dataset));
    if (!current) {
      return "Latest run: -";
    }
    return `Latest run: ${current.run_id || "-"}`;
  }, [datasetOptions, dataset]);

  return (
    <>
      <div className="datasetSwitchCard">
        <div className="datasetSwitchTitle">Dataset</div>
        <select className="datasetSelect" value={dataset} onChange={(e) => onDatasetChange(e.target.value)}>
          {datasetOptions.map((row) => (
            <option key={String(row.dataset)} value={String(row.dataset)}>
              {String(row.dataset)}
            </option>
          ))}
        </select>
        <div className="datasetSwitchHint">{selectorHint}</div>
      </div>

      <nav className="menu">
        {links.map(([href, label, Icon]) => {
          const active = pathname === href || (href !== "/" && pathname.startsWith(href));
          return (
            <Link key={href} href={buildHrefWithDataset(href, dataset)} className={active ? "active" : ""}>
              <Icon size={15} />
              {label}
            </Link>
          );
        })}
      </nav>

      <SidebarRetrainControl />
    </>
  );
}
