"use client";

import { useEffect, useState } from "react";
import { BarChart } from "@/components/charts";
import DataTableCard from "@/components/DataTableCard";
import { fetchApi, num } from "@/lib/api";
import { Database } from "lucide-react";

export default function DatasetPage() {
  const [data, setData] = useState(null);

  useEffect(() => {
    fetchApi("/api/dashboard").then(setData).catch(console.error);
  }, []);

  if (!data) {
    return <p>Loading dataset analysis...</p>;
  }

  const d = data.dataset_overview.class_distribution;
  const stats = [
    ...data.dataset_overview.feature_stats.static_top_variance.map((x) => ({ group: "static", ...x })),
    ...data.dataset_overview.feature_stats.temporal_top_variance.map((x) => ({ group: "temporal", ...x })),
  ];

  const columns = [
    { field: "group", headerName: "Group", width: 110 },
    { field: "feature", headerName: "Feature", flex: 1, minWidth: 220 },
    { field: "mean", headerName: "Mean", width: 120, valueFormatter: (v) => num(v, 4) },
    { field: "std", headerName: "Std", width: 120, valueFormatter: (v) => num(v, 4) },
    { field: "min", headerName: "Min", width: 120, valueFormatter: (v) => num(v, 4) },
    { field: "max", headerName: "Max", width: 120, valueFormatter: (v) => num(v, 4) },
  ];

  return (
    <>
      <h2 className="pageTitle titleRow">
        <Database size={20} />
        <span>Dataset Analysis</span>
      </h2>
      <div className="grid">
        <section className="card">
          <h3>Class Distribution</h3>
          <BarChart
            data={{
              labels: ["Benign", "Malicious"],
              datasets: [
                { label: "Train", data: [d.train.benign, d.train.malicious], backgroundColor: "#2563eb" },
                { label: "Test", data: [d.test.benign, d.test.malicious], backgroundColor: "#7c3aed" },
              ],
            }}
          />
        </section>

        <section className="card wide">
          <h3>Top Feature Statistics</h3>
          <DataTableCard rows={stats} columns={columns} height={430} pageSize={12} sortModel={[{ field: "std", sort: "desc" }]} />
        </section>
      </div>
    </>
  );
}
