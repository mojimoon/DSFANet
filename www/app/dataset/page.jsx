"use client";

import { useEffect, useState } from "react";
import { BarChart, PieChart } from "@/components/charts";
import DataTableCard from "@/components/DataTableCard";
import { fetchApi, num } from "@/lib/api";
import { Database } from "lucide-react";
import LoadingOverlay from "@/components/LoadingOverlay";

export default function DatasetPage() {
  const [data, setData] = useState(null);
  const [stats, setStats] = useState(null);

  useEffect(() => {
    fetchApi("/api/dashboard").then(setData).catch(console.error);
    fetchApi("/api/dataset/stats").then(setStats).catch(console.error);
  }, []);

  if (!data || !stats) {
    return <LoadingOverlay text="Loading dataset analysis..." />;
  }

  const classRows = stats.class_distribution || { benign: 0, malicious: 0 };
  const featureStats = [
    ...data.dataset_overview.feature_stats.static_top_variance.map((x) => ({ group: "static", ...x })),
    ...data.dataset_overview.feature_stats.temporal_top_variance.map((x) => ({ group: "temporal", ...x })),
  ];
  const attackDist = Array.isArray(stats.attack_distribution) ? stats.attack_distribution : [];

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
                { label: "Count", data: [classRows.benign || 0, classRows.malicious || 0], backgroundColor: ["#0ea5e9", "#ef4444"] },
              ],
            }}
            options={{ plugins: { legend: { display: false } } }}
          />
        </section>

        <section className="card">
          <h3>Attack Type Distribution</h3>
          <PieChart
            data={{
              labels: attackDist.slice(0, 12).map((x) => x.attack),
              datasets: [
                {
                  data: attackDist.slice(0, 12).map((x) => x.count),
                  backgroundColor: ["#0ea5e9", "#10b981", "#f59e0b", "#f43f5e", "#6366f1", "#14b8a6", "#22c55e", "#eab308", "#ef4444", "#06b6d4", "#8b5cf6", "#84cc16"],
                },
              ],
            }}
            options={{ plugins: { legend: { position: "bottom" } } }}
          />
        </section>

        <section className="card wide">
          <h3>Top Feature Statistics</h3>
          <DataTableCard rows={featureStats} columns={columns} height={430} pageSize={12} sortModel={[{ field: "std", sort: "desc" }]} />
        </section>
      </div>
    </>
  );
}
