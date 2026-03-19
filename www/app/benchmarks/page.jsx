"use client";

import { useEffect, useState } from "react";
import { BarChart } from "@/components/charts";
import DataTableCard from "@/components/DataTableCard";
import { fetchApi, num } from "@/lib/api";
import { Trophy } from "lucide-react";

export default function BenchmarksPage() {
  const [rows, setRows] = useState([]);

  const columns = [
    { field: "model", headerName: "Model", minWidth: 180, flex: 1 },
    { field: "accuracy", headerName: "Accuracy", width: 120, valueFormatter: (v) => num(v) },
    { field: "precision", headerName: "Precision", width: 120, valueFormatter: (v) => num(v) },
    { field: "recall", headerName: "Recall", width: 120, valueFormatter: (v) => num(v) },
    { field: "f1", headerName: "F1", width: 120, valueFormatter: (v) => num(v) },
    { field: "average_precision", headerName: "AP", width: 120, valueFormatter: (v) => num(v) },
  ];

  useEffect(() => {
    fetchApi("/api/dashboard")
      .then((d) => setRows(d.benchmark_models || []))
      .catch(console.error);
  }, []);

  if (!rows.length) {
    return <p>Loading benchmark comparison...</p>;
  }

  return (
    <>
      <h2 className="pageTitle titleRow">
        <Trophy size={20} />
        <span>Model and Ensemble Benchmarks</span>
      </h2>
      <div className="grid">
        <section className="card">
          <h3>Average Precision by Model</h3>
          <BarChart
            data={{
              labels: rows.map((x) => x.model),
              datasets: [{ label: "AP", data: rows.map((x) => x.average_precision), backgroundColor: "#0ea5e9" }],
            }}
            options={{ plugins: { legend: { display: false } } }}
          />
        </section>

        <section className="card wide">
          <h3>Metrics Table</h3>
          <DataTableCard rows={rows} columns={columns} height={430} pageSize={10} sortModel={[{ field: "average_precision", sort: "desc" }]} />
        </section>
      </div>
    </>
  );
}
