"use client";

import { useEffect, useState } from "react";
import { BarChart } from "@/components/charts";
import DataTableCard from "@/components/DataTableCard";
import { fetchApi, num } from "@/lib/api";
import { ShieldAlert } from "lucide-react";

export default function AttacksPage() {
  const [rows, setRows] = useState([]);

  const columns = [
    { field: "attack", headerName: "Attack", minWidth: 140, flex: 1 },
    { field: "model", headerName: "Model", minWidth: 160, flex: 1 },
    { field: "accuracy", headerName: "Accuracy", width: 120, valueFormatter: (v) => num(v) },
    { field: "recall", headerName: "Recall", width: 120, valueFormatter: (v) => num(v) },
    { field: "f1", headerName: "F1", width: 120, valueFormatter: (v) => num(v) },
    { field: "average_precision", headerName: "AP", width: 120, valueFormatter: (v) => num(v) },
  ];

  useEffect(() => {
    fetchApi("/api/dashboard")
      .then((d) => setRows(d.attack_results || []))
      .catch(console.error);
  }, []);

  if (!rows.length) {
    return <p>Loading attack analysis...</p>;
  }

  const attacks = [...new Set(rows.map((x) => x.attack))];
  const models = [...new Set(rows.map((x) => x.model))];

  return (
    <>
      <h2 className="pageTitle titleRow">
        <ShieldAlert size={20} />
        <span>Attack Comparison</span>
      </h2>
      <div className="grid">
        <section className="card">
          <h3>AP under Attack Types</h3>
          <BarChart
            data={{
              labels: attacks,
              datasets: models.map((m, i) => ({
                label: m,
                data: attacks.map((a) => rows.find((r) => r.attack === a && r.model === m)?.average_precision ?? 0),
                backgroundColor: i === 0 ? "#ef4444" : "#22c55e",
              })),
            }}
          />
        </section>

        <section className="card wide">
          <h3>Attack Metrics Table</h3>
          <DataTableCard rows={rows} columns={columns} height={430} pageSize={12} sortModel={[{ field: "average_precision", sort: "desc" }]} />
        </section>
      </div>
    </>
  );
}
