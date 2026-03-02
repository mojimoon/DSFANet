"use client";

import { useEffect, useState } from "react";
import { BarChart } from "@/components/charts";
import { fetchApi, num } from "@/lib/api";

export default function BenchmarksPage() {
  const [rows, setRows] = useState([]);

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
      <h2 className="pageTitle">Model and Ensemble Benchmarks</h2>
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
          <div className="tableWrap">
            <table>
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Accuracy</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>F1</th>
                  <th>AP</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.model}>
                    <td>{r.model}</td>
                    <td>{num(r.accuracy)}</td>
                    <td>{num(r.precision)}</td>
                    <td>{num(r.recall)}</td>
                    <td>{num(r.f1)}</td>
                    <td>{num(r.average_precision)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </>
  );
}
