"use client";

import { useEffect, useState } from "react";
import { BarChart } from "@/components/charts";
import { fetchApi, num } from "@/lib/api";

export default function AttacksPage() {
  const [rows, setRows] = useState([]);

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
      <h2 className="pageTitle">Attack Comparison</h2>
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
          <div className="tableWrap">
            <table>
              <thead>
                <tr>
                  <th>Attack</th>
                  <th>Model</th>
                  <th>Accuracy</th>
                  <th>Recall</th>
                  <th>F1</th>
                  <th>AP</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r, idx) => (
                  <tr key={`${r.attack}-${r.model}-${idx}`}>
                    <td>{r.attack}</td>
                    <td>{r.model}</td>
                    <td>{num(r.accuracy)}</td>
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
