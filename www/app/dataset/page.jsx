"use client";

import { useEffect, useState } from "react";
import { BarChart } from "@/components/charts";
import { fetchApi, num } from "@/lib/api";

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

  return (
    <>
      <h2 className="pageTitle">Dataset Analysis</h2>
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
          <div className="tableWrap">
            <table>
              <thead>
                <tr>
                  <th>Group</th>
                  <th>Feature</th>
                  <th>Mean</th>
                  <th>Std</th>
                  <th>Min</th>
                  <th>Max</th>
                </tr>
              </thead>
              <tbody>
                {stats.map((row, idx) => (
                  <tr key={`${row.feature}-${idx}`}>
                    <td>{row.group}</td>
                    <td>{row.feature}</td>
                    <td>{num(row.mean, 4)}</td>
                    <td>{num(row.std, 4)}</td>
                    <td>{num(row.min, 4)}</td>
                    <td>{num(row.max, 4)}</td>
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
