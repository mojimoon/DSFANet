"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { LineChart } from "@/components/charts";
import { fetchApi, num } from "@/lib/api";

export default function ModelDetailPage() {
  const params = useParams();
  const modelId = params?.modelId;
  const [detail, setDetail] = useState(null);

  useEffect(() => {
    if (!modelId) {
      return;
    }
    fetchApi(`/api/model/${encodeURIComponent(modelId)}`).then(setDetail).catch(console.error);
  }, [modelId]);

  if (!detail) {
    return <p>Loading model details...</p>;
  }

  const metricItems = [
    ["Accuracy", detail.metrics.accuracy],
    ["Precision", detail.metrics.precision],
    ["Recall", detail.metrics.recall],
    ["F1", detail.metrics.f1],
    ["Average Precision", detail.metrics.average_precision],
  ];

  const p = detail.pr_curve.precision;
  const r = detail.pr_curve.recall;

  return (
    <>
      <h2 className="pageTitle">Model: {modelId}</h2>
      <div className="grid">
        <section className="card">
          <h3>Metrics</h3>
          <div className="kpis">
            {metricItems.map(([k, v]) => (
              <div className="kpi" key={k}>
                <div className="name">{k}</div>
                <div className="value">{num(v)}</div>
              </div>
            ))}
          </div>
        </section>

        <section className="card">
          <h3>PR Curve</h3>
          <LineChart
            data={{ datasets: [{ label: "PR", data: p.map((x, i) => ({ x: r[i], y: x })), parsing: false, pointRadius: 0, borderColor: "#2563eb" }] }}
            options={{ scales: { x: { type: "linear", min: 0, max: 1 }, y: { min: 0, max: 1 } } }}
          />
        </section>

        <section className="card wide">
          <h3>Top Features</h3>
          <div className="tableWrap">
            <table>
              <thead>
                <tr>
                  <th>Feature</th>
                  <th>Weight</th>
                </tr>
              </thead>
              <tbody>
                {(detail.top_features || []).map((row, idx) => (
                  <tr key={`${row.feature}-${idx}`}>
                    <td>{row.feature}</td>
                    <td>{num(row.importance ?? row.mean_abs_shap, 6)}</td>
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
