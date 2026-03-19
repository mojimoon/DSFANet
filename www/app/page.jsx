"use client";

import { useEffect, useState } from "react";
import { BarChart, LineChart } from "@/components/charts";
import { fetchApi, num } from "@/lib/api";
import LoadingOverlay from "@/components/LoadingOverlay";

export default function OverviewPage() {
  const [data, setData] = useState(null);

  useEffect(() => {
    fetchApi("/api/dashboard").then(setData).catch(console.error);
  }, []);

  if (!data) {
    return <LoadingOverlay text="Loading overview..." />;
  }

  const p = data.pr_curve.precision;
  const r = data.pr_curve.recall;
  const prPoints = p.map((v, i) => ({ x: r[i], y: v }));

  const histogramLabels = data.score_histogram.bins.slice(0, -1).map((v, i) => `${num(v, 2)}-${num(data.score_histogram.bins[i + 1], 2)}`);

  const driftLabels = data.drift_windows.map((x) => `W${x.window}`);

  const metricItems = [
    ["Accuracy", data.metrics.accuracy],
    ["Precision", data.metrics.precision],
    ["Recall", data.metrics.recall],
    ["F1", data.metrics.f1],
    ["Average Precision", data.metrics.average_precision],
    ["TP/FP/FN", `${data.confusion.tp}/${data.confusion.fp}/${data.confusion.fn}`],
  ];

  const renderShap = (name, list, color) => (
    <div>
      <h3>{name}</h3>
      <BarChart
        data={{
          labels: (list || []).slice(0, 12).map((x) => x.feature),
          datasets: [{ label: "mean |SHAP|", data: (list || []).slice(0, 12).map((x) => x.mean_abs_shap), backgroundColor: color }],
        }}
        options={{
          indexAxis: "y",
          responsive: true,
          plugins: { legend: { display: false } },
        }}
      />
    </div>
  );

  return (
    <>
      <h2 className="pageTitle">Overview</h2>
      <div className="grid">
        <section className="card">
          <h3>Key Metrics</h3>
          <div className="kpis">
            {metricItems.map(([name, value]) => (
              <div className="kpi" key={name}>
                <div className="name">{name}</div>
                <div className="value">{typeof value === "string" ? value : num(value)}</div>
              </div>
            ))}
          </div>
        </section>

        <section className="card">
          <h3>Primary PR Curve</h3>
          <LineChart
            data={{
              datasets: [{ label: "PR", data: prPoints, parsing: false, borderColor: "#2563eb", pointRadius: 0 }],
            }}
            options={{
              scales: { x: { type: "linear", min: 0, max: 1 }, y: { min: 0, max: 1 } },
            }}
          />
        </section>

        <section className="card">
          <h3>Score Distribution</h3>
          <BarChart
            data={{
              labels: histogramLabels,
              datasets: [{ label: "Count", data: data.score_histogram.counts, backgroundColor: "#7c3aed" }],
            }}
            options={{ plugins: { legend: { display: false } } }}
          />
        </section>

        <section className="card">
          <h3>Drift Windows</h3>
          <LineChart
            data={{
              labels: driftLabels,
              datasets: [
                { label: "Mean Score", data: data.drift_windows.map((x) => x.mean_score), borderColor: "#16a34a" },
                { label: "Positive Ratio", data: data.drift_windows.map((x) => x.positive_ratio), borderColor: "#dc2626" },
              ],
            }}
            options={{ scales: { y: { min: 0, max: 1 } } }}
          />
        </section>

        <section className="card wide">
          <h3>SHAP Top Features by Model</h3>
          <div className="shapGrid">
            {renderShap("Autoencoder", data.shap_by_model.Autoencoder, "#7c3aed")}
            {renderShap("LSTM", data.shap_by_model.LSTM, "#0ea5e9")}
            {renderShap("DSFANet", data.shap_by_model.DSFANet, "#16a34a")}
          </div>
        </section>
      </div>
    </>
  );
}
