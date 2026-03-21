"use client";

import { useEffect, useState } from "react";
import { BarChart, LineChart } from "@/components/charts";
import { fetchApi, num } from "@/lib/api";
import LoadingOverlay from "@/components/LoadingOverlay";

export default function OverviewPage() {
  const SHAP_MAX_FEATURES = 10;
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

  const toReadableTime = (isoString) => {
    if (!isoString) return "N/A";
    const date = new Date(isoString);
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, "0");
    const day = String(date.getDate()).padStart(2, "0");
    const hours = String(date.getHours()).padStart(2, "0");
    const minutes = String(date.getMinutes()).padStart(2, "0");
    return `${year}-${month}-${day} ${hours}:${minutes}`;
  }

  const metricItems = [
    ["Accuracy", data.metrics.accuracy],
    ["Precision", data.metrics.precision],
    ["Recall", data.metrics.recall],
    ["F1", data.metrics.f1],
    ["Average Precision", data.metrics.average_precision],
    ["Last Updated", toReadableTime(data.meta.generated_at)],
  ];

  const renderShap = (name, list, color) => {
    const items = (list || []).slice(0, SHAP_MAX_FEATURES);
    const chartHeight = Math.max(260, items.length * 26);

    return (
      <div>
        <h3>{name}</h3>
        <div className="shapChartWrap" style={{ height: `${chartHeight}px` }}>
          <BarChart
            data={{
              labels: items.map((x) => x.feature),
              datasets: [{ label: "mean", data: items.map((x) => x.mean_abs_shap), backgroundColor: color }],
            }}
            options={{
              indexAxis: "y",
              responsive: true,
              maintainAspectRatio: false,
              plugins: { legend: { display: false } },
              scales: {
                y: {
                  ticks: {
                    autoSkip: false,
                    font: { size: 11 },
                  },
                },
              },
            }}
          />
        </div>
      </div>
    );
  };

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
