"use client";

import { useEffect, useState } from "react";
import { BarChart, LineChart } from "@/components/charts";
import { fetchApi } from "@/lib/api";
import { Bot, Cpu } from "lucide-react";
import { num } from "@/lib/api";
import LoadingOverlay from "@/components/LoadingOverlay";

const MODEL_ORDER = ["RandomForest", "SGD", "AE", "LSTM", "DSFANet", "Voting", "Stacking", "XGBoostStacking"];
const T1_MODELS = ["RandomForest", "SGD", "AE", "LSTM", "DSFANet"];
const T2_MODELS = ["Voting", "Stacking", "XGBoostStacking"];

function sortModels(names) {
  return [...names].sort((a, b) => {
    const ia = MODEL_ORDER.includes(a) ? MODEL_ORDER.indexOf(a) : -1;
    const ib = MODEL_ORDER.includes(b) ? MODEL_ORDER.indexOf(b) : -1;
    if (ia !== ib) {
      return ia - ib;
    }
    return String(a).localeCompare(String(b));
  });
}

function ShapBoxPlot({ rows }) {
  const [active, setActive] = useState(null);

  if (!rows || !rows.length) {
    return <p className="subtle">No SHAP distribution data for this model.</p>;
  }

  const domainMin = Math.min(...rows.map((r) => Number(r.min ?? 0)));
  const domainMax = Math.max(...rows.map((r) => Number(r.max ?? 0)));
  const minBound = Math.min(domainMin, 0);
  const maxBound = Math.max(domainMax, 0);
  const denom = Math.max(maxBound - minBound, 1e-9);

  const leftPad = 190;
  const rightPad = 20;
  const plotWidth = 700;
  const rowHeight = 36;
  const topPad = 24;
  const svgWidth = leftPad + plotWidth + rightPad;
  const svgHeight = topPad + rows.length * rowHeight + 10;

  const x = (value) => leftPad + ((Number(value) - minBound) / denom) * plotWidth;
  const zeroX = x(0);

  return (
    <div>
      {active ? (
        <p className="subtle" style={{ marginTop: 0, marginBottom: 8 }}>
          {active.feature}: min {num(active.min, 4)} | q1 {num(active.q1, 4)} | median {num(active.median, 4)} | q3 {num(active.q3, 4)} | max {num(active.max, 4)}
        </p>
      ) : (
        <p className="subtle" style={{ marginTop: 0, marginBottom: 8 }}>Hover a box to inspect quantiles.</p>
      )}

      <div style={{ overflowX: "auto" }}>
        <svg viewBox={`0 0 ${svgWidth} ${svgHeight}`} width="100%" role="img" aria-label="SHAP distribution box plot">
          <line x1={zeroX} y1={8} x2={zeroX} y2={svgHeight - 6} stroke="#cbd5e1" strokeDasharray="4 4" />
          {rows.map((row, idx) => {
            const y = topPad + idx * rowHeight;
            const yMid = y + rowHeight * 0.5;
            const xMin = x(row.min);
            const xQ1 = x(row.q1);
            const xMedian = x(row.median);
            const xQ3 = x(row.q3);
            const xMax = x(row.max);

            return (
              <g key={row.feature} onMouseEnter={() => setActive(row)} onFocus={() => setActive(row)}>
                <text x={leftPad - 8} y={yMid + 4} textAnchor="end" fontSize="12" fill="#334155">
                  {row.feature}
                </text>

                <line x1={xMin} y1={yMid} x2={xMax} y2={yMid} stroke="#64748b" strokeWidth="1.5" />
                <line x1={xMin} y1={yMid - 7} x2={xMin} y2={yMid + 7} stroke="#64748b" strokeWidth="1.5" />
                <line x1={xMax} y1={yMid - 7} x2={xMax} y2={yMid + 7} stroke="#64748b" strokeWidth="1.5" />

                <rect x={Math.min(xQ1, xQ3)} y={yMid - 10} width={Math.max(Math.abs(xQ3 - xQ1), 1)} height={20} fill="#93c5fd" stroke="#2563eb" rx="2" />
                <line x1={xMedian} y1={yMid - 11} x2={xMedian} y2={yMid + 11} stroke="#1e3a8a" strokeWidth="2" />

                <rect
                  x={leftPad}
                  y={y + 1}
                  width={plotWidth}
                  height={rowHeight - 2}
                  fill="transparent"
                  style={{ cursor: "pointer" }}
                >
                  <title>
                    {`${row.feature} | min=${num(row.min, 4)}, q1=${num(row.q1, 4)}, median=${num(row.median, 4)}, q3=${num(row.q3, 4)}, max=${num(row.max, 4)}`}
                  </title>
                </rect>
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
}

export default function ModelsPage() {
  const [models, setModels] = useState({});
  const [loaded, setLoaded] = useState(false);
  const [selected, setSelected] = useState("");
  const [detail, setDetail] = useState(null);

  useEffect(() => {
    fetchApi("/api/models")
      .then(setModels)
      .catch(console.error)
      .finally(() => setLoaded(true));
  }, []);

  useEffect(() => {
    if (!selected) {
      setDetail(null);
      return;
    }
    fetchApi(`/api/model/${encodeURIComponent(selected)}`).then(setDetail).catch(console.error);
  }, [selected]);

  const modelNames = sortModels(Object.keys(models));
  const t1Names = modelNames.filter((name) => T1_MODELS.includes(name));
  const t2Names = modelNames.filter((name) => T2_MODELS.includes(name));

  const featureRows = (detail?.top_features || []).map((row, idx) => ({
    id: `${row.feature}-${idx}`,
    feature: row.feature,
    weight: row.importance ?? row.mean_abs_shap,
  }));

  const p = detail?.pr_curve?.precision || [];
  const r = detail?.pr_curve?.recall || [];

  if (!loaded) {
    return <LoadingOverlay text="Loading models..." />;
  }

  return (
    <>
      <h2 className="pageTitle">Model Pages</h2>
      <div className="card">
        <h3>T1 Learners</h3>
        <div className="pillList">
          {t1Names.map((name) => (
            <button className={`pill ${selected === name ? "active" : ""}`} key={name} onClick={() => setSelected(name)}>
              {name}
            </button>
          ))}
        </div>

        <h3 style={{ marginTop: 12 }}>T2 Learners</h3>
        <div className="pillList">
          {t2Names.map((name) => (
            <button className={`pill ${selected === name ? "active" : ""}`} key={name} onClick={() => setSelected(name)}>
              {name}
            </button>
          ))}
        </div>
      </div>

      {!detail ? (
        <section className="card wide" style={{ marginTop: 12 }}>
          <h3 className="titleRow">
            <Bot size={18} />
            <span>Select a model from above</span>
          </h3>
          <p className="subtle">Model details, PR curve, and feature importance will appear here.</p>
        </section>
      ) : (
        <section className="card wide" style={{ marginTop: 12 }}>
          <h3 className="titleRow">
            <Cpu size={18} />
            <span>{selected}</span>
          </h3>
          <div className="grid">
            <div className="card">
              <p>ACC: {num(detail.metrics?.accuracy)}</p>
              <p>Precision: {num(detail.metrics?.precision)}</p>
              <p>Recall: {num(detail.metrics?.recall)}</p>
              <p>F1: {num(detail.metrics?.f1)}</p>
              <p>AP: {num(detail.metrics?.average_precision)}</p>
            </div>
            <div className="card">
              <LineChart
                data={{ datasets: [{ label: "PR", data: p.map((x, i) => ({ x: r[i], y: x })), parsing: false, pointRadius: 0, borderColor: "#2563eb" }] }}
                options={{ scales: { x: { type: "linear", min: 0, max: 1 }, y: { min: 0, max: 1 } } }}
              />
            </div>
            {featureRows.length > 0 ? (
              <div className="card wide">
                <h3>Feature Importance</h3>
                <BarChart
                  data={{
                    labels: featureRows.map((x) => x.feature),
                    datasets: [{ label: "Weight", data: featureRows.map((x) => x.weight), backgroundColor: "#0ea5e9" }],
                  }}
                  options={{
                    indexAxis: "y",
                    plugins: { legend: { display: false } },
                  }}
                />
              </div>
            ) : null}

            {Array.isArray(detail?.shap_distribution) && detail.shap_distribution.length > 0 ? (
              <div className="card wide">
                <h3>SHAP Distribution (Box Plot)</h3>
                <ShapBoxPlot rows={detail.shap_distribution} />
              </div>
            ) : null}
          </div>
        </section>
      )}
    </>
  );
}
