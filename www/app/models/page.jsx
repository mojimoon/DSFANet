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

function sortModelNames(names) {
  return [...names].sort((a, b) => {
    const ia = MODEL_ORDER.includes(a) ? MODEL_ORDER.indexOf(a) : -1;
    const ib = MODEL_ORDER.includes(b) ? MODEL_ORDER.indexOf(b) : -1;
    if (ia !== ib) {
      return ia - ib;
    }
    return String(a).localeCompare(String(b));
  });
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

  const modelNames = sortModelNames(Object.keys(models));
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

        <h3 style={{ marginTop: 12 }}>T2 Learners (Ensemble)</h3>
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
          </div>
        </section>
      )}
    </>
  );
}
