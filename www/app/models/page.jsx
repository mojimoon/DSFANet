"use client";

import { useEffect, useState } from "react";
import { LineChart } from "@/components/charts";
import DataTableCard from "@/components/DataTableCard";
import { fetchApi } from "@/lib/api";
import { Bot, Cpu } from "lucide-react";
import { num } from "@/lib/api";

export default function ModelsPage() {
  const [models, setModels] = useState({});
  const [selected, setSelected] = useState("");
  const [detail, setDetail] = useState(null);

  useEffect(() => {
    fetchApi("/api/models").then(setModels).catch(console.error);
  }, []);

  useEffect(() => {
    if (!selected) {
      setDetail(null);
      return;
    }
    fetchApi(`/api/model/${encodeURIComponent(selected)}`).then(setDetail).catch(console.error);
  }, [selected]);

  const modelNames = Object.keys(models);

  const featureRows = (detail?.top_features || []).map((row, idx) => ({
    id: `${row.feature}-${idx}`,
    feature: row.feature,
    weight: row.importance ?? row.mean_abs_shap,
  }));

  const featureColumns = [
    { field: "feature", headerName: "Feature", minWidth: 240, flex: 1 },
    { field: "weight", headerName: "Weight", width: 140, valueFormatter: (v) => num(v, 6) },
  ];

  const p = detail?.pr_curve?.precision || [];
  const r = detail?.pr_curve?.recall || [];

  return (
    <>
      <h2 className="pageTitle">Model Pages</h2>
      <div className="card">
        <div className="pillList">
          {modelNames.map((name) => (
            <button className="pill" key={name} onClick={() => setSelected(name)} style={{ cursor: "pointer" }}>
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
            <div className="card wide">
              <DataTableCard rows={featureRows} columns={featureColumns} height={360} pageSize={10} sortModel={[{ field: "weight", sort: "desc" }]} />
            </div>
          </div>
        </section>
      )}
    </>
  );
}
