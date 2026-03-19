"use client";

import { useEffect, useState } from "react";
import { BarChart } from "@/components/charts";
import { fetchApi, num } from "@/lib/api";
import { Trophy } from "lucide-react";

export default function BenchmarksPage() {
  const [rows, setRows] = useState([]);

  useEffect(() => {
    fetchApi("/api/benchmarks")
      .then((d) => setRows(Array.isArray(d) ? d : []))
      .catch(console.error);
  }, []);

  if (!rows.length) {
    return <p>Loading benchmark comparison...</p>;
  }

  return (
    <>
      <h2 className="pageTitle titleRow">
        <Trophy size={20} />
        <span>Model and Ensemble Benchmarks</span>
      </h2>
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
          <h3>Confusion Matrix by Model</h3>
          <div className="twoColCards">
            {rows.map((row) => {
              const c = row.confusion || { tn: 0, fp: 0, fn: 0, tp: 0 };
              const values = [c.tn, c.fp, c.fn, c.tp];
              const maxVal = Math.max(...values, 1);
              const tint = (v, hue) => `hsla(${hue}, 85%, 80%, ${0.25 + (Number(v) / maxVal) * 0.7})`;
              return (
                <div className="card" key={row.model}>
                  <h4>{row.model}</h4>
                  <div className="confMatrix">
                    <div className="confCell" style={{ background: tint(c.tn, 145) }}>TN: {c.tn}</div>
                    <div className="confCell" style={{ background: tint(c.fp, 15) }}>FP: {c.fp}</div>
                    <div className="confCell" style={{ background: tint(c.fn, 28) }}>FN: {c.fn}</div>
                    <div className="confCell" style={{ background: tint(c.tp, 195) }}>TP: {c.tp}</div>
                  </div>
                  <div className="metricRow">ACC: {num(row.accuracy)} | Precision: {num(row.precision)} | Recall: {num(row.recall)}</div>
                  <div className="metricRow">F1: {num(row.f1)} | AP: {num(row.average_precision)}</div>
                </div>
              );
            })}
          </div>
        </section>
      </div>
    </>
  );
}
