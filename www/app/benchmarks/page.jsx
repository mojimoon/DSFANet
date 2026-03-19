"use client";

import { useEffect, useState } from "react";
import { BarChart } from "@/components/charts";
import { fetchApi, num } from "@/lib/api";
import { Trophy } from "lucide-react";
import LoadingOverlay from "@/components/LoadingOverlay";

const MODEL_ORDER = ["RandomForest", "SGD", "AE", "LSTM", "DSFANet", "Voting", "Stacking", "XGBoostStacking"];

function sortByModelOrder(rows) {
  const rank = new Map(MODEL_ORDER.map((name, idx) => [name, idx]));
  return [...rows].sort((a, b) => {
    const ra = rank.has(a.model) ? rank.get(a.model) : 999;
    const rb = rank.has(b.model) ? rank.get(b.model) : 999;
    if (ra !== rb) {
      return ra - rb;
    }
    return String(a.model).localeCompare(String(b.model));
  });
}

function safeDivide(a, b) {
  return b ? a / b : 0;
}

export default function BenchmarksPage() {
  const [rows, setRows] = useState([]);

  useEffect(() => {
    fetchApi("/api/benchmarks")
      .then((d) => setRows(sortByModelOrder(Array.isArray(d) ? d : [])))
      .catch(console.error);
  }, []);

  if (!rows.length) {
    return <LoadingOverlay text="Loading benchmark comparison..." />;
  }

  return (
    <>
      <h2 className="pageTitle titleRow">
        <Trophy size={20} />
        <span>Model and Ensemble Benchmarks</span>
      </h2>
      <div className="grid">
        <section className="card">
          <h3>Accuracy by Model</h3>
          <BarChart
            data={{
              labels: rows.map((x) => x.model),
              datasets: [{ label: "ACC", data: rows.map((x) => x.accuracy), backgroundColor: "#16a34a" }],
            }}
            options={{ plugins: { legend: { display: false } } }}
          />
        </section>

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
              const precision = safeDivide(c.tp, c.tp + c.fp);
              const recall = safeDivide(c.tp, c.tp + c.fn);
              const f1 = safeDivide(2 * precision * recall, precision + recall);
              const values = [c.tn, c.fp, c.fn, c.tp];
              const maxVal = Math.max(...values, 1);
              const tint = (v, hue) => `hsla(${hue}, 85%, 80%, ${0.05 + (Number(v) / maxVal) * 0.95})`;
              return (
                <div className="card" key={row.model}>
                  <h4>{row.model}</h4>
                  <table className="confTable">
                    <thead>
                      <tr>
                        <th className="diagHeader">
                          <span className="diagTop">
                            <span>Predicted</span>
                            <span className="diagRight">Actual</span>
                          </span>
                        </th>
                        <th>Malicious</th>
                        <th>Benign</th>
                        <th></th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <th>Malicious</th>
                        <td style={{ background: tint(c.tp, 195) }}>TP: <span style={{ fontWeight: "bold" }}>{c.tp}</span></td>
                        <td style={{ background: tint(c.fp, 15) }}>FP: <span style={{ fontWeight: "bold" }}>{c.fp}</span></td>
                        <td>Precision: {num(precision)}</td>
                      </tr>
                      <tr>
                        <th>Benign</th>
                        <td style={{ background: tint(c.fn, 28) }}>FN: <span style={{ fontWeight: "bold" }}>{c.fn}</span></td>
                        <td style={{ background: tint(c.tn, 145) }}>TN: <span style={{ fontWeight: "bold" }}>{c.tn}</span></td>
                        <td></td>
                      </tr>
                      <tr>
                        <th></th>
                        <td>Recall: {num(recall)}</td>
                        <td></td>
                        <td>F1: <span style={{ fontWeight: "bold" }}>{num(f1)}</span></td>
                      </tr>
                    </tbody>
                  </table>
                  <div className="metricRow" style={{ display: "flex", justifyContent: "space-between", gap: 8 }}>
                    <span>ACC: <span style={{ fontWeight: "bold" }}>{num(row.accuracy)}</span></span>
                    <span>AP: <span style={{ fontWeight: "bold" }}>{num(row.average_precision)}</span></span>
                  </div>
                </div>
              );
            })}
          </div>
        </section>
      </div>
    </>
  );
}
