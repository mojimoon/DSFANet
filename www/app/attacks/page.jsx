"use client";

import { useEffect, useState } from "react";
import { BarChart } from "@/components/charts";
import { fetchApi, num } from "@/lib/api";
import { ShieldAlert } from "lucide-react";
import LoadingOverlay from "@/components/LoadingOverlay";

const MODEL_ORDER = ["RandomForest", "SGD", "AE", "LSTM", "DSFANet", "Voting", "Stacking", "XGBoostStacking"];

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

function toNum(v, fallback = 0) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function toDeltaAcc(row) {
  if (row.acc_loss !== undefined && row.acc_loss !== null && row.acc_loss !== "") {
    return -toNum(row.acc_loss, 0);
  }
  return toNum(row.new_acc, 0) - toNum(row.baseline_acc, 0);
}

function buildMatrix(rows) {
  const safeRows = Array.isArray(rows) ? rows : [];
  const models = sortModels(Array.from(new Set(safeRows.map((r) => String(r.model || "")).filter(Boolean))));
  const names = Array.from(new Set(safeRows.map((r) => String(r.name || "")).filter(Boolean))).sort((a, b) => a.localeCompare(b));

  const baseAccByModel = {};
  models.forEach((model) => {
    const vals = safeRows.filter((r) => String(r.model || "") === model).map((r) => toNum(r.baseline_acc, NaN)).filter((v) => Number.isFinite(v));
    baseAccByModel[model] = vals.length ? vals.reduce((acc, cur) => acc + cur, 0) / vals.length : null;
  });

  const rowsMatrix = names.map((name) => {
    const cells = {};
    models.forEach((model) => {
      const hit = safeRows.find((r) => String(r.name || "") === name && String(r.model || "") === model);
      if (!hit) {
        cells[model] = null;
        return;
      }
      cells[model] = {
        deltaAcc: toDeltaAcc(hit),
        newAcc: toNum(hit.new_acc, 0),
      };
    });

    const deltas = models.map((m) => cells[m]?.deltaAcc).filter((v) => Number.isFinite(v));
    const averageDelta = deltas.length ? deltas.reduce((acc, cur) => acc + cur, 0) / deltas.length : null;
    const worstDelta = deltas.length ? Math.min(...deltas) : null;
    return {
      name,
      cells,
      averageDelta,
      worstDelta,
    };
  });

  return { models, rows: rowsMatrix, baseAccByModel };
}

function MatrixTable({ title, rows }) {
  const matrix = buildMatrix(rows);
  return (
    <section className="card wide">
      <h3>{title}</h3>
      <div className="matrixTableWrap">
        <table className="matrixTable">
          <thead>
            <tr>
              <th>Attacks/Shifts</th>
              {matrix.models.map((model) => (
                <th key={model}>
                  <div className="matrixHeadLabel">{model}</div>
                  <div className="matrixHeadSub">base: {matrix.baseAccByModel[model] === null ? "-" : num(matrix.baseAccByModel[model])}</div>
                </th>
              ))}
              <th>Average</th>
            </tr>
          </thead>
          <tbody>
            {matrix.rows.map((row) => (
              <tr key={row.name}>
                <th>{row.name}</th>
                {matrix.models.map((model) => {
                  const cell = row.cells[model];
                  if (!cell) {
                    return <td key={`${row.name}-${model}`}>-</td>;
                  }
                  const isWorst = row.worstDelta !== null && cell.deltaAcc === row.worstDelta && cell.deltaAcc < 0;
                  // const cls = cell.deltaAcc >= 0 ? "posValue" : "negValue";
                  const cls = cell.deltaAcc >= 0 ? "posValue" : cell.deltaAcc < -0.3 ? "negValue" : "";
                  return (
                    <td key={`${row.name}-${model}`}>
                      <div className={`matrixCell ${isWorst ? "matrixCellWorst" : ""}`}>
                        <div className={`matrixDelta ${cls}`}>{cell.deltaAcc >= 0 ? `+${num(cell.deltaAcc)}` : num(cell.deltaAcc)}</div>
                        <div className="matrixNewAcc">new: {num(cell.newAcc)}</div>
                      </div>
                    </td>
                  );
                })}
                <td>
                  {row.averageDelta === null ? (
                    "-"
                  ) : (
                    <span className={row.averageDelta >= 0 ? "posValue" : "negValue"}>{row.averageDelta >= 0 ? `+${num(row.averageDelta)}` : num(row.averageDelta)}</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export default function AttacksPage() {
  const [loading, setLoading] = useState(true);
  const [summary, setSummary] = useState([]);
  const [attacks, setAttacks] = useState([]);
  const [shifts, setShifts] = useState([]);

  useEffect(() => {
    fetchApi("/api/attacks")
      .then((d) => {
        setSummary(Array.isArray(d?.summary) ? d.summary : []);
        setAttacks(Array.isArray(d?.attacks) ? d.attacks : []);
        setShifts(Array.isArray(d?.shifts) ? d.shifts : []);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <LoadingOverlay text="Loading attack analysis..." />;
  }

  return (
    <>
      <h2 className="pageTitle titleRow">
        <ShieldAlert size={20} />
        <span>Attacks and Shifts</span>
      </h2>
      <div className="grid">
        <section className="card">
          <h3>Mean ACC Loss Across Models</h3>
          <BarChart
            data={{
              labels: summary.map((x) => x.name),
              datasets: [
                {
                  label: "Mean ACC Loss",
                  data: summary.map((x) => x.mean_acc_loss),
                  backgroundColor: summary.map((x) => (x.kind === "attack" ? "#ef4444" : "#f59e0b")),
                },
              ],
            }}
            options={{ plugins: { legend: { display: false } } }}
          />
        </section>

        <section className="card wide">
          <h3>Legend</h3>
          <div className="legendRow" style={{ marginBottom: 8 }}>
            <span>
              <strong>Delta Acc = new acc - baseline acc</strong>
            </span>
            <span>
              <span className="legendDot green" /> Positive delta acc
            </span>
            <span>
              <span className="legendDot red" /> Strongly negative delta acc
            </span>
          </div>
          <div className="legendRow">
            Tinted cells indicate the most negatively impacted model for that attack or shift.
          </div>
        </section>

        <MatrixTable title="Adversarial Attacks Accuracy Changes" rows={attacks} />
        <MatrixTable title="Natural and Distribution Shifts Accuracy Changes" rows={shifts} />
      </div>
    </>
  );
}
