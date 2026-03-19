"use client";

import { useEffect, useState } from "react";
import { BarChart } from "@/components/charts";
import DataTableCard from "@/components/DataTableCard";
import { fetchApi, num } from "@/lib/api";
import { ShieldAlert } from "lucide-react";

export default function AttacksPage() {
  const [summary, setSummary] = useState([]);
  const [attacks, setAttacks] = useState([]);
  const [shifts, setShifts] = useState([]);

  const decorateRows = (rows) => {
    const losses = rows.map((x) => Number(x.acc_loss || 0));
    const minLoss = Math.min(...losses, 0);
    const maxLoss = Math.max(...losses, 0);
    return rows.map((x, idx) => ({ ...x, id: `${x.name}-${x.model}-${idx}`, __minLoss: minLoss, __maxLoss: maxLoss }));
  };

  const columns = [
    { field: "name", headerName: "Attack/Shift", minWidth: 180, flex: 1 },
    { field: "model", headerName: "Model", minWidth: 160, flex: 1 },
    { field: "baseline_acc", headerName: "Baseline ACC", width: 140, valueFormatter: (v) => num(v) },
    {
      field: "new_acc",
      headerName: "New ACC",
      width: 120,
      renderCell: (params) => {
        const v = Number(params.value || 0);
        const color = v >= 0.8 ? "#16a34a" : v <= 0.5 ? "#dc2626" : "#334155";
        return <span style={{ color, fontWeight: 700 }}>{num(v)}</span>;
      },
    },
    {
      field: "acc_loss",
      headerName: "ACC Loss",
      width: 120,
      renderCell: (params) => {
        const v = Number(params.value || 0);
        const row = params.row || {};
        const isMax = v === row.__maxLoss;
        const isMin = v === row.__minLoss;
        const cls = isMax ? "negValue" : isMin ? "posValue" : "";
        return <span className={cls}>{num(v)}</span>;
      },
    },
  ];

  useEffect(() => {
    fetchApi("/api/attacks")
      .then((d) => {
        setSummary(Array.isArray(d?.summary) ? d.summary : []);
        setAttacks(decorateRows(Array.isArray(d?.attacks) ? d.attacks : []));
        setShifts(decorateRows(Array.isArray(d?.shifts) ? d.shifts : []));
      })
      .catch(console.error);
  }, []);

  if (!summary.length && !attacks.length && !shifts.length) {
    return <p>Loading attack analysis...</p>;
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
          <h3>Adversarial Attacks</h3>
          <DataTableCard rows={attacks} columns={columns} height={360} pageSize={8} sortModel={[{ field: "acc_loss", sort: "desc" }]} />
        </section>

        <section className="card wide">
          <h3>Natural and Distribution Shifts</h3>
          <DataTableCard rows={shifts} columns={columns} height={360} pageSize={8} sortModel={[{ field: "acc_loss", sort: "desc" }]} />
        </section>
      </div>
    </>
  );
}
