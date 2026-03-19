"use client";

import { useEffect, useState } from "react";
import DataTableCard from "@/components/DataTableCard";
import { fetchApi, num } from "@/lib/api";
import { FlaskConical } from "lucide-react";
import LoadingOverlay from "@/components/LoadingOverlay";

function inferColumns(rows) {
  if (!rows || rows.length === 0) {
    return [];
  }
  return Object.keys(rows[0]).map((key) => {
    const sample = rows.find((x) => x?.[key] !== null && x?.[key] !== undefined)?.[key];
    const isNumber = typeof sample === "number";
    return {
      field: key,
      headerName: key,
      minWidth: 150,
      flex: 1,
      valueFormatter: isNumber ? (value) => num(value, 5) : undefined,
    };
  });
}

export default function ExperimentsPage() {
  const [payload, setPayload] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    fetchApi("/api/experiments/all")
      .then(setPayload)
      .catch((e) => setError(e.message));
  }, []);

  if (error) {
    return (
      <>
        <h2 className="pageTitle titleRow">
          <FlaskConical size={20} />
          <span>Experiments</span>
        </h2>
        <p>{error}</p>
      </>
    );
  }

  if (!payload) {
    return <LoadingOverlay text="Loading experiment results..." />;
  }

  const runs = Array.isArray(payload?.runs) ? payload.runs : [];

  return (
    <>
      <h2 className="pageTitle titleRow">
        <FlaskConical size={20} />
        <span>Experiment Results</span>
      </h2>
      <p className="subtle">Latest Run ID: {payload.latest_run_id}</p>

      <div className="grid">
        {runs.map((run) => {
          const keys = Object.keys(run)
            .filter((k) => /^summary_step\d+$/.test(k))
            .sort((a, b) => {
              const na = Number((a.match(/summary_step(\d+)/) || ["", "999"])[1]);
              const nb = Number((b.match(/summary_step(\d+)/) || ["", "999"])[1]);
              return na - nb;
            });
          return (
            <section className="card wide" key={run.run_id}>
              <h3>{run.run_id}</h3>
              <p className="subtle">Base Dataset: {run.base_dataset}</p>
              <p className="subtle">Generated: {run.generated_at}</p>

              {keys.map((key) => {
                const rows = run[key] || [];
                const cols = inferColumns(rows);
                return (
                  <div key={`${run.run_id}-${key}`} style={{ marginTop: 12 }}>
                    <h4>{key}</h4>
                    <DataTableCard rows={rows} columns={cols} height={380} pageSize={10} />
                  </div>
                );
              })}
            </section>
          );
        })}
      </div>
    </>
  );
}
