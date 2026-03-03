"use client";

import { useEffect, useState } from "react";
import { fetchApi, num } from "@/lib/api";

function inferColumns(rows) {
  if (!rows || rows.length === 0) {
    return [];
  }
  return Object.keys(rows[0]);
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
        <h2 className="pageTitle">Experiments</h2>
        <p>{error}</p>
        <p className="subtle">Run experiments_main.py with step 6 to export web payload.</p>
      </>
    );
  }

  if (!payload) {
    return <p>Loading experiment results...</p>;
  }

  const runs = Array.isArray(payload?.runs) ? payload.runs : [];

  return (
    <>
      <h2 className="pageTitle">Experiment Results</h2>
      <p className="subtle">Latest Run ID: {payload.latest_run_id}</p>

      <div className="grid">
        {runs.map((run) => {
          const keys = Object.keys(run).filter((k) => k.startsWith("summary_step"));
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
                    <div className="tableWrap">
                      <table>
                        <thead>
                          <tr>
                            {cols.map((c) => (
                              <th key={c}>{c}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {rows.map((r, idx) => (
                            <tr key={idx}>
                              {cols.map((c) => (
                                <td key={`${idx}-${c}`}>{typeof r[c] === "number" ? num(r[c], 5) : String(r[c])}</td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
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
