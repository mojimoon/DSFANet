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
    fetchApi("/api/experiments/latest")
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

  const keys = Object.keys(payload).filter((k) => k.startsWith("summary_step"));

  return (
    <>
      <h2 className="pageTitle">Experiment Results</h2>
      <p className="subtle">Run ID: {payload.run_id}</p>
      <p className="subtle">Generated: {payload.generated_at}</p>

      <div className="grid">
        {keys.map((key) => {
          const rows = payload[key] || [];
          const cols = inferColumns(rows);
          return (
            <section className="card wide" key={key}>
              <h3>{key}</h3>
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
            </section>
          );
        })}
      </div>
    </>
  );
}
