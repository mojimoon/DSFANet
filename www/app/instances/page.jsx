"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { fetchApi, num } from "@/lib/api";

export default function InstancesPage() {
  const [alerts, setAlerts] = useState([]);
  const [threshold, setThreshold] = useState(0.5);

  useEffect(() => {
    fetchApi("/api/alerts").then(setAlerts).catch(console.error);
  }, []);

  const filtered = alerts.filter((x) => Number(x.voting_score) >= threshold).slice(0, 200);

  return (
    <>
      <h2 className="pageTitle">Instance Pages</h2>
      <section className="card wide">
        <label>
          Threshold: {threshold.toFixed(2)}
          <input
            style={{ marginLeft: 8 }}
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={threshold}
            onChange={(e) => setThreshold(Number(e.target.value))}
          />
        </label>
      </section>
      <section className="card wide">
        <div className="tableWrap">
          <table>
            <thead>
              <tr>
                <th>Sample</th>
                <th>Voting Score</th>
                <th>Stacking Score</th>
                <th>Prediction</th>
                <th>Label</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((row) => (
                <tr key={row.sample_id}>
                  <td>
                    <Link href={`/instance/${row.sample_id}`}>{row.sample_id}</Link>
                  </td>
                  <td>{num(row.voting_score)}</td>
                  <td>{num(row.stacking_score)}</td>
                  <td>{row.pred}</td>
                  <td>{row.label}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </>
  );
}
