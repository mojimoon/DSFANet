"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { BarChart } from "@/components/charts";
import { fetchApi, num } from "@/lib/api";

export default function InstanceDetailPage() {
  const params = useParams();
  const instanceId = params?.instanceId;
  const [detail, setDetail] = useState(null);

  useEffect(() => {
    if (!instanceId) {
      return;
    }
    fetchApi(`/api/sample/${instanceId}`).then(setDetail).catch(console.error);
  }, [instanceId]);

  if (!detail) {
    return <p>Loading instance details...</p>;
  }

  const entries = Object.entries(detail.model_scores || {});
  const rows = [
    ...(detail.top_static_features || []).map((x) => ({ group: "static", ...x })),
    ...(detail.top_temporal_features || []).map((x) => ({ group: "temporal", ...x })),
  ];

  return (
    <>
      <h2 className="pageTitle">Instance: {instanceId}</h2>
      <div className="grid">
        <section className="card">
          <h3>Model Scores</h3>
          <BarChart
            data={{
              labels: entries.map(([k]) => k),
              datasets: [{ label: "Score", data: entries.map(([, v]) => v), backgroundColor: "#7c3aed" }],
            }}
            options={{ scales: { y: { min: 0, max: 1 } } }}
          />
        </section>

        <section className="card">
          <h3>Summary</h3>
          <p>Sample ID: {detail.sample_id}</p>
          <p>Ground Truth Label: {detail.label}</p>
        </section>

        <section className="card wide">
          <h3>Feature Values</h3>
          <div className="tableWrap">
            <table>
              <thead>
                <tr>
                  <th>Group</th>
                  <th>Feature</th>
                  <th>Value</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r, idx) => (
                  <tr key={`${r.group}-${r.feature}-${idx}`}>
                    <td>{r.group}</td>
                    <td>{r.feature}</td>
                    <td>{num(r.value, 5)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </>
  );
}
