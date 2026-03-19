"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { BarChart } from "@/components/charts";
import DataTableCard from "@/components/DataTableCard";
import { fetchApi, num } from "@/lib/api";
import { Fingerprint } from "lucide-react";
import LoadingOverlay from "@/components/LoadingOverlay";

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
    return <LoadingOverlay text="Loading instance details..." />;
  }

  const scoreMap = detail.model_scores || {};
  const scoreOrder = ["before_retrain_acc", "after_retrain_acc", "acc_gain"];
  const entries = scoreOrder
    .filter((key) => scoreMap[key] !== undefined)
    .map((key) => [key, scoreMap[key]]);
  const rows = [
    ...(detail.top_static_features || []).map((x) => ({ group: "static", ...x })),
    ...(detail.top_temporal_features || []).map((x) => ({ group: "temporal", ...x })),
  ];

  const columns = [
    { field: "group", headerName: "Group", width: 120 },
    { field: "feature", headerName: "Feature", minWidth: 260, flex: 1 },
    { field: "value", headerName: "Value", width: 140, valueFormatter: (v) => num(v, 5) },
  ];

  return (
    <>
      <h2 className="pageTitle titleRow">
        <Fingerprint size={20} />
        <span>Instance: {instanceId}</span>
      </h2>
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
          <DataTableCard rows={rows} columns={columns} height={430} pageSize={15} sortModel={[{ field: "group", sort: "asc" }]} />
        </section>
      </div>
    </>
  );
}
