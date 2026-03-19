"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { Box, Chip, Slider, Stack } from "@mui/material";
import DataTableCard from "@/components/DataTableCard";
import { fetchApi, num } from "@/lib/api";
import { buildHrefWithDataset, getStoredDataset } from "@/lib/dataset";
import { ListChecks } from "lucide-react";

export default function InstancesPage() {
  const [dataset, setDataset] = useState("");
  const [alerts, setAlerts] = useState([]);
  const [threshold, setThreshold] = useState(0.5);

  useEffect(() => {
    setDataset(getStoredDataset());
  }, []);

  useEffect(() => {
    fetchApi("/api/alerts", { min_score: threshold })
      .then((payload) => {
        if (Array.isArray(payload)) {
          setAlerts(payload);
          return;
        }
        setAlerts(Array.isArray(payload?.rows) ? payload.rows : []);
      })
      .catch(console.error);
  }, [threshold]);

  const columns = [
    {
      field: "sample_id",
      headerName: "Sample",
      width: 120,
      renderCell: (params) => <Link href={buildHrefWithDataset(`/instance/${params.value}`, dataset)}>{params.value}</Link>,
    },
    { field: "voting_score", headerName: "Voting Score", width: 150, valueFormatter: (v) => num(v) },
    { field: "stacking_score", headerName: "Stacking Score", width: 150, valueFormatter: (v) => num(v) },
    { field: "pred", headerName: "Prediction", width: 120 },
    { field: "label", headerName: "Label", width: 100 },
  ];

  return (
    <>
      <h2 className="pageTitle titleRow">
        <ListChecks size={20} />
        <span>Instance Pages</span>
      </h2>
      <section className="card wide">
        <Stack spacing={1.5}>
          <Stack direction="row" spacing={1} alignItems="center" sx={{ flexWrap: "wrap", rowGap: 1 }}>
            <Chip color="primary" variant="outlined" label={`Threshold: ${threshold.toFixed(2)}`} />
            <Chip color="secondary" variant="outlined" label={`Detected: ${alerts.length}`} />
          </Stack>
          <p className="subtle">Threshold controls the minimum voting score. Higher threshold means fewer but more confident alerts.</p>
          <Box sx={{ px: 1 }}>
            <Slider
              min={0}
              max={1}
              step={0.01}
              value={threshold}
              onChange={(_, v) => setThreshold(Number(v))}
            />
          </Box>
        </Stack>
      </section>
      <section className="card wide">
        <DataTableCard rows={alerts} columns={columns} height={500} pageSize={50} pageSizeOptions={[25, 50, 100, 200]} sortModel={[{ field: "voting_score", sort: "desc" }]} />
      </section>
    </>
  );
}
