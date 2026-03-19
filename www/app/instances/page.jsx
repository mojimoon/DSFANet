"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { Box, Chip, Slider, Stack } from "@mui/material";
import DataTableCard from "@/components/DataTableCard";
import { fetchApi, num } from "@/lib/api";
import { ListChecks } from "lucide-react";

export default function InstancesPage() {
  const [alerts, setAlerts] = useState([]);
  const [threshold, setThreshold] = useState(0.5);

  useEffect(() => {
    fetchApi("/api/alerts").then(setAlerts).catch(console.error);
  }, []);

  const filtered = alerts.filter((x) => Number(x.voting_score) >= threshold).slice(0, 200);

  const columns = [
    {
      field: "sample_id",
      headerName: "Sample",
      width: 120,
      renderCell: (params) => <Link href={`/instance/${params.value}`}>{params.value}</Link>,
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
            <Chip color="secondary" variant="outlined" label={`Rows: ${filtered.length}`} />
          </Stack>
          <Box sx={{ px: 1 }}>
            <Slider min={0} max={1} step={0.01} value={threshold} onChange={(_, v) => setThreshold(Number(v))} />
          </Box>
        </Stack>
      </section>
      <section className="card wide">
        <DataTableCard rows={filtered} columns={columns} height={500} pageSize={25} sortModel={[{ field: "voting_score", sort: "desc" }]} />
      </section>
    </>
  );
}
