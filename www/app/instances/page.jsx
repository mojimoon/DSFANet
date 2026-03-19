"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { Box, Button, Chip, MenuItem, Select, Slider, Stack } from "@mui/material";
import DataTableCard from "@/components/DataTableCard";
import { fetchApi, num } from "@/lib/api";
import { buildHrefWithDataset, getStoredDataset } from "@/lib/dataset";
import { ListChecks } from "lucide-react";

export default function InstancesPage() {
  const [dataset, setDataset] = useState("");
  const [alerts, setAlerts] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(50);
  const [threshold, setThreshold] = useState(0.5);

  useEffect(() => {
    setDataset(getStoredDataset());
  }, []);

  useEffect(() => {
    fetchApi("/api/alerts", { page, page_size: pageSize, min_score: threshold })
      .then((payload) => {
        setAlerts(Array.isArray(payload?.rows) ? payload.rows : []);
        setTotal(Number(payload?.total || 0));
      })
      .catch(console.error);
  }, [page, pageSize, threshold]);

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
            <Chip color="secondary" variant="outlined" label={`Rows: ${alerts.length}/${total}`} />
          </Stack>
          <p className="subtle">Threshold controls the minimum voting score. Higher threshold means fewer but more confident alerts.</p>
          <Box sx={{ px: 1 }}>
            <Slider
              min={0}
              max={1}
              step={0.01}
              value={threshold}
              onChange={(_, v) => {
                setThreshold(Number(v));
                setPage(1);
              }}
            />
          </Box>
        </Stack>
      </section>
      <section className="card wide">
        <DataTableCard rows={alerts} columns={columns} height={500} pageSize={pageSize} sortModel={[{ field: "voting_score", sort: "desc" }]} />
        <Stack direction="row" spacing={1} alignItems="center" sx={{ mt: 1.5 }}>
          <Button variant="outlined" size="small" disabled={page <= 1} onClick={() => setPage((p) => Math.max(1, p - 1))}>
            Prev
          </Button>
          <Chip label={`Page ${page}`} />
          <Button variant="outlined" size="small" disabled={page * pageSize >= total} onClick={() => setPage((p) => p + 1)}>
            Next
          </Button>
          <Select size="small" value={pageSize} onChange={(e) => {
            setPageSize(Number(e.target.value));
            setPage(1);
          }}>
            {[25, 50, 100, 200].map((size) => (
              <MenuItem key={size} value={size}>{size}/page</MenuItem>
            ))}
          </Select>
        </Stack>
      </section>
    </>
  );
}
