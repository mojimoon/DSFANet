"use client";

import { useMemo } from "react";
import { Paper } from "@mui/material";
import { DataGrid } from "@mui/x-data-grid";

export default function DataTableCard({
  rows,
  columns,
  height = 420,
  pageSize = 10,
  pageSizeOptions = [10, 25, 50],
  sortModel,
  getRowId,
}) {
  const safeRows = useMemo(() => {
    const arr = Array.isArray(rows) ? rows : [];
    if (getRowId) {
      return arr;
    }
    return arr.map((row, idx) => ({
      __row_id: row?.id ?? `${idx}`,
      ...row,
    }));
  }, [rows, getRowId]);

  const resolvedGetRowId = getRowId || ((row) => row.__row_id);

  return (
    <Paper variant="outlined" sx={{ height, borderRadius: 2, overflow: "hidden", background: "#fff" }}>
      <DataGrid
        rows={safeRows}
        columns={columns}
        getRowId={resolvedGetRowId}
        pageSizeOptions={pageSizeOptions}
        initialState={{
          pagination: { paginationModel: { pageSize, page: 0 } },
          sorting: sortModel ? { sortModel } : undefined,
        }}
        disableRowSelectionOnClick
        density="compact"
      />
    </Paper>
  );
}
