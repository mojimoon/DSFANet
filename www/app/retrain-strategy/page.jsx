"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Box,
  Card,
  CardContent,
  Chip,
  FormControl,
  Grid,
  InputLabel,
  MenuItem,
  Paper,
  Select,
  Stack,
  Tab,
  Tabs,
  Typography,
} from "@mui/material";
import { ThemeProvider, createTheme } from "@mui/material/styles";
import { DataGrid } from "@mui/x-data-grid";
import { BarChart, LineChart } from "@/components/charts";
import { fetchApi, num } from "@/lib/api";
import { RefreshCcw } from "lucide-react";
import LoadingOverlay from "@/components/LoadingOverlay";

const MODEL_ORDER = ["RandomForest", "SGD", "AE", "LSTM", "DSFANet", "Voting", "Stacking", "XGBoostStacking"];

function normalizeAttack(raw) {
  const key = String(raw || "").toLowerCase();
  if (key === "adv_pgd") return "PGD";
  if (key === "adv_gdkde") return "GDKDE";
  if (key === "adv_fgsm") return "FGSM";
  if (key === "adv_mimicry") return "Mimicry";
  if (key.startsWith("ood_") || key.includes("bot_iot")) return "Natural (BoT-IoT)";
  return String(raw || "");
}

function normalizeModel(raw) {
  const key = String(raw || "");
  if (key === "XGBoostStacking") return "XGBoostStacking";
  return key;
}

function toNumber(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function pickFirstNumber(obj, keys, fallback = 0) {
  for (const key of keys) {
    if (obj && obj[key] !== undefined && obj[key] !== null && obj[key] !== "") {
      return toNumber(obj[key], fallback);
    }
  }
  return fallback;
}

function optionValues(rows, key) {
  const uniq = new Set(rows.map((x) => x[key]));
  return ["All", ...Array.from(uniq).sort((a, b) => String(a).localeCompare(String(b)))];
}

function mean(values) {
  if (!values.length) return 0;
  return values.reduce((acc, cur) => acc + cur, 0) / values.length;
}

function byModelOrder(a, b) {
  const ia = MODEL_ORDER.includes(a) ? MODEL_ORDER.indexOf(a) : 999;
  const ib = MODEL_ORDER.includes(b) ? MODEL_ORDER.indexOf(b) : 999;
  if (ia !== ib) return ia - ib;
  return String(a).localeCompare(String(b));
}

const pageTheme = createTheme({
  palette: {
    primary: { main: "#0f766e" },
    secondary: { main: "#1d4ed8" },
    background: { default: "#f5f7fb" },
  },
  shape: { borderRadius: 12 },
  typography: {
    fontFamily: '"Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif',
    h4: { fontWeight: 700 },
    h6: { fontWeight: 600 },
  },
});

export default function RetrainStrategyPage() {
  const [allRows, setAllRows] = useState([]);
  const [error, setError] = useState("");
  const [loaded, setLoaded] = useState(false);
  const [tabIndex, setTabIndex] = useState(0);

  const [modelFilter, setModelFilter] = useState("All");
  const [attackFilter, setAttackFilter] = useState("All");
  const [metricFilter, setMetricFilter] = useState("All");
  const [budgetFilter, setBudgetFilter] = useState("All");
  const [idRatioFilter, setIdRatioFilter] = useState("All");

  useEffect(() => {
    fetchApi("/api/retrain-strategy")
      .then((rows) => {
        const sourceRows = Array.isArray(rows) ? rows : [];
        const flat = sourceRows.map((r, idx) => ({
          run_id: String(r.run_id || "latest"),
          model: normalizeModel(r.model),
          attack: normalizeAttack(r.drift_case),
          selection_metric: String(r.selection_metric || ""),
          budget: pickFirstNumber(r, ["budget_ratio", "budget"], 0),
          id_ratio: pickFirstNumber(r, ["id_ratio", "idRate", "id"], 0),
          acc: pickFirstNumber(r, ["before_acc", "acc", "base_acc"], 0),
          retrain_acc: pickFirstNumber(r, ["after_acc", "retrain_acc", "new_acc"], 0),
          acc_gain: pickFirstNumber(r, ["acc_gain", "gain"], 0),
          _idx: idx,
        }));
        setAllRows(flat);
        setLoaded(true);
      })
      .catch((e) => {
        setError(e.message || "Failed to load retrain strategy data.");
        setLoaded(true);
      });
  }, []);

  const models = useMemo(() => optionValues(allRows, "model"), [allRows]);
  const attacks = useMemo(() => optionValues(allRows, "attack"), [allRows]);
  const metrics = useMemo(() => optionValues(allRows, "selection_metric"), [allRows]);
  const budgets = useMemo(() => optionValues(allRows, "budget"), [allRows]);
  const idRatios = useMemo(() => optionValues(allRows, "id_ratio"), [allRows]);

  const filteredRows = useMemo(
    () =>
      allRows.filter((r) => {
        if (modelFilter !== "All" && r.model !== modelFilter) return false;
        if (attackFilter !== "All" && r.attack !== attackFilter) return false;
        if (metricFilter !== "All" && r.selection_metric !== metricFilter) return false;
        if (budgetFilter !== "All" && String(r.budget) !== String(budgetFilter)) return false;
        if (idRatioFilter !== "All" && String(r.id_ratio) !== String(idRatioFilter)) return false;
        return true;
      }),
    [allRows, modelFilter, attackFilter, metricFilter, budgetFilter, idRatioFilter]
  );

  const chartByModel = useMemo(() => {
    const modelNames = Array.from(new Set(filteredRows.map((x) => x.model))).sort(byModelOrder);
    return {
      labels: modelNames,
      datasets: [
        {
          label: "Mean ACC Gain",
          data: modelNames.map((m) => mean(filteredRows.filter((x) => x.model === m).map((x) => x.acc_gain))),
          backgroundColor: "#2563eb",
        },
      ],
    };
  }, [filteredRows]);

  const chartByBudget = useMemo(() => {
    const budgetKeys = Array.from(new Set(filteredRows.map((x) => x.budget))).sort((a, b) => a - b);
    return {
      labels: budgetKeys.map((x) => num(x, 2)),
      datasets: [
        {
          label: "Mean ACC Gain",
          data: budgetKeys.map((b) => mean(filteredRows.filter((x) => x.budget === b).map((x) => x.acc_gain))),
          borderColor: "#16a34a",
          backgroundColor: "#16a34a",
          tension: 0.2,
        },
      ],
    };
  }, [filteredRows]);

  const chartByIdRatio = useMemo(() => {
    const idKeys = Array.from(new Set(filteredRows.map((x) => x.id_ratio))).sort((a, b) => a - b);
    return {
      labels: idKeys.map((x) => num(x, 2)),
      datasets: [
        {
          label: "Mean ACC Gain",
          data: idKeys.map((id) => mean(filteredRows.filter((x) => x.id_ratio === id).map((x) => x.acc_gain))),
          borderColor: "#7c3aed",
          backgroundColor: "#7c3aed",
          tension: 0.2,
        },
      ],
    };
  }, [filteredRows]);

  const chartByAttack = useMemo(() => {
    const attackNames = Array.from(new Set(filteredRows.map((x) => x.attack))).sort((a, b) => a.localeCompare(b));
    return {
      labels: attackNames,
      datasets: [
        {
          label: "Mean ACC Gain",
          data: attackNames.map((a) => mean(filteredRows.filter((x) => x.attack === a).map((x) => x.acc_gain))),
          backgroundColor: "#f59e0b",
        },
      ],
    };
  }, [filteredRows]);

  const kpiMeanGain = mean(filteredRows.map((x) => x.acc_gain));
  const kpiMeanAcc = mean(filteredRows.map((x) => x.acc));
  const kpiMeanRetrainAcc = mean(filteredRows.map((x) => x.retrain_acc));

  const dataGridRows = useMemo(
    () =>
      filteredRows.map((r, idx) => ({
        id: `${r.run_id}-${idx}`,
        ...r,
      })),
    [filteredRows]
  );

  const detailColumns = useMemo(
    () => [
      { field: "model", headerName: "Model", width: 160 },
      { field: "attack", headerName: "Attack", width: 160 },
      { field: "selection_metric", headerName: "Selection Metric", width: 170 },
      { field: "budget", headerName: "Budget", width: 100, valueFormatter: (p) => num(p.value, 2) },
      { field: "id_ratio", headerName: "ID Ratio", width: 100, valueFormatter: (p) => num(p.value, 2) },
      { field: "acc", headerName: "ACC", width: 120, valueFormatter: (p) => num(p.value, 4) },
      { field: "retrain_acc", headerName: "Retrain ACC", width: 130, valueFormatter: (p) => num(p.value, 4) },
      { field: "acc_gain", headerName: "ACC Gain", width: 150, valueFormatter: (p) => num(p.value, 4) },
    ],
    []
  );

  const pivotSummaryRows = useMemo(() => {
    const map = new Map();
    filteredRows.forEach((r) => {
      const key = `${r.model}`;
      if (!map.has(key)) {
        map.set(key, {
          id: key,
          model: r.model,
          n: 0,
          accValues: [],
          retrainAccValues: [],
          gainValues: [],
        });
      }
      const item = map.get(key);
      item.n += 1;
      item.accValues.push(r.acc);
      item.retrainAccValues.push(r.retrain_acc);
      item.gainValues.push(r.acc_gain);
    });

    return Array.from(map.values())
      .map((x) => ({
        id: x.id,
        model: x.model,
        n: x.n,
        acc: mean(x.accValues),
        retrain_acc: mean(x.retrainAccValues),
        acc_gain: mean(x.gainValues),
      }))
      .sort((a, b) => byModelOrder(a.model, b.model));
  }, [filteredRows]);

  const pivotColumns = useMemo(
    () => [
      { field: "model", headerName: "Model", width: 180 },
      { field: "n", headerName: "N", width: 90 },
      { field: "acc", headerName: "Mean ACC", width: 140, valueFormatter: (p) => num(p.value, 4) },
      { field: "retrain_acc", headerName: "Mean Retrain ACC", width: 170, valueFormatter: (p) => num(p.value, 4) },
      { field: "acc_gain", headerName: "Mean ACC Gain", width: 150, valueFormatter: (p) => num(p.value, 4) },
    ],
    []
  );

  const pivotMatrix = useMemo(() => {
    const attacksAxis = Array.from(new Set(filteredRows.map((x) => x.attack))).sort((a, b) => a.localeCompare(b));
    const modelsAxis = Array.from(new Set(filteredRows.map((x) => x.model))).sort(byModelOrder);
    const matrix = attacksAxis.map((attack) => {
      const row = { attack };
      modelsAxis.forEach((model) => {
        const vals = filteredRows.filter((x) => x.attack === attack && x.model === model).map((x) => x.acc_gain);
        row[model] = vals.length ? mean(vals) : null;
      });
      return row;
    });
    return { models: modelsAxis, rows: matrix };
  }, [filteredRows]);

  const pivotMatrixColumns = useMemo(
    () => [
      { field: "attack", headerName: "Attack", minWidth: 180, flex: 1 },
      ...pivotMatrix.models.map((model) => ({
        field: model,
        headerName: model,
        minWidth: 150,
        flex: 1,
        renderCell: (params) => {
          const v = params.value;
          let bg = "transparent";
          if (v !== null && v !== undefined) {
            if (v >= 0.25) bg = "#dcfce7";
            else if (v >= 0.1) bg = "#fef9c3";
            else if (v < 0) bg = "#fee2e2";
          }
          return (
            <Box sx={{ width: "100%", px: 1, py: 0.5, borderRadius: 1, backgroundColor: bg, textAlign: "right" }}>
              {v === null || v === undefined ? "-" : num(v, 4)}
            </Box>
          );
        },
      })),
    ],
    [pivotMatrix.models]
  );

  if (error) {
    return (
      <>
        <h2 className="pageTitle">Retrain Strategy</h2>
        <p>{error}</p>
        <p className="subtle">Run step 8 export and ensure /api/retrain-strategy has rows.</p>
      </>
    );
  }

  if (!loaded) return <LoadingOverlay text="Loading retrain strategy analysis..." />;

  if (!allRows.length) {
    return (
      <>
        <h2 className="pageTitle">Retrain Strategy</h2>
        <p>No retraining rows for the currently selected dataset.</p>
      </>
    );
  }

  return (
    <ThemeProvider theme={pageTheme}>
      <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
        <Box>
          <Typography variant="h4" sx={{ color: "#0b1324", display: "flex", alignItems: "center", gap: 1 }}>
            <RefreshCcw size={24} />
            Retrain Strategy Explorer
          </Typography>
        </Box>

        <Card elevation={2} sx={{ borderRadius: 3 }}>
          <CardContent>
            <Grid container spacing={1.5}>
              <Grid item xs={12} md={4}>
                <FormControl fullWidth size="small">
                  <InputLabel>Model</InputLabel>
                  <Select label="Model" value={modelFilter} onChange={(e) => setModelFilter(e.target.value)}>
                    {models.map((x) => (
                      <MenuItem key={String(x)} value={x}>
                        {String(x)}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>

              <Grid item xs={12} md={4}>
                <FormControl fullWidth size="small">
                  <InputLabel>Attack</InputLabel>
                  <Select label="Attack" value={attackFilter} onChange={(e) => setAttackFilter(e.target.value)}>
                    {attacks.map((x) => (
                      <MenuItem key={String(x)} value={x}>
                        {String(x)}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>

              <Grid item xs={12} md={4}>
                <FormControl fullWidth size="small">
                  <InputLabel>Selection Metric</InputLabel>
                  <Select label="Selection Metric" value={metricFilter} onChange={(e) => setMetricFilter(e.target.value)}>
                    {metrics.map((x) => (
                      <MenuItem key={String(x)} value={x}>
                        {String(x)}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>

              <Grid item xs={12} md={4}>
                <FormControl fullWidth size="small">
                  <InputLabel>Budget</InputLabel>
                  <Select label="Budget" value={budgetFilter} onChange={(e) => setBudgetFilter(e.target.value)}>
                    {budgets.map((x) => (
                      <MenuItem key={String(x)} value={x}>
                        {String(x)}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>

              <Grid item xs={12} md={4}>
                <FormControl fullWidth size="small">
                  <InputLabel>ID Ratio</InputLabel>
                  <Select label="ID Ratio" value={idRatioFilter} onChange={(e) => setIdRatioFilter(e.target.value)}>
                    {idRatios.map((x) => (
                      <MenuItem key={String(x)} value={x}>
                        {String(x)}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
            </Grid>

            <Stack direction="row" spacing={1} sx={{ mt: 2, flexWrap: "wrap", rowGap: 1 }}>
              <Chip label={`Filtered Rows: ${filteredRows.length}`} color="primary" variant="outlined" />
              <Chip label={`Mean ACC: ${num(kpiMeanAcc, 4)}`} color="secondary" variant="outlined" />
              <Chip label={`Mean Retrain ACC: ${num(kpiMeanRetrainAcc, 4)}`} color="secondary" variant="outlined" />
              <Chip label={`Mean ACC Gain: ${num(kpiMeanGain, 4)}`} color="success" variant="filled" />
            </Stack>
          </CardContent>
        </Card>

        <Grid container spacing={2}>
          <Grid item xs={12} lg={6}>
            <Card elevation={2} sx={{ borderRadius: 3 }}>
              <CardContent>
                <Typography variant="h6" sx={{ mb: 1 }}>Mean ACC Gain by Model</Typography>
                <BarChart data={chartByModel} options={{ plugins: { legend: { display: false } }, scales: { y: { title: { display: true, text: "ACC Gain" } } } }} />
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} lg={6}>
            <Card elevation={2} sx={{ borderRadius: 3 }}>
              <CardContent>
                <Typography variant="h6" sx={{ mb: 1 }}>Mean ACC Gain vs Budget</Typography>
                <LineChart data={chartByBudget} options={{ plugins: { legend: { display: false } }, scales: { y: { title: { display: true, text: "ACC Gain" } } } }} />
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} lg={6}>
            <Card elevation={2} sx={{ borderRadius: 3 }}>
              <CardContent>
                <Typography variant="h6" sx={{ mb: 1 }}>Mean ACC Gain vs ID Ratio</Typography>
                <LineChart data={chartByIdRatio} options={{ plugins: { legend: { display: false } }, scales: { y: { title: { display: true, text: "ACC Gain" } } } }} />
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} lg={6}>
            <Card elevation={2} sx={{ borderRadius: 3 }}>
              <CardContent>
                <Typography variant="h6" sx={{ mb: 1 }}>Mean ACC Gain by Attack</Typography>
                <BarChart data={chartByAttack} options={{ plugins: { legend: { display: false } }, scales: { y: { title: { display: true, text: "ACC Gain" } } } }} />
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        <Card elevation={2} sx={{ borderRadius: 3 }}>
          <CardContent>
            <Tabs value={tabIndex} onChange={(_, v) => setTabIndex(v)} sx={{ mb: 1.5 }}>
              <Tab label="Detail Table" />
              <Tab label="Pivot View" />
            </Tabs>

            {tabIndex === 0 ? (
              <Paper variant="outlined" sx={{ height: 560, borderRadius: 2, overflow: "hidden" }}>
                <DataGrid
                  rows={dataGridRows}
                  columns={detailColumns}
                  pageSizeOptions={[25, 50, 100]}
                  initialState={{ pagination: { paginationModel: { pageSize: 25, page: 0 } }, sorting: { sortModel: [{ field: "acc_gain", sort: "desc" }] } }}
                  disableRowSelectionOnClick
                />
              </Paper>
            ) : (
              <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
                <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>Pivot Summary: Model</Typography>
                <Paper variant="outlined" sx={{ height: 360, borderRadius: 2, overflow: "hidden" }}>
                  <DataGrid
                    rows={pivotSummaryRows}
                    columns={pivotColumns}
                    pageSizeOptions={[10, 25, 50]}
                    initialState={{ pagination: { paginationModel: { pageSize: 10, page: 0 } }, sorting: { sortModel: [{ field: "acc_gain", sort: "desc" }] } }}
                    disableRowSelectionOnClick
                  />
                </Paper>

                <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>Pivot Matrix: Mean ACC Gain (Attack x Model)</Typography>
                <Paper variant="outlined" sx={{ height: 400, borderRadius: 2, overflow: "hidden" }}>
                  <DataGrid
                    rows={pivotMatrix.rows.map((r) => ({ id: r.attack, ...r }))}
                    columns={pivotMatrixColumns}
                    pageSizeOptions={[10, 25, 50]}
                    initialState={{ pagination: { paginationModel: { pageSize: 10, page: 0 } } }}
                    disableRowSelectionOnClick
                    density="compact"
                  />
                </Paper>
              </Box>
            )}
          </CardContent>
        </Card>
      </Box>
    </ThemeProvider>
  );
}
