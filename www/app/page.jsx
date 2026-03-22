"use client";

import { useEffect, useMemo, useState } from "react";
import { Alert, Button, Chip, Dialog, DialogActions, DialogContent, DialogTitle, Stack, TextField } from "@mui/material";
import { BarChart, LineChart } from "@/components/charts";
import { fetchApi, num, postApi } from "@/lib/api";
import LoadingOverlay from "@/components/LoadingOverlay";
import { AlertCircle, CheckCircle2, ChevronDown, ChevronUp, FlaskConical, LoaderCircle, Zap, PlayIcon } from "lucide-react";

const DEFAULT_DATASET = "NF-UNSW-NB15-v3.csv";
const DEFAULT_RETRAIN_FORM = {
  run_id: "unsw-main",
  base_dataset: DEFAULT_DATASET,
  steps: "3,8",
  epochs: "10,10,20",
  size_limit: 0,
  ood_dataset: "NF-BoT-IoT-v3.csv",
  device: "cpu",
};

const RETRAIN_PRESETS = [
  {
    key: "default",
    label: "Default Retrain",
    values: {
      steps: "3,8",
      epochs: "10,10,20",
      size_limit: 0,
      ood_dataset: "NF-BoT-IoT-v3.csv",
      device: "cpu",
    },
  },
  {
    key: "quick",
    label: "Quick Validation",
    values: {
      steps: "3,8",
      epochs: "3,3,5",
      size_limit: 3000,
      ood_dataset: "NF-BoT-IoT-v3.csv",
      device: "cpu",
    },
  },
  {
    key: "full",
    label: "Full Experiment",
    values: {
      steps: "1,2,3,4,5,6,7,8",
      epochs: "10,10,20",
      size_limit: 0,
      ood_dataset: "NF-BoT-IoT-v3.csv",
      device: "cuda",
    },
  },
];

export default function OverviewPage() {
  const SHAP_MAX_FEATURES = 10;
  const [data, setData] = useState(null);
  const [retrainOpen, setRetrainOpen] = useState(false);
  const [retrainForm, setRetrainForm] = useState(DEFAULT_RETRAIN_FORM);
  const [task, setTask] = useState(null);
  const [taskError, setTaskError] = useState("");
  const [refreshCountdown, setRefreshCountdown] = useState(null);
  const [logsExpanded, setLogsExpanded] = useState(false);

  useEffect(() => {
    fetchApi("/api/dashboard").then(setData).catch(console.error);
    fetchApi("/api/retrain/status")
      .then((payload) => {
        if (payload?.status && payload.status !== "idle") {
          setTask(payload);
        }
      })
      .catch(() => null);
  }, []);

  useEffect(() => {
    if (!task || !["starting", "running"].includes(task.status)) {
      return undefined;
    }

    const timer = window.setInterval(() => {
      fetchApi("/api/retrain/status")
        .then((payload) => setTask(payload))
        .catch(() => null);
    }, 1500);

    return () => window.clearInterval(timer);
  }, [task?.status]);

  useEffect(() => {
    if (!task || task.status !== "succeeded") {
      setRefreshCountdown(null);
      return undefined;
    }

    setRefreshCountdown(5);
    const timer = window.setInterval(() => {
      setRefreshCountdown((prev) => {
        if (prev === null) {
          return null;
        }
        if (prev <= 1) {
          window.clearInterval(timer);
          window.location.reload();
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => window.clearInterval(timer);
  }, [task?.task_id, task?.status]);

  const statusChipColor = useMemo(() => {
    if (!task) return "default";
    if (task.status === "succeeded") return "success";
    if (task.status === "failed") return "error";
    return "warning";
  }, [task]);

  const latestLogLine = useMemo(() => {
    if (!Array.isArray(task?.log_tail) || task.log_tail.length === 0) {
      return "";
    }
    return task.log_tail[task.log_tail.length - 1] || "";
  }, [task]);

  const allLogLines = useMemo(() => {
    if (!Array.isArray(task?.log_tail)) {
      return [];
    }
    return task.log_tail;
  }, [task]);

  const openRetrainDialog = () => {
    const currentDataset = String(data?.meta?.dataset || DEFAULT_DATASET);
    const currentRunId = String(data?.meta?.run_id || "unsw-main");
    setRetrainForm({
      run_id: currentRunId,
      base_dataset: currentDataset,
      steps: "3,8",
      epochs: "10,10,20",
      size_limit: 0,
      ood_dataset: "NF-BoT-IoT-v3.csv",
      device: "cpu",
    });
    setTaskError("");
    setRetrainOpen(true);
  };

  const onChangeForm = (key, value) => {
    setRetrainForm((prev) => ({ ...prev, [key]: value }));
  };

  const applyPreset = (presetValues) => {
    setRetrainForm((prev) => ({
      ...prev,
      ...presetValues,
      size_limit: Number(presetValues.size_limit ?? prev.size_limit ?? 0),
    }));
  };

  const submitRetrain = async (event) => {
    event.preventDefault();
    setTaskError("");

    const payload = {
      ...retrainForm,
      size_limit: Number(retrainForm.size_limit || 0),
    };

    try {
      const res = await postApi("/api/retrain/start", payload);
      if (res?.task) {
        setTask(res.task);
        setLogsExpanded(false);
      }
      setRetrainOpen(false);
    } catch (err) {
      const existingTask = err?.payload?.task;
      if (existingTask) {
        setTask(existingTask);
      }
      setTaskError(err?.message || "Failed to start retrain task.");
    }
  };

  if (!data) {
    return <LoadingOverlay text="Loading overview..." />;
  }

  const p = data.pr_curve.precision;
  const r = data.pr_curve.recall;
  const prPoints = p.map((v, i) => ({ x: r[i], y: v }));

  const histogramLabels = data.score_histogram.bins.slice(0, -1).map((v, i) => `${num(v, 2)}-${num(data.score_histogram.bins[i + 1], 2)}`);

  const driftLabels = data.drift_windows.map((x) => `W${x.window}`);

  const toReadableTime = (isoString) => {
    if (!isoString) return "N/A";
    const date = new Date(isoString);
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, "0");
    const day = String(date.getDate()).padStart(2, "0");
    const hours = String(date.getHours()).padStart(2, "0");
    const minutes = String(date.getMinutes()).padStart(2, "0");
    return `${year}-${month}-${day} ${hours}:${minutes}`;
  }

  const metricItems = [
    ["Accuracy", data.metrics.accuracy],
    ["Precision", data.metrics.precision],
    ["Recall", data.metrics.recall],
    ["F1", data.metrics.f1],
    ["Average Precision", data.metrics.average_precision],
    ["Last Updated", toReadableTime(data.meta.generated_at)],
  ];

  const renderShap = (name, list, color) => {
    const items = (list || []).slice(0, SHAP_MAX_FEATURES);
    const chartHeight = Math.max(260, items.length * 26);

    return (
      <div>
        <h3>{name}</h3>
        <div className="shapChartWrap" style={{ height: `${chartHeight}px` }}>
          <BarChart
            data={{
              labels: items.map((x) => x.feature),
              datasets: [{ label: "mean", data: items.map((x) => x.mean_abs_shap), backgroundColor: color }],
            }}
            options={{
              indexAxis: "y",
              responsive: true,
              maintainAspectRatio: false,
              plugins: { legend: { display: false } },
              scales: {
                y: {
                  ticks: {
                    autoSkip: false,
                    font: { size: 11 },
                  },
                },
              },
            }}
          />
        </div>
      </div>
    );
  };

  return (
    <>
      <div className="pageHeaderBar">
        <h2 className="pageTitle">Overview</h2>
        <Button variant="contained" color="success" startIcon={<FlaskConical size={16} />} onClick={openRetrainDialog}>
          Retrain
        </Button>
      </div>

      {taskError ? (
        <Alert severity="error" sx={{ mb: 1.5 }}>
          {taskError}
        </Alert>
      ) : null}

      <div className="grid">
        <section className="card">
          <h3>Key Metrics</h3>
          <div className="kpis">
            {metricItems.map(([name, value]) => (
              <div className="kpi" key={name}>
                <div className="name">{name}</div>
                <div className="value">{typeof value === "string" ? value : num(value)}</div>
              </div>
            ))}
          </div>
        </section>

        <section className="card">
          <h3>Primary PR Curve</h3>
          <LineChart
            data={{
              datasets: [{ label: "PR", data: prPoints, parsing: false, borderColor: "#2563eb", pointRadius: 0 }],
            }}
            options={{
              scales: { x: { type: "linear", min: 0, max: 1 }, y: { min: 0, max: 1 } },
            }}
          />
        </section>

        <section className="card">
          <h3>Score Distribution</h3>
          <BarChart
            data={{
              labels: histogramLabels,
              datasets: [{ label: "Count", data: data.score_histogram.counts, backgroundColor: "#7c3aed" }],
            }}
            options={{ plugins: { legend: { display: false } } }}
          />
        </section>

        <section className="card">
          <h3>Drift Windows</h3>
          <LineChart
            data={{
              labels: driftLabels,
              datasets: [
                { label: "Mean Score", data: data.drift_windows.map((x) => x.mean_score), borderColor: "#16a34a" },
                { label: "Positive Ratio", data: data.drift_windows.map((x) => x.positive_ratio), borderColor: "#dc2626" },
              ],
            }}
            options={{ scales: { y: { min: 0, max: 1 } } }}
          />
        </section>

        <section className="card wide">
          <h3>SHAP Top Features by Model</h3>
          <div className="shapGrid">
            {renderShap("Autoencoder", data.shap_by_model.Autoencoder, "#7c3aed")}
            {renderShap("LSTM", data.shap_by_model.LSTM, "#0ea5e9")}
            {renderShap("DSFANet", data.shap_by_model.DSFANet, "#16a34a")}
          </div>
        </section>
      </div>

      <Dialog open={retrainOpen} onClose={() => setRetrainOpen(false)} fullWidth maxWidth="sm">
        <form onSubmit={submitRetrain}>
          <DialogTitle>Run Retrain Experiment</DialogTitle>
          <DialogContent>
            <Stack spacing={1.5} sx={{ mt: 0.5 }}>
              <div className="retrainPresetRow">
                {RETRAIN_PRESETS.map((preset) => (
                  <Button key={preset.key} size="small" variant="outlined" startIcon={<Zap size={14} />} onClick={() => applyPreset(preset.values)}>
                    {preset.label}
                  </Button>
                ))}
              </div>
              <TextField label="Run ID" value={retrainForm.run_id} onChange={(e) => onChangeForm("run_id", e.target.value)} size="small" required />
              <TextField label="Base Dataset" value={retrainForm.base_dataset} onChange={(e) => onChangeForm("base_dataset", e.target.value)} size="small" required />
              <TextField label="Steps" helperText="Comma-separated, default: 3,8" value={retrainForm.steps} onChange={(e) => onChangeForm("steps", e.target.value)} size="small" />
              <TextField label="Epochs" helperText="Comma-separated, default: 10,10,20" value={retrainForm.epochs} onChange={(e) => onChangeForm("epochs", e.target.value)} size="small" />
              <TextField
                label="Size Limit"
                type="number"
                value={retrainForm.size_limit}
                onChange={(e) => onChangeForm("size_limit", e.target.value)}
                helperText="0 means full dataset"
                size="small"
              />
              <TextField label="OOD Dataset" value={retrainForm.ood_dataset} onChange={(e) => onChangeForm("ood_dataset", e.target.value)} size="small" />
              <TextField label="Device" value={retrainForm.device} onChange={(e) => onChangeForm("device", e.target.value)} size="small" />
            </Stack>
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setRetrainOpen(false)}>Cancel</Button>
            <Button type="submit" variant="contained" startIcon={<PlayIcon size={16} />}>
              Start
            </Button>
          </DialogActions>
        </form>
      </Dialog>

      {task && task.status !== "idle" ? (
        <div className={`retrainToast retrainToast--${task.status}`} role="status" aria-live="polite">
          <div className="retrainToastHeader">
            <div className="retrainToastTitle">
              {task.status === "succeeded" ? (
                <CheckCircle2 size={16} />
              ) : task.status === "failed" ? (
                <AlertCircle size={16} />
              ) : (
                <LoaderCircle size={16} className="retrainSpin" />
              )}
              <span>Retrain Task</span>
            </div>
            <Chip size="small" color={statusChipColor} label={String(task.status || "unknown").toUpperCase()} />
          </div>

          <div className="retrainToastMessage">{task.message || "Processing retrain request..."}</div>

          {task?.params?.run_id ? <div className="retrainToastMeta">Run ID: {task.params.run_id}</div> : null}
          {task?.params?.base_dataset ? <div className="retrainToastMeta">Dataset: {task.params.base_dataset}</div> : null}

          {latestLogLine ? <div className="retrainToastLog">{latestLogLine}</div> : null}

          {allLogLines.length > 0 ? (
            <Button
              size="small"
              variant="text"
              onClick={() => setLogsExpanded((prev) => !prev)}
              endIcon={logsExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
              sx={{ mt: 0.5, minWidth: 0, px: 0 }}
            >
              {logsExpanded ? "Hide Logs" : `Show Logs (${allLogLines.length})`}
            </Button>
          ) : null}

          {logsExpanded && allLogLines.length > 0 ? (
            <div className="retrainToastLogPanel">
              {allLogLines.map((line, idx) => (
                <div className="retrainToastLogLine" key={`${idx}-${line.slice(0, 24)}`}>
                  {line}
                </div>
              ))}
            </div>
          ) : null}

          {task.status === "succeeded" && refreshCountdown !== null ? <div className="retrainToastRefresh">Finished. Refreshing in {refreshCountdown}s...</div> : null}
        </div>
      ) : null}
    </>
  );
}
