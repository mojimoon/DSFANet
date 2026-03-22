"use client";

import { useEffect, useMemo, useState } from "react";
import { Alert, Button, Chip, Dialog, DialogActions, DialogContent, DialogTitle, IconButton, Stack, TextField } from "@mui/material";
import { AlertCircle, CheckCircle2, ChevronDown, ChevronUp, FlaskConical, LoaderCircle, PlayIcon, Zap } from "lucide-react";
import { fetchApi, postApi } from "@/lib/api";

const DEFAULT_DATASET = "NF-UNSW-NB15-v3.csv";
const REFRESHED_TASK_KEY = "ids:retrainRefreshedTaskSignature";
const TOAST_MINIMIZED_KEY = "ids:retrainToastMinimized";

const DEFAULT_RETRAIN_FORM = {
  run_id: "unsw-main",
  base_dataset: DEFAULT_DATASET,
  steps: "3,8",
  epochs: "10,10,20",
  retrain_metrics: "random,uncertainty,entropy,gd,ensemble_rank,ensemble_p_value,ensemble_hybrid",
  retrain_budgets: "0.05,0.1,0.2,0.3",
  retrain_id_ratios: "0.1,0.3,0.5,0.7,0.9",
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
      retrain_metrics: "random,uncertainty,entropy,gd,ensemble_rank,ensemble_p_value,ensemble_hybrid",
      retrain_budgets: "0.05,0.1,0.2,0.3",
      retrain_id_ratios: "0.1,0.3,0.5,0.7,0.9",
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
      retrain_metrics: "random,entropy,ensemble_rank",
      retrain_budgets: "0.1,0.3",
      retrain_id_ratios: "0.3,0.7",
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
      retrain_metrics: "random,uncertainty,entropy,gd,ensemble_rank,ensemble_p_value,ensemble_hybrid",
      retrain_budgets: "0.05,0.1,0.2,0.3",
      retrain_id_ratios: "0.1,0.3,0.5,0.7,0.9",
      size_limit: 0,
      ood_dataset: "NF-BoT-IoT-v3.csv",
      device: "cuda",
    },
  },
];

export default function SidebarRetrainControl() {
  const [retrainOpen, setRetrainOpen] = useState(false);
  const [retrainForm, setRetrainForm] = useState(DEFAULT_RETRAIN_FORM);
  const [task, setTask] = useState(null);
  const [taskError, setTaskError] = useState("");
  const [refreshCountdown, setRefreshCountdown] = useState(null);
  const [logsExpanded, setLogsExpanded] = useState(false);
  const [toastMinimized, setToastMinimized] = useState(false);

  useEffect(() => {
    fetchApi("/api/retrain/status")
      .then((payload) => {
        if (payload?.status && payload.status !== "idle") {
          setTask(payload);
        }
      })
      .catch(() => null);
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const saved = window.localStorage.getItem(TOAST_MINIMIZED_KEY);
    if (saved === "1") {
      setToastMinimized(true);
    }
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    window.localStorage.setItem(TOAST_MINIMIZED_KEY, toastMinimized ? "1" : "0");
  }, [toastMinimized]);

  const taskRefreshSignature = useMemo(() => {
    if (!task) {
      return "";
    }
    const taskId = String(task.task_id || "");
    const startedAt = String(task.started_at || "");
    const finishedAt = String(task.finished_at || "");
    return [taskId, startedAt, finishedAt].filter(Boolean).join("|");
  }, [task]);

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

    const refreshedTaskSignature = typeof window !== "undefined" ? window.sessionStorage.getItem(REFRESHED_TASK_KEY) || "" : "";
    if (taskRefreshSignature && taskRefreshSignature === refreshedTaskSignature) {
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
          if (taskRefreshSignature && typeof window !== "undefined") {
            window.sessionStorage.setItem(REFRESHED_TASK_KEY, taskRefreshSignature);
          }
          window.location.reload();
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => window.clearInterval(timer);
  }, [task?.status, taskRefreshSignature]);

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

  const hasLogs = allLogLines.length > 0;

  const openRetrainDialog = async () => {
    let currentDataset = DEFAULT_DATASET;
    let currentRunId = "unsw-main";
    try {
      const dashboard = await fetchApi("/api/dashboard");
      currentDataset = String(dashboard?.meta?.dataset || DEFAULT_DATASET);
      currentRunId = String(dashboard?.meta?.run_id || "unsw-main");
    } catch (_err) {
      // Keep defaults when dashboard API is unavailable.
    }

    setRetrainForm({
      ...DEFAULT_RETRAIN_FORM,
      run_id: currentRunId,
      base_dataset: currentDataset,
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
        if (typeof window !== "undefined") {
          window.sessionStorage.removeItem(REFRESHED_TASK_KEY);
        }
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

  return (
    <div className="sidebarRetrainArea">
      <Button className="sidebarRetrainBtn" variant="contained" color="success" startIcon={<FlaskConical size={16} />} onClick={openRetrainDialog} fullWidth>
        Retrain
      </Button>

      {taskError ? <div className="sidebarRetrainError">{taskError}</div> : null}

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
              <TextField label="Steps" helperText="default: 1,2,3,4,5,6,7,8" value={retrainForm.steps} onChange={(e) => onChangeForm("steps", e.target.value)} size="small" />
              <TextField label="Epochs" helperText="default: 10,10,20" value={retrainForm.epochs} onChange={(e) => onChangeForm("epochs", e.target.value)} size="small" />
              <TextField
                label="Retrain Metrics"
                helperText="random,uncertainty,entropy,gd,ensemble_rank,ensemble_p_value,ensemble_hybrid"
                value={retrainForm.retrain_metrics}
                onChange={(e) => onChangeForm("retrain_metrics", e.target.value)}
                size="small"
              />
              <TextField
                label="Retrain Budgets"
                helperText="default: 0.05,0.1,0.2,0.3"
                value={retrainForm.retrain_budgets}
                onChange={(e) => onChangeForm("retrain_budgets", e.target.value)}
                size="small"
              />
              <TextField
                label="Retrain ID Ratios"
                helperText="default: 0.1,0.3,0.5,0.7,0.9"
                value={retrainForm.retrain_id_ratios}
                onChange={(e) => onChangeForm("retrain_id_ratios", e.target.value)}
                size="small"
              />
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
        <div className={`retrainToast retrainToast--${task.status} ${toastMinimized ? "retrainToast--min" : ""}`} role="status" aria-live="polite">
          <div className="retrainToastHeader">
            <div className="retrainToastTitle">
              {task.status === "succeeded" ? <CheckCircle2 size={16} /> : task.status === "failed" ? <AlertCircle size={16} /> : <LoaderCircle size={16} className="retrainSpin" />}
              <span>Retrain</span>
            </div>
            <IconButton
              size="small"
              className="retrainToggleBtn"
              onClick={() => setToastMinimized((prev) => !prev)}
              aria-label={toastMinimized ? "Expand retrain panel" : "Collapse retrain panel"}
            >
              {toastMinimized ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
            </IconButton>
          </div>

          {!toastMinimized ? (
            <div className="retrainToastStatusLine">
              <Chip size="small" color={statusChipColor} label={String(task.status || "unknown")} />
            </div>
          ) : null}

          {!toastMinimized ? <div className="retrainToastMessage">{task.message || "Processing retrain request..."}</div> : null}
          {!toastMinimized && task?.params?.run_id ? <div className="retrainToastMeta">Run ID: {task.params.run_id}</div> : null}
          {!toastMinimized && task?.params?.base_dataset ? <div className="retrainToastMeta">Dataset: {task.params.base_dataset}</div> : null}
          {!toastMinimized && latestLogLine ? <div className="retrainToastLog">{latestLogLine}</div> : null}

          {!toastMinimized ? (
            <Button
              size="small"
              variant="text"
              onClick={() => setLogsExpanded((prev) => !prev)}
              endIcon={logsExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
              sx={{ mt: 0.5, minWidth: 0, px: 0 }}
            >
              {logsExpanded ? "Hide Logs" : hasLogs ? `Show Logs (${allLogLines.length})` : "Show Logs (waiting...)"}
            </Button>
          ) : null}

          {!toastMinimized && logsExpanded ? (
            <div className="retrainToastLogPanel">
              {hasLogs ? (
                allLogLines.map((line, idx) => (
                  <div className="retrainToastLogLine" key={`${idx}-${line.slice(0, 24)}`}>
                    {line}
                  </div>
                ))
              ) : (
                <div className="retrainToastLogEmpty">No logs yet. The process may still be initializing.</div>
              )}
            </div>
          ) : null}

          {!toastMinimized && task.status === "succeeded" && refreshCountdown !== null ? <div className="retrainToastRefresh">Finished. Refreshing in {refreshCountdown}s...</div> : null}
        </div>
      ) : null}
    </div>
  );
}
