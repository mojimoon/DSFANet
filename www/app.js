const state = {
  data: null,
  alerts: [],
  models: {},
  charts: {},
};

function num(v, d = 4) {
  return Number(v).toFixed(d);
}

function destroyChart(key) {
  if (state.charts[key]) {
    state.charts[key].destroy();
    state.charts[key] = null;
  }
}

function renderMeta(meta) {
  const el = document.getElementById("metaInfo");
  el.textContent = `dataset=${meta.dataset} | primary=${meta.primary_model} | train=${meta.n_train} | test=${meta.n_test} | device=${meta.device}`;
}

function renderKpis(metrics, confusion) {
  const kpis = document.getElementById("kpis");
  const items = [
    ["Accuracy", metrics.accuracy],
    ["Precision", metrics.precision],
    ["Recall", metrics.recall],
    ["F1", metrics.f1],
    ["AP", metrics.average_precision],
    ["TP/FP/FN", `${confusion.tp}/${confusion.fp}/${confusion.fn}`],
  ];

  kpis.innerHTML = items
    .map(([name, value]) => {
      const shown = typeof value === "number" ? num(value) : value;
      return `<div class="kpi"><div class="name">${name}</div><div class="value">${shown}</div></div>`;
    })
    .join("");
}

function renderPr(pr) {
  destroyChart("pr");
  const points = pr.precision.map((p, i) => ({ x: pr.recall[i], y: p }));
  state.charts.pr = new Chart(document.getElementById("prChart"), {
    type: "line",
    data: {
      datasets: [
        {
          label: "PR Curve",
          data: points,
          borderColor: "#2563eb",
          parsing: false,
          pointRadius: 0,
          tension: 0.15,
        },
      ],
    },
    options: {
      responsive: true,
      scales: {
        x: { type: "linear", min: 0, max: 1, title: { display: true, text: "Recall" } },
        y: { min: 0, max: 1, title: { display: true, text: "Precision" } },
      },
    },
  });
}

function renderHist(hist) {
  const labels = hist.bins.slice(0, -1).map((x, i) => `${num(x, 2)}-${num(hist.bins[i + 1], 2)}`);
  destroyChart("hist");
  state.charts.hist = new Chart(document.getElementById("histChart"), {
    type: "bar",
    data: {
      labels,
      datasets: [{ label: "Count", data: hist.counts, backgroundColor: "#7c3aed" }],
    },
    options: { responsive: true, plugins: { legend: { display: false } } },
  });
}

function renderDrift(rows) {
  destroyChart("drift");
  state.charts.drift = new Chart(document.getElementById("driftChart"), {
    type: "line",
    data: {
      labels: rows.map((r) => `W${r.window}`),
      datasets: [
        {
          label: "Mean Score",
          data: rows.map((r) => r.mean_score),
          borderColor: "#059669",
          tension: 0.2,
        },
        {
          label: "Positive Ratio",
          data: rows.map((r) => r.positive_ratio),
          borderColor: "#dc2626",
          tension: 0.2,
        },
      ],
    },
    options: {
      responsive: true,
      scales: { y: { min: 0, max: 1 } },
    },
  });
}

function renderShap(features) {
  const top = features.slice(0, 15).reverse();
  destroyChart("shap");
  state.charts.shap = new Chart(document.getElementById("shapChart"), {
    type: "bar",
    data: {
      labels: top.map((x) => x.feature),
      datasets: [{ label: "mean |SHAP|", data: top.map((x) => x.mean_abs_shap), backgroundColor: "#0ea5e9" }],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      plugins: { legend: { display: false } },
    },
  });
}

function renderClassChart(overview) {
  const d = overview.class_distribution;
  destroyChart("class");
  state.charts.class = new Chart(document.getElementById("classChart"), {
    type: "bar",
    data: {
      labels: ["Benign", "Malicious"],
      datasets: [
        { label: "Train", data: [d.train.benign, d.train.malicious], backgroundColor: "#2563eb" },
        { label: "Test", data: [d.test.benign, d.test.malicious], backgroundColor: "#7c3aed" },
      ],
    },
    options: { responsive: true },
  });
}

function renderFeatureStatsTable(overview) {
  const rows = [
    ...overview.feature_stats.static_top_variance.map((r) => ({ ...r, group: "static" })),
    ...overview.feature_stats.temporal_top_variance.map((r) => ({ ...r, group: "temporal" })),
  ];
  const cols = ["group", "feature", "mean", "std", "min", "max"];
  const table = document.getElementById("featureStatsTable");
  table.querySelector("thead").innerHTML = `<tr>${cols.map((c) => `<th>${c}</th>`).join("")}</tr>`;
  table.querySelector("tbody").innerHTML = rows
    .map((r) => `<tr>${cols.map((c) => `<td>${typeof r[c] === "number" ? num(r[c], 4) : r[c]}</td>`).join("")}</tr>`)
    .join("");
}

function renderBenchmarkViews(rows) {
  destroyChart("benchmark");
  state.charts.benchmark = new Chart(document.getElementById("benchmarkChart"), {
    type: "bar",
    data: {
      labels: rows.map((r) => r.model),
      datasets: [{ label: "Average Precision", data: rows.map((r) => r.average_precision), backgroundColor: "#0ea5e9" }],
    },
    options: { responsive: true, plugins: { legend: { display: false } } },
  });

  const cols = ["model", "accuracy", "precision", "recall", "f1", "average_precision"];
  const table = document.getElementById("benchmarkTable");
  table.querySelector("thead").innerHTML = `<tr>${cols.map((c) => `<th>${c}</th>`).join("")}</tr>`;
  table.querySelector("tbody").innerHTML = rows
    .map((r) => `<tr>${cols.map((c) => `<td>${typeof r[c] === "number" ? num(r[c], 4) : r[c]}</td>`).join("")}</tr>`)
    .join("");
}

function renderAttackViews(rows) {
  const labels = [...new Set(rows.map((r) => r.attack))];
  const models = [...new Set(rows.map((r) => r.model))];
  const datasets = models.map((m, idx) => ({
    label: m,
    data: labels.map((atk) => {
      const found = rows.find((r) => r.attack === atk && r.model === m);
      return found ? found.average_precision : 0;
    }),
    backgroundColor: idx === 0 ? "#ef4444" : "#22c55e",
  }));

  destroyChart("attack");
  state.charts.attack = new Chart(document.getElementById("attackChart"), {
    type: "bar",
    data: { labels, datasets },
    options: { responsive: true },
  });

  const cols = ["attack", "model", "accuracy", "recall", "f1", "average_precision"];
  const table = document.getElementById("attackTable");
  table.querySelector("thead").innerHTML = `<tr>${cols.map((c) => `<th>${c}</th>`).join("")}</tr>`;
  table.querySelector("tbody").innerHTML = rows
    .map((r) => `<tr>${cols.map((c) => `<td>${typeof r[c] === "number" ? num(r[c], 4) : r[c]}</td>`).join("")}</tr>`)
    .join("");
}

function renderAlertsTable(rows) {
  const table = document.getElementById("alertsTable");
  const thead = table.querySelector("thead");
  const tbody = table.querySelector("tbody");

  if (!rows.length) {
    thead.innerHTML = "";
    tbody.innerHTML = "";
    return;
  }

  const cols = Object.keys(rows[0]).slice(0, 8);
  thead.innerHTML = `<tr>${cols.map((c) => `<th>${c}</th>`).join("")}</tr>`;

  tbody.innerHTML = rows
    .map((row) => `<tr>${cols.map((c) => `<td>${typeof row[c] === "number" ? num(row[c], 3) : row[c]}</td>`).join("")}</tr>`)
    .join("");
}

function renderModelDetail(name) {
  const detail = state.models[name];
  if (!detail) {
    return;
  }

  const m = detail.metrics;
  const metricsEl = document.getElementById("modelMetrics");
  metricsEl.innerHTML = [
    ["Accuracy", m.accuracy],
    ["Precision", m.precision],
    ["Recall", m.recall],
    ["F1", m.f1],
    ["AP", m.average_precision],
  ]
    .map(([k, v]) => `<div class="kpi"><div class="name">${k}</div><div class="value">${num(v, 4)}</div></div>`)
    .join("");

  destroyChart("modelPr");
  const points = detail.pr_curve.precision.map((p, i) => ({ x: detail.pr_curve.recall[i], y: p }));
  state.charts.modelPr = new Chart(document.getElementById("modelPrChart"), {
    type: "line",
    data: {
      datasets: [{ label: `${name} PR`, data: points, parsing: false, borderColor: "#2563eb", pointRadius: 0 }],
    },
    options: {
      responsive: true,
      scales: {
        x: { type: "linear", min: 0, max: 1, title: { display: true, text: "Recall" } },
        y: { min: 0, max: 1, title: { display: true, text: "Precision" } },
      },
    },
  });

  const featRows = detail.top_features || [];
  const cols = featRows.length && featRows[0].importance !== undefined ? ["feature", "importance"] : ["feature", "mean_abs_shap"];
  const table = document.getElementById("modelFeatureTable");
  table.querySelector("thead").innerHTML = `<tr>${cols.map((c) => `<th>${c}</th>`).join("")}</tr>`;
  table.querySelector("tbody").innerHTML = featRows
    .map((r) => `<tr>${cols.map((c) => `<td>${typeof r[c] === "number" ? num(r[c], 6) : r[c]}</td>`).join("")}</tr>`)
    .join("");
}

async function renderSampleDetail(sampleId) {
  const resp = await fetch(`/api/sample/${sampleId}`);
  const detail = await resp.json();
  if (detail.error) {
    return;
  }

  const entries = Object.entries(detail.model_scores || {});
  destroyChart("sampleScore");
  state.charts.sampleScore = new Chart(document.getElementById("sampleScoreChart"), {
    type: "bar",
    data: {
      labels: entries.map(([k]) => k),
      datasets: [{ label: "Score", data: entries.map(([, v]) => v), backgroundColor: "#7c3aed" }],
    },
    options: { responsive: true, scales: { y: { min: 0, max: 1 } } },
  });

  const sum = document.getElementById("sampleSummary");
  sum.innerHTML = `
    <div>sample_id: ${detail.sample_id}</div>
    <div>label: ${detail.label}</div>
  `;

  const featRows = [
    ...(detail.top_static_features || []).map((x) => ({ group: "static", ...x })),
    ...(detail.top_temporal_features || []).map((x) => ({ group: "temporal", ...x })),
  ];
  const cols = ["group", "feature", "value"];
  const table = document.getElementById("sampleFeatureTable");
  table.querySelector("thead").innerHTML = `<tr>${cols.map((c) => `<th>${c}</th>`).join("")}</tr>`;
  table.querySelector("tbody").innerHTML = featRows
    .map((r) => `<tr>${cols.map((c) => `<td>${typeof r[c] === "number" ? num(r[c], 4) : r[c]}</td>`).join("")}</tr>`)
    .join("");
}

function setupThreshold() {
  const slider = document.getElementById("thresholdSlider");
  const valueEl = document.getElementById("thresholdValue");
  const allRows = state.alerts;

  const update = () => {
    const th = Number(slider.value);
    valueEl.textContent = th.toFixed(2);
    const filtered = allRows.filter((r) => Number(r.voting_score) >= th).slice(0, 200);
    renderAlertsTable(filtered);
  };

  slider.addEventListener("input", update);
  update();
}

function setupNav() {
  const btns = document.querySelectorAll(".navBtn");
  btns.forEach((btn) => {
    btn.addEventListener("click", () => {
      btns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const view = btn.dataset.view;
      document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
      document.getElementById(`view-${view}`).classList.add("active");
    });
  });
}

function setupModelSelector() {
  const sel = document.getElementById("modelSelect");
  const names = Object.keys(state.models);
  sel.innerHTML = names.map((n) => `<option value="${n}">${n}</option>`).join("");
  sel.addEventListener("change", () => renderModelDetail(sel.value));
  if (names.length) {
    renderModelDetail(names[0]);
  }
}

function setupSampleSelector() {
  const sel = document.getElementById("sampleSelect");
  const ids = state.data.sample_ids || [];
  sel.innerHTML = ids.slice(0, 200).map((id) => `<option value="${id}">${id}</option>`).join("");
  sel.addEventListener("change", () => renderSampleDetail(sel.value));
  if (ids.length) {
    renderSampleDetail(ids[0]);
  }
}

async function main() {
  const [dataResp, alertsResp, modelsResp] = await Promise.all([
    fetch("/api/dashboard"),
    fetch("/api/alerts"),
    fetch("/api/models"),
  ]);

  state.data = await dataResp.json();
  state.alerts = await alertsResp.json();
  state.models = await modelsResp.json();

  renderMeta(state.data.meta);
  renderKpis(state.data.metrics, state.data.confusion);
  renderPr(state.data.pr_curve);
  renderHist(state.data.score_histogram);
  renderDrift(state.data.drift_windows);
  renderShap(state.data.shap_top_features || []);
  renderClassChart(state.data.dataset_overview);
  renderFeatureStatsTable(state.data.dataset_overview);
  renderBenchmarkViews(state.data.benchmark_models || []);
  renderAttackViews(state.data.attack_results || []);

  setupNav();
  setupModelSelector();
  setupSampleSelector();
  setupThreshold();
}

main().catch((e) => {
  console.error(e);
  alert("Dashboard 加载失败，请先运行 web_main.py 生成并启动后端。\n" + e.message);
});
