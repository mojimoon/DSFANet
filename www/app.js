const state = {
  data: null,
  alerts: [],
};

function num(v, d = 4) {
  return Number(v).toFixed(d);
}

function renderMeta(meta) {
  const el = document.getElementById("metaInfo");
  el.textContent = `dataset=${meta.dataset} | model=${meta.model} | train=${meta.n_train} | test=${meta.n_test} | device=${meta.device}`;
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
  const points = pr.precision.map((p, i) => ({ x: pr.recall[i], y: p }));
  new Chart(document.getElementById("prChart"), {
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
  new Chart(document.getElementById("histChart"), {
    type: "bar",
    data: {
      labels,
      datasets: [{ label: "Count", data: hist.counts, backgroundColor: "#7c3aed" }],
    },
    options: { responsive: true, plugins: { legend: { display: false } } },
  });
}

function renderDrift(rows) {
  new Chart(document.getElementById("driftChart"), {
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
  new Chart(document.getElementById("shapChart"), {
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

function setupThreshold() {
  const slider = document.getElementById("thresholdSlider");
  const valueEl = document.getElementById("thresholdValue");
  const allRows = state.alerts;

  const update = () => {
    const th = Number(slider.value);
    valueEl.textContent = th.toFixed(2);
    const filtered = allRows.filter((r) => Number(r.score) >= th).slice(0, 200);
    renderAlertsTable(filtered);
  };

  slider.addEventListener("input", update);
  update();
}

async function main() {
  const [dataResp, alertsResp] = await Promise.all([
    fetch("/api/dashboard"),
    fetch("/api/alerts"),
  ]);

  state.data = await dataResp.json();
  state.alerts = await alertsResp.json();

  renderMeta(state.data.meta);
  renderKpis(state.data.metrics, state.data.confusion);
  renderPr(state.data.pr_curve);
  renderHist(state.data.score_histogram);
  renderDrift(state.data.drift_windows);
  renderShap(state.data.shap_top_features || []);
  setupThreshold();
}

main().catch((e) => {
  console.error(e);
  alert("Dashboard 加载失败，请先运行 web_main.py 生成并启动后端。\n" + e.message);
});
