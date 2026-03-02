# Ensemble IDS Dashboard (Python API + Next.js Frontend)

This project provides:
- A Python backend pipeline to generate offline model artifacts and expose API endpoints.
- A Next.js dashboard frontend with multi-page navigation and dynamic routes.

## 1) Prerequisites

- Python 3.13+
- `pip`
- `pnpm` (v10+ recommended)
- Windows PowerShell (for the one-click launcher)

## 2) Install Python Dependencies

From the project root:

```bash
python -m pip install -U pip
python -m pip install numpy pandas scikit-learn torch shap flask flask-cors
```

## 3) Install Frontend Dependencies

```bash
cd www
pnpm install
cd ..
```

## 4) One-Click Start (Recommended)

Double-click `start_dashboard.cmd` or run:

```bash
powershell -ExecutionPolicy Bypass -File .\start_dashboard.ps1
```

What it does:
1. Checks Python and pnpm.
2. Generates dashboard artifacts if `out/www/dashboard_data.json` does not exist.
3. Starts the Python API server on `http://127.0.0.1:8000`.
4. Starts Next.js frontend on `http://127.0.0.1:3000`.
5. Opens the browser automatically.

Optional flags:

```bash
powershell -ExecutionPolicy Bypass -File .\start_dashboard.ps1 -ForceRebuild
powershell -ExecutionPolicy Bypass -File .\start_dashboard.ps1 -Device cpu -ApiPort 8000 -WebPort 3000
```

## 5) Manual Start (Alternative)

### Terminal A: Generate artifacts

```bash
python web_main.py --skip-serve --device cpu
```

### Terminal B: Start API only

```bash
python web_main.py --serve-only --host 127.0.0.1 --port 8000
```

### Terminal C: Start Next.js frontend

```bash
cd www
set NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
pnpm dev --port 3000
```

Then open `http://127.0.0.1:3000`.

## 6) API Endpoints

- `GET /api/dashboard`
- `GET /api/alerts`
- `GET /api/models`
- `GET /api/model/<name>`
- `GET /api/sample/<sample_id>`

## 7) Frontend Routes

- `/` (Overview)
- `/dataset`
- `/benchmarks`
- `/attacks`
- `/models`
- `/model/[modelId]`
- `/instances`
- `/instance/[instanceId]`
- `/experiments`

## 8) Main Experiment Entry

Use the new report-oriented entry script:

```bash
python experiments_main.py --device cpu --steps 1,2,3,4,5,6 --base-dataset NF-UNSW-NB15-v3.csv --include-xgboost
```

Useful options:

```bash
python experiments_main.py --run-id exp_a1 --steps 1
python experiments_main.py --run-id exp_a1 --steps 2,3 --base-dataset NF-UNSW-NB15-v3.csv --retrain-metrics random,uncertainty,entropy,gd,ensemble_rank,ensemble_hybrid --retrain-budgets 0.1,0.2,0.3 --retrain-id-ratios 0.25,0.5,0.75
python experiments_main.py --run-id exp_a1 --steps 4,5,6
```

Key argument naming:

- `--base-dataset`: the primary dataset used in drift/retraining/ablation steps.
- `--datasets`: datasets used in benchmark step (step 1).

Unified selection interface for retraining:

- `random`
- `uncertainty`
- `entropy`
- `gd`
- `ensemble_rank`
- `ensemble_hybrid`

Output layout:

- `out/experiments/<run_id>/summary_step1_benchmark_<run_id>.csv`
- `out/experiments/<run_id>/summary_step2_drift_<run_id>.csv`
- `out/experiments/<run_id>/summary_step3_retrain_<run_id>.csv`
- `out/experiments/<run_id>/summary_step4_best_ensemble_shap_<run_id>.csv`
- `out/experiments/<run_id>/summary_step5_dsfanet_ablation_<run_id>.csv`

Every output filename includes the `run_id`, and prediction/model artifacts include dataset/model tags for easier resume and debugging.

## 9) Troubleshooting

### A) Frontend cannot fetch API

- Make sure the Python API is running on `127.0.0.1:8000`.
- Make sure `NEXT_PUBLIC_API_BASE_URL` matches the API address.
- Restart Next.js after changing environment variables.

### B) First startup is slow

The backend may retrain models and run SHAP analysis. This is expected for artifact generation.

Use one-click launcher without `-ForceRebuild` to reuse existing artifacts.

### C) SHAP warning for LSTM module

You may see warnings like `unrecognized nn.Module: LSTM`. Current implementation still produces SHAP outputs and exports files.

### D) Port conflict

If ports are occupied, use custom ports:

```bash
powershell -ExecutionPolicy Bypass -File .\start_dashboard.ps1 -ApiPort 8010 -WebPort 3010
```
