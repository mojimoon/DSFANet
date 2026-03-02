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

## 8) Troubleshooting

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
