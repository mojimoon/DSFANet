# Operations Log - 2026-03-02

This document records implementation changes made today for environment validation and troubleshooting.

## Scope Completed Today

1. Added SHAP analysis support for three models:
   - Autoencoder (AE)
   - LSTM
   - DSFANet
2. Expanded backend artifact generation for dashboard use:
   - Dataset overview
   - Benchmark comparison (single models + ensembles)
   - Attack comparison (clean/FGSM/PGD)
   - Model detail payload
   - Instance detail payload
3. Migrated frontend from a single static page to Next.js multi-page app with dynamic routes.
4. Added one-click startup scripts and updated docs.

## Backend Changes

### File: `src/shap_analysis.py`

- Added `analyze_dsfanet_shap(...)`.
- Added fallback to `GradientExplainer` when `DeepExplainer` is incompatible.
- Ensured export files are generated for all model SHAP outputs.

### File: `web_main.py`

- Extended dashboard payload to include:
  - `shap_by_model` for AE/LSTM/DSFANet
  - benchmark tables
  - attack result tables
  - model details
  - instance details
- Added API-only mode:
  - `--serve-only` now starts API without recomputing artifacts.
- Switched serving style to API-first and enabled CORS for `/api/*`.

### File: `pyproject.toml`

- Added `flask-cors` dependency.

## Frontend Migration (Next.js)

### Replaced static frontend

Removed old files:
- `www/index.html`
- `www/app.js`
- `www/style.css`

### Added Next.js app structure

- `www/package.json`
- `www/next.config.mjs`
- `www/jsconfig.json`
- `www/app/layout.js`
- `www/app/globals.css`
- `www/lib/api.js`
- `www/components/NavMenu.jsx`
- `www/components/charts.jsx`

### Added pages

- `/` -> `www/app/page.jsx`
- `/dataset` -> `www/app/dataset/page.jsx`
- `/benchmarks` -> `www/app/benchmarks/page.jsx`
- `/attacks` -> `www/app/attacks/page.jsx`
- `/models` -> `www/app/models/page.jsx`
- `/model/[modelId]` -> `www/app/model/[modelId]/page.jsx`
- `/instances` -> `www/app/instances/page.jsx`
- `/instance/[instanceId]` -> `www/app/instance/[instanceId]/page.jsx`

## Startup Scripts

### `start_dashboard.ps1`

- Validates Python and pnpm.
- Rebuilds artifacts if missing (or with `-ForceRebuild`).
- Starts Python API in a new terminal.
- Starts Next.js dev server in a new terminal.
- Opens browser automatically.

### `start_dashboard.cmd`

- Windows launcher wrapper for one-click startup.

## Validation Performed

1. Python pipeline run:
   - Command: `python web_main.py --skip-serve --device cpu`
   - Result: success, artifacts generated in `out/www/`.
2. Next.js dependency install and build:
   - Commands: `pnpm install`, `pnpm build` in `www/`
   - Result: success, all pages built including dynamic routes.

## Known Warnings

- SHAP may emit LSTM warnings such as:
  - `unrecognized nn.Module: LSTM`
- Current implementation still exports usable SHAP outputs.

## Quick Debug Checklist

1. If UI cannot load data:
   - Verify API is reachable at `http://127.0.0.1:8000/api/dashboard`.
2. If API fails to start:
   - Ensure Python dependencies are installed, especially `flask-cors`.
3. If frontend fails to connect:
   - Verify `NEXT_PUBLIC_API_BASE_URL` for Next.js process.
4. If startup is slow:
   - Avoid rebuild by skipping `-ForceRebuild` in launcher.
