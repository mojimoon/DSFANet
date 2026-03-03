param(
    [string]$Dataset = "NF-UNSW-NB15-v3.csv",
    [string]$Device = "cpu",
    [int]$ApiPort = 8000,
    [int]$WebPort = 3000,
    [switch]$ForceRebuild
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

Write-Host "[1/5] Checking Python..."
python --version | Out-Null

Write-Host "[2/5] Checking pnpm..."
pnpm --version | Out-Null

$dashboardJson = Join-Path $projectRoot "out/www/dashboard_data.json"
if ($ForceRebuild -or -not (Test-Path $dashboardJson)) {
    Write-Host "[3/5] Generating dashboard artifacts (this can take several minutes)..."
    python web_main.py --skip-serve --dataset $Dataset --device $Device
} else {
    Write-Host "[3/5] Reusing existing dashboard artifacts at out/www/."
}

Write-Host "[4/5] Starting Python API server on http://127.0.0.1:$ApiPort ..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$projectRoot'; python web_main.py --serve-only --host 127.0.0.1 --port $ApiPort"

Write-Host "[5/5] Starting Next.js frontend on http://127.0.0.1:$WebPort ..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$projectRoot/www'; if (-not (Test-Path node_modules)) { pnpm install }; `$env:NEXT_PUBLIC_API_BASE_URL='http://127.0.0.1:$ApiPort'; pnpm dev --port $WebPort"

Start-Sleep -Seconds 3
Start-Process "http://127.0.0.1:$WebPort"
Write-Host "Done. Two new terminals were opened for API and frontend."