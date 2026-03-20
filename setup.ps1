param(
    [ValidateSet("cpu", "cu118", "cu121", "cu124", "cu128", "cu130")]
    [string]$Cuda = "cu130",
    [string]$Python = "3.13",
    [switch]$SkipTorch
)

$ErrorActionPreference = "Stop"

Write-Host "[setup] Creating/using Poetry environment (Python $Python) ..."
poetry env use $Python | Out-Host

Write-Host "[setup] Installing Python dependencies via Poetry ..."
poetry install | Out-Host

if (-not $SkipTorch) {
    if ($Cuda -eq "cpu") {
        $indexUrl = "https://download.pytorch.org/whl/cpu"
    }
    else {
        $indexUrl = "https://download.pytorch.org/whl/$Cuda"
    }
    Write-Host "[setup] Installing torch from $indexUrl ..."
    poetry run pip install --index-url $indexUrl torch | Out-Host

    Write-Host "[setup] Verifying torch backend ..."
    poetry run python -c "import torch; print(torch.__version__); print(torch.version.cuda)" | Out-Host
}

Write-Host "[setup] Installing frontend dependencies ..."
Push-Location "www"
try {
    npm install | Out-Host
}
finally {
    Pop-Location
}

Write-Host "[setup] Done."
