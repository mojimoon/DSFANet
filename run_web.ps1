param(
    [string]$BindHost = "127.0.0.1",
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 3000,
    [switch]$Quiet,
    [switch]$BackendOnly,
    [switch]$FrontendOnly
)

$ErrorActionPreference = "Stop"

if (-not $FrontendOnly) {
    $quietArg = if ($Quiet) { "--quiet" } else { "" }
    $backendCmd = "poetry run python web_main.py --host $BindHost --port $BackendPort $quietArg"
    Write-Host "[run_web] Starting backend: $backendCmd"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCmd | Out-Null
}

if (-not $BackendOnly) {
    Write-Host "[run_web] Starting frontend on port $FrontendPort ..."
    Push-Location "www"
    try {
        if ($FrontendPort -ne 3000) {
            $env:PORT = [string]$FrontendPort
        }
        npm run dev
    }
    finally {
        Pop-Location
    }
}
