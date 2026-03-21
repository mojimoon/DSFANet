param(
    [string]$BindHost = "127.0.0.1",
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 3000,
    [switch]$RunExperiment,
    [string]$RunIdSuffix = "main",
    [string]$Device = "cpu",
    [switch]$Verbose,
    [switch]$BackendOnly,
    [switch]$FrontendOnly
)

$ErrorActionPreference = "Stop"

if (-not $FrontendOnly) {
    $verboseArg = if ($Verbose) { "--verbose" } else { "" }
    $experimentArgs = ""
    if ($RunExperiment) {
        $experimentArgs = "--run-experiment --run-id-suffix $RunIdSuffix --device $Device"
    }
    $backendCmd = "poetry run python web_main.py --host $BindHost --port $BackendPort $experimentArgs $verboseArg"
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
