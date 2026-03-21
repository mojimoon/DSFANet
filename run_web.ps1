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

function Wait-BackendReady {
    param(
        [string]$BindAddress,
        [int]$Port,
        [int]$TimeoutSec = 60
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    $url = "http://${BindAddress}:${Port}/"

    while ((Get-Date) -lt $deadline) {
        try {
            $resp = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2
            if ($resp.StatusCode -ge 200 -and $resp.StatusCode -lt 500) {
                return $true
            }
        }
        catch {
            Start-Sleep -Milliseconds 500
        }
    }

    return $false
}

if (-not $FrontendOnly) {
    $verboseArg = if ($Verbose) { "--verbose" } else { "" }
    $experimentArgs = ""
    if ($RunExperiment) {
        $experimentArgs = "--run-experiment --run-id-suffix $RunIdSuffix --device $Device"
    }
    $backendCmd = "poetry run python web_main.py --host $BindHost --port $BackendPort $experimentArgs $verboseArg"
    Write-Host "[run_web] Starting backend: $backendCmd"
    $backendProcess = Start-Process powershell -WorkingDirectory $PSScriptRoot -ArgumentList "-NoExit", "-Command", $backendCmd -PassThru

    if (-not $BackendOnly) {
        Write-Host "[run_web] Waiting for backend to become ready at http://${BindHost}:${BackendPort} ..."
        if (-not (Wait-BackendReady -BindAddress $BindHost -Port $BackendPort -TimeoutSec 120)) {
            if ($backendProcess.HasExited) {
                throw "Backend process exited before becoming ready (exit code: $($backendProcess.ExitCode))."
            }
            throw "Backend did not become ready in time at http://${BindHost}:${BackendPort}."
        }
    }
}

if (-not $BackendOnly) {
    Write-Host "[run_web] Starting frontend on port $FrontendPort ..."
    Push-Location "www"
    try {
        $env:NEXT_PUBLIC_API_BASE_URL = "http://${BindHost}:${BackendPort}"
        if ($FrontendPort -ne 3000) {
            $env:PORT = [string]$FrontendPort
        }
        npm run dev
    }
    finally {
        Pop-Location
    }
}
