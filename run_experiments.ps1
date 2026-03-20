param(
    [ValidateSet("single", "all")]
    [string]$Mode = "all",
    [string]$RunId = "unsw-main",
    [string]$RunIdSuffix = "main",
    [string]$BaseDataset = "NF-UNSW-NB15-v3.csv",
    [string]$Device = "cuda",
    [object]$Steps = "1,2,3,4,5,6,7,8",
    [object]$Epochs = "10,10,20",
    [int]$SizeLimit = 0,
    [string]$OodDataset = "NF-BoT-IoT-v3.csv"
)

$ErrorActionPreference = "Stop"

function Normalize-CSVArg {
    param([object]$Value)
    if ($Value -is [System.Array]) {
        return (($Value | ForEach-Object { [string]$_ }) -join ",")
    }
    return [string]$Value
}

$StepsText = Normalize-CSVArg -Value $Steps
$EpochsText = Normalize-CSVArg -Value $Epochs

function Invoke-Experiment {
    param(
        [string]$CurrentRunId,
        [string]$CurrentBaseDataset
    )

    $args = @(
        "experiments_main.py",
        "--run-id", $CurrentRunId,
        "--steps", $StepsText,
        "--epochs", $EpochsText,
        "--base-dataset", $CurrentBaseDataset,
        "--ood-dataset", $OodDataset,
        "--device", $Device
    )

    if ($SizeLimit -gt 0) {
        $limit = [string]$SizeLimit
        $args += @(
            "--test-size", $limit,
            "--max-train-samples", $limit,
            "--drift-subset-size", $limit,
            "--natural-shift-size", $limit,
            "--max-benign-for-attacks", $limit,
            "--step5-train-max-samples", $limit,
            "--step5-eval-max-samples", $limit,
            "--step6-val-max-samples", $limit,
            "--step6-eval-max-samples", $limit
        )
    }

    Write-Host "[run_experiments] Running: $CurrentRunId ($CurrentBaseDataset)"
    poetry run python @args
}

if ($Mode -eq "single") {
    Invoke-Experiment -CurrentRunId $RunId -CurrentBaseDataset $BaseDataset
}
else {
    $jobs = @(
        @{ RunId = "unsw-$RunIdSuffix"; Dataset = "NF-UNSW-NB15-v3.csv" },
        @{ RunId = "ton-$RunIdSuffix"; Dataset = "NF-ToN-IoT-v3.csv" },
        @{ RunId = "ids2018-$RunIdSuffix"; Dataset = "NF-CICIDS2018-v3.csv" }
    )

    foreach ($job in $jobs) {
        Invoke-Experiment -CurrentRunId $job.RunId -CurrentBaseDataset $job.Dataset
    }
}
