# Ensemble-based Anomaly Detection for Cybersecurity

This repository contains the implementation of an ensemble-based Network Intrusion Detection System (NIDS).

## One-click Setup

- Run `setup.ps1 --cuda cu130` to set up the environment with CUDA 13.0 support. (Or replace `cu130` with your CUDA version or `cpu`)
- Run `run_experiments.ps1` to execute the full experiment pipeline on all three datasets sequentially.
- Run `run_web.ps1` to start the web dashboard.

## Setup the Environment

The project is implemented in Python 3.13 and manages dependencies using [poetry](https://python-poetry.org/). Please install poetry and a Python 3.13 environment before proceeding.

(1) Install base dependencies first:

```bash
poetry install
```

(2) Install PyTorch. The installation command depends on your hardware and CUDA version.

**Note**: You can skip installing PyTorch if you only want to host the web dashboard without running the experiments.

### CPU

```bash
poetry run pip install --index-url https://download.pytorch.org/whl/cpu torch
```

### CUDA 13.0

```bash
poetry run pip install --index-url https://download.pytorch.org/whl/cu130 torch
```

### Other CUDA versions

Replace `cu130` with the appropriate version (e.g., `cu128`, `cu124`, `cu121`, `cu118`, etc.):

```bash
poetry run pip install --index-url https://download.pytorch.org/whl/cu121 torch
```

You can verify the installed backend with:

```bash
poetry run python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
```

It should be noted that the project is only tested with PyTorch 2.10.0 on CUDA 13.0 and CPU. Compatibility with other versions may vary.

(3) Install the web dashboard dependencies:

```bash
cd www
npm install
```

## Running the Experiments

Running the training and evaluation scripts.

**Note**: You can skip this step if you only want to host the web dashboard without running the experiments. Running the experiments will take a significant amount of time and computational resources, so a GPU is recommended.

```bash
poetry run python experiments_main.py --run-id unsw-main --steps 1,2,3,4,5,6,7,8 --epochs 10,10,20 --base-dataset NF-UNSW-NB15-v3.csv --device cuda
poetry run python experiments_main.py --run-id ton-main --steps 1,2,3,4,5,6,7,8 --epochs 10,10,20 --base-dataset NF-ToN-IoT-v3.csv --device cuda
poetry run python experiments_main.py --run-id ids2018-main --steps 1,2,3,4,5,6,7,8 --epochs 10,10,20 --base-dataset NF-CICIDS2018-v3.csv --device cuda
```

`experiments_main.py` is the entry point for the whole pipeline. A detailed list of parameters and their descriptions can be found in the source code or by running:

```bash
poetry run python experiments_main.py --help
```

Below is a brief description of the parameters used in the above commands:

- `--run-id`: A unique identifier for the experiment run. This will be used to organize the results and logs. If you wish to continue a previous run, use the same `run-id` and specify the steps you want to overwrite.
- `--steps`: A comma-separated list of steps to execute.
    1. Benchmarking the models on the base dataset.
    2. Evaluating the models under various shifts, including natural, label, corruption, and adversarial shifts.
    3. Adversarial retraining of the models.
    4. Evaluating the best ensemble and generating SHAP explanations.
    5. Ablation study for DSFANet.
    6. Ablation study for the ensemble.
    7. Comparative evaluation of transfer ensembles on natural shifts.
    8. Exporting results for the web dashboard.
- `--epochs`: A comma-separated list of the number of epochs for training AutoEncoder, LSTM, and DSFANet, respectively.
- `--base-dataset`: The base dataset to use for training and evaluation. This should be a CSV file located in the `data/` directory.
- `--device`: Options include `cpu`, `cuda`, or a specific CUDA device like `cuda:0`.

The experiments results will be saved in the `out/experiments/<run-id>/` directory, organized by steps.

The web export results will be saved in `out/web/` directory, organized by run IDs.

## Running the Web Dashboard

(1) Starting the backend server:

```bash
poetry run python web_server.py --quiet
```

The backend server will start on `http://127.0.0.1:8000/` by default.

Remove the `--quiet` flag if you want to see the API request logs in the console.

(2) Starting the frontend server:

```bash
cd www
npm start
```

The frontend server will start on `http://localhost:3000/` by default and will automatically open in your default web browser.
