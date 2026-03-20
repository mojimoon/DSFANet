# Ensemble-based Anomaly Detection for Cybersecurity

This repository contains the implementation of an ensemble-based Network Intrusion Detection System (NIDS).

## Setup

The project is implemented in Python 3.13 and manages dependencies using [poetry](https://python-poetry.org/). Please install poetry and a Python 3.13 environment before proceeding.

Install base dependencies first:

```bash
poetry install --no-root
```

Then install `torch` inside the Poetry virtual environment for your target backend.

### CPU

```bash
poetry run pip install --index-url https://download.pytorch.org/whl/cpu torch
```

### CUDA 12.8

```bash
poetry run pip install --index-url https://download.pytorch.org/whl/cu128 torch
```

### Other CUDA versions

Replace `cu128` with your CUDA index (for example `cu121`, `cu124`, `cu126`, `cu130`):

```bash
poetry run pip install --index-url https://download.pytorch.org/whl/cu121 torch
```

You can verify the installed backend with:

```bash
poetry run python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
```