# Ensemble-based Anomaly Detection for Cybersecurity

This repository contains the implementation of an ensemble-based Network Intrusion Detection System (NIDS).

## Setup

The project is implemented in Python 3.13 and manages dependencies using [poetry](https://python-poetry.org/). Please install poetry and a Python 3.13 environment before proceeding.

To set up the environment, execute:

```bash
poetry install --no-root
```

It should be noted that the project relies on PyTorch on CUDA 13.0 (cu130). If you have a NVIDIA GPU but with a different CUDA version, or only have a CPU, please follow the following steps:

1. Add corresponding source:

```bash
poetry source add --priority explicit pytorch-cu121 https://download.pytorch.org/whl/cu121
```

Replace `cu121` with your CUDA version, e.g., `cu118` for CUDA 11.8, or `cpu` for CPU-only:

```bash
poetry source add --priority explicit pytorch-cpu https://download.pytorch.org/whl/cpu
```

2. Update the `torch` dependency in poetry:

```bash
