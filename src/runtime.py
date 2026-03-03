from __future__ import annotations

import torch


def resolve_device(device: str | torch.device = "cpu") -> torch.device:
    if isinstance(device, torch.device):
        if device.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return device

    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
