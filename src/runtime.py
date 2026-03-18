

import torch


def resolve_device(device="cpu") -> torch.device:
    """Turn a device string or torch.device into a valid torch.device object.

    Returns:
        device_obj: torch.device
    """
    if isinstance(device, torch.device):
        if device.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return device

    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
