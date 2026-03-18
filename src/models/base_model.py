

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.runtime import resolve_device


class BaseIDSModel(nn.Module, ABC):
    """Shared checkpoint/device utilities for IDS torch models."""

    def __init__(self, device="cpu"):
        """Initialize model on resolved runtime device."""
        super().__init__()
        self.device = resolve_device(device)
        self.to(self.device)

    @staticmethod
    def _default_checkpoint_dir() -> Path:
        """Return default checkpoint directory under project root.

        Returns:
            ckpt_dir: Path
        """
        project_root = Path(__file__).resolve().parents[2]
        ckpt_dir = project_root / "models"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        return ckpt_dir

    def set_device(self, device):
        """Move model parameters to target device."""
        self.device = resolve_device(device)
        self.to(self.device)

    @abstractmethod
    def get_init_params(self) -> dict[str, Any]:
        """Return constructor params required to reload this model."""
        pass

    def save_checkpoint(self, filename: str | None = None, checkpoint_dir: str | Path | None = None, extra: dict[str, Any] | None = None) -> str:
        """Save model state and init params to a checkpoint file.

        Returns:
            checkpoint_path: str
        """
        ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else self._default_checkpoint_dir()
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            filename = f"{self.__class__.__name__}.pt"

        path = ckpt_dir / filename
        payload = {
            "class_name": self.__class__.__name__,
            "init_params": self.get_init_params(),
            "state_dict": self.state_dict(),
            "extra": extra or {},
        }
        torch.save(payload, path)
        return str(path)

    @classmethod
    def load_checkpoint(cls, checkpoint_path, device="cpu") -> "BaseIDSModel":
        """Load model from checkpoint and switch to eval mode.

        Returns:
            model: BaseIDSModel
        """
        map_location = resolve_device(device)
        payload = torch.load(checkpoint_path, map_location=map_location)
        init_params = payload.get("init_params", {})
        init_params["device"] = str(map_location)
        model = cls(**init_params)
        model.load_state_dict(payload["state_dict"])
        model.to(map_location)
        model.eval()
        return model
