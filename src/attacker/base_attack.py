

from abc import ABC, abstractmethod

import torch

from src.runtime import resolve_device


class BaseAttack(ABC):
    """Base interface for adversarial sample generators."""

    def __init__(self, model, device="cpu"):
        """Initialize attack with target model and runtime device."""
        self.model = model
        self.device = resolve_device(device)

    @abstractmethod
    def generate(self, x_static, x_temporal, y):
        """Generate adversarial variants for input batch.

        Returns:
            adv_x_static: torch.Tensor
            adv_x_temporal: torch.Tensor
        """
        raise NotImplementedError
