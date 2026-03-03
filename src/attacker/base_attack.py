from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from src.runtime import resolve_device


class BaseAttack(ABC):
    def __init__(self, model, device: str = "cpu"):
        self.model = model
        self.device = resolve_device(device)

    @abstractmethod
    def generate(self, x_static, x_temporal, y):
        raise NotImplementedError
