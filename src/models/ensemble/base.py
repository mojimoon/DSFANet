from __future__ import annotations

import json
from abc import ABC
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.runtime import resolve_device


class UnificationLayer:
    def __init__(self):
        self.stats: dict[str, dict[str, float]] = {}

    def register_stats(self, model_name: str, scores: np.ndarray) -> None:
        self.stats[model_name] = {
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
        }
        if self.stats[model_name]["max"] == self.stats[model_name]["min"]:
            self.stats[model_name]["max"] += 1e-6

    def unify(self, model_name: str, raw_scores: np.ndarray) -> np.ndarray:
        if model_name not in self.stats:
            return raw_scores
        stat = self.stats[model_name]
        unified = (raw_scores - stat["min"]) / (stat["max"] - stat["min"])
        return np.clip(unified, 0.0, 1.0)


class ModelWrapper:
    def __init__(
        self,
        name: str,
        model: Any,
        model_type: str,
        input_req: str,
        unifier: UnificationLayer,
        checkpoint_path: str | None = None,
        device: str | torch.device = "cpu",
        temporal_keep_indices: list[int] | None = None,
    ):
        self.name = name
        self.model = model
        self.model_type = model_type
        self.input_req = input_req
        self.unifier = unifier
        self.checkpoint_path = checkpoint_path
        self.device = resolve_device(device)
        self.temporal_keep_indices = temporal_keep_indices

    def _build_combined_input(self, x_static: np.ndarray, x_temporal: np.ndarray) -> np.ndarray:
        if self.input_req == "combined_no_ts":
            if self.temporal_keep_indices is not None:
                x_t_use = x_temporal[:, self.temporal_keep_indices]
            elif x_temporal.shape[1] > 2:
                x_t_use = x_temporal[:, :-2]
            else:
                x_t_use = x_temporal
            return np.concatenate([x_static, x_t_use], axis=1)
        return np.concatenate([x_static, x_temporal], axis=1)

    def get_raw_score(self, x_static: np.ndarray, x_temporal: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError(
                f"Model '{self.name}' is unresolved. Please load and attach model from checkpoint: {self.checkpoint_path}"
            )

        inputs: list[Any] = []
        is_torch_model = isinstance(self.model, torch.nn.Module)

        if self.input_req in ["combined_all", "combined_no_ts"]:
            x_combined = self._build_combined_input(x_static, x_temporal)
            combined_in = torch.FloatTensor(x_combined).to(self.device) if is_torch_model else x_combined
            inputs.append(combined_in)
        else:
            if self.input_req in ["static", "both"]:
                static_in = torch.FloatTensor(x_static).to(self.device) if is_torch_model else x_static
                inputs.append(static_in)
            if self.input_req in ["temporal", "both"]:
                temporal_in = torch.FloatTensor(x_temporal).to(self.device) if is_torch_model else x_temporal
                inputs.append(temporal_in)

        if is_torch_model:
            self.model.eval()
            self.model.to(self.device)
            with torch.no_grad():
                out = self.model(inputs[0], inputs[1]) if len(inputs) == 2 else self.model(inputs[0])

                if self.model_type == "classifier":
                    probs = torch.softmax(out, dim=1).cpu().numpy()
                    raw = probs[:, 1]
                else:
                    x_in = inputs[0].detach().cpu().numpy()
                    x_out = out.detach().cpu().numpy()
                    raw = np.mean(np.power(x_in - x_out, 2), axis=1)
        else:
            if hasattr(self.model, "predict_proba"):
                raw = self.model.predict_proba(inputs[0])[:, 1]
            else:
                raw = self.model.predict(inputs[0])

        return raw

    def get_unified_score(self, x_static: np.ndarray, x_temporal: np.ndarray) -> np.ndarray:
        raw = self.get_raw_score(x_static, x_temporal)
        return self.unifier.unify(self.name, raw)


class BaseEnsemble(ABC):
    def __init__(self, unifier: UnificationLayer, device: str | torch.device = "cpu"):
        self.unifier = unifier
        self.models: list[ModelWrapper] = []
        self.last_intermediate_results: dict[str, np.ndarray] = {}
        self.device = resolve_device(device)

    @staticmethod
    def _default_checkpoint_dir() -> Path:
        project_root = Path(__file__).resolve().parents[3]
        ckpt_dir = project_root / "models"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        return ckpt_dir

    def add_model(
        self,
        name: str,
        model: Any,
        model_type: str = "classifier",
        input_req: str = "static",
        checkpoint_path: str | None = None,
        temporal_keep_indices: list[int] | None = None,
    ) -> None:
        wrapper = ModelWrapper(
            name=name,
            model=model,
            model_type=model_type,
            input_req=input_req,
            unifier=self.unifier,
            checkpoint_path=checkpoint_path,
            device=self.device,
            temporal_keep_indices=temporal_keep_indices,
        )
        self.models.append(wrapper)

    def calibrate(self, x_static_val: np.ndarray, x_temporal_val: np.ndarray) -> None:
        print(f"[{self.__class__.__name__}] Calibrating base models...")
        for wrapper in self.models:
            raw = wrapper.get_raw_score(x_static_val, x_temporal_val)
            self.unifier.register_stats(wrapper.name, raw)

    def _collect_base_scores(self, x_static: np.ndarray, x_temporal: np.ndarray) -> np.ndarray:
        scores_list = []
        self.last_intermediate_results = {}
        for wrapper in self.models:
            score = wrapper.get_unified_score(x_static, x_temporal)
            scores_list.append(score)
            self.last_intermediate_results[wrapper.name] = score
        return np.column_stack(scores_list)

    def get_intermediate_results(self) -> dict[str, np.ndarray]:
        return self.last_intermediate_results

    def get_hparams(self) -> dict[str, Any]:
        return {}

    @classmethod
    def from_hparams(cls, hparams: dict[str, Any], unifier: UnificationLayer, device: str | torch.device = "cpu") -> "BaseEnsemble":
        return cls(unifier=unifier, device=device, **hparams)

    def save_checkpoint(self, filename: str | None = None, checkpoint_dir: str | Path | None = None) -> str:
        ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else self._default_checkpoint_dir()
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        if filename is None:
            filename = f"{self.__class__.__name__}.json"

        payload = {
            "ensemble_class": self.__class__.__name__,
            "hparams": self.get_hparams(),
            "unifier_stats": self.unifier.stats,
            "models": [
                {
                    "name": m.name,
                    "model_type": m.model_type,
                    "input_req": m.input_req,
                    "checkpoint_path": m.checkpoint_path,
                    "temporal_keep_indices": m.temporal_keep_indices,
                }
                for m in self.models
            ],
        }

        path = ckpt_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return str(path)

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: str | Path,
        model_registry: dict[str, Any] | None = None,
        device: str | torch.device = "cpu",
    ) -> "BaseEnsemble":
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        unifier = UnificationLayer()
        unifier.stats = payload.get("unifier_stats", {})
        instance = cls.from_hparams(payload.get("hparams", {}), unifier, device=device)

        for model_info in payload.get("models", []):
            model_name = model_info["name"]
            model = model_registry.get(model_name) if model_registry else None
            instance.add_model(
                name=model_name,
                model=model,
                model_type=model_info["model_type"],
                input_req=model_info["input_req"],
                checkpoint_path=model_info.get("checkpoint_path"),
                temporal_keep_indices=model_info.get("temporal_keep_indices"),
            )
        return instance
