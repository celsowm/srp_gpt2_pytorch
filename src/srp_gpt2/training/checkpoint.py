"""Checkpoint save/load responsibilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from srp_gpt2.config import ModelConfig, TrainingConfig


@dataclass
class TrainState:
    """Serializable training progress."""

    step: int = 0
    epoch: int = 0
    best_val_loss: float | None = None


class CheckpointManager:
    """Save and load training checkpoints."""

    def __init__(self, out_dir: str | Path) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        name: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        scheduler: Any | None,
        train_state: TrainState,
        model_config: ModelConfig,
        training_config: TrainingConfig | None = None,
    ) -> Path:
        path = self.out_dir / name
        raw_model = getattr(model, "_orig_mod", model)
        payload: dict[str, Any] = {
            "model_state": raw_model.state_dict(),
            "train_state": asdict(train_state),
            "model_config": asdict(model_config),
            "training_config": asdict(training_config) if training_config else None,
        }
        if optimizer is not None:
            payload["optimizer_state"] = optimizer.state_dict()
        if scheduler is not None and hasattr(scheduler, "state_dict"):
            payload["scheduler_state"] = scheduler.state_dict()
        torch.save(payload, path)
        return path

    @staticmethod
    def load(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
        return torch.load(path, map_location=map_location)
