"""Learning-rate scheduling."""

from __future__ import annotations

import math

import torch

from srp_gpt2.config import TrainingConfig


class WarmupCosineScheduler:
    """Warmup followed by cosine decay."""

    def __init__(self, optimizer: torch.optim.Optimizer, config: TrainingConfig) -> None:
        self.optimizer = optimizer
        self.config = config

    def lr_at(self, step: int) -> float:
        if step < self.config.warmup_steps:
            return self.config.learning_rate * (step + 1) / max(1, self.config.warmup_steps)
        if step > self.config.max_steps:
            return self.config.min_learning_rate
        decay_ratio = (step - self.config.warmup_steps) / max(
            1, self.config.max_steps - self.config.warmup_steps
        )
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.config.min_learning_rate + coeff * (
            self.config.learning_rate - self.config.min_learning_rate
        )

    def step(self, step: int) -> float:
        lr = self.lr_at(step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def state_dict(self) -> dict[str, object]:
        return {}

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        return None
