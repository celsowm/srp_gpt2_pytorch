"""Configuration objects and serialization helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    """Architecture hyperparameters for a GPT-style decoder-only Transformer."""

    vocab_size: int = 50_257
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = True

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")


@dataclass(frozen=True)
class TrainingConfig:
    """Training-loop and optimizer hyperparameters."""

    batch_size: int = 4
    max_steps: int = 100_000
    gradient_accumulation_steps: int = 8
    learning_rate: float = 6e-4
    min_learning_rate: float = 6e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 2_000
    eval_interval: int = 1_000
    eval_batches: int = 50
    log_interval: int = 10
    save_interval: int = 1_000
    num_workers: int = 2
    amp: bool = True
    compile: bool = False
    seed: int = 1337

    def __post_init__(self) -> None:
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")


@dataclass(frozen=True)
class DataConfig:
    """Dataset and text loading hyperparameters."""

    stride: int = 1024
    encoding: str = "utf-8"


@dataclass(frozen=True)
class ProjectConfig:
    """Top-level project configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "ProjectConfig":
        model_payload = payload.get("model", {}) or {}
        training_payload = payload.get("training", {}) or {}
        data_payload = payload.get("data", {}) or {}
        return cls(
            model=ModelConfig(**model_payload),
            training=TrainingConfig(**training_payload),
            data=DataConfig(**data_payload),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ProjectConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        return cls.from_mapping(payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
