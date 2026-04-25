"""Optimizer construction."""

from __future__ import annotations

import torch
from torch import nn

from srp_gpt2.config import TrainingConfig


def build_adamw(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    """Build AdamW with explicit decay/no-decay parameter groups."""
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2 and not _is_embedding_weight(name):
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        param_groups,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
    )


def _is_embedding_weight(name: str) -> bool:
    return "embedding" in name and name.endswith("weight")
