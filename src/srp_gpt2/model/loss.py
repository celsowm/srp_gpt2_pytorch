"""Language-model loss functions."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def causal_lm_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute next-token cross entropy for logits shaped ``[B, T, vocab]``."""
    if logits.ndim != 3:
        raise ValueError(f"expected logits [B, T, V], got {tuple(logits.shape)}")
    if targets.ndim != 2:
        raise ValueError(f"expected targets [B, T], got {tuple(targets.shape)}")
    batch, time, vocab = logits.shape
    return F.cross_entropy(logits.reshape(batch * time, vocab), targets.reshape(batch * time))
