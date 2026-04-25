"""Transformer block composition."""

from __future__ import annotations

import torch
from torch import nn

from srp_gpt2.config import ModelConfig
from srp_gpt2.model.attention import CausalSelfAttention
from srp_gpt2.model.feed_forward import FeedForward


class TransformerBlock(nn.Module):
    """Pre-norm GPT Transformer block."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attention = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.feed_forward = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln_1(x))
        x = x + self.feed_forward(self.ln_2(x))
        return x
