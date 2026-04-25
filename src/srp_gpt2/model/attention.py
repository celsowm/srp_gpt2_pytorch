"""Causal multi-head self-attention."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from srp_gpt2.config import ModelConfig


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal mask."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.config = config
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv_projection = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.out_projection = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, channels = x.shape
        qkv = self.qkv_projection(x)
        q, k, v = qkv.split(channels, dim=2)

        q = self._split_heads(q, batch, time)
        k = self._split_heads(k, batch, time)
        v = self._split_heads(v, batch, time)

        y = self._scaled_dot_product_attention(q, k, v, time)
        y = y.transpose(1, 2).contiguous().view(batch, time, channels)
        return self.resid_dropout(self.out_projection(y))

    def _split_heads(self, x: torch.Tensor, batch: int, time: int) -> torch.Tensor:
        return x.view(batch, time, self.n_head, self.head_dim).transpose(1, 2)

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        time: int,
    ) -> torch.Tensor:
        dropout_p = self.attn_dropout.p if self.training else 0.0
        if hasattr(F, "scaled_dot_product_attention"):
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=True,
            )
        return self._manual_attention(q, k, v, time)

    def _manual_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        time: int,
    ) -> torch.Tensor:
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        mask = self.causal_mask[:, :, :time, :time]
        att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        return att @ v
