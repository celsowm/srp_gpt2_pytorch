"""Embedding layers for token and absolute position IDs."""

from __future__ import annotations

import torch
from torch import nn

from srp_gpt2.config import ModelConfig


class TokenPositionEmbeddings(nn.Module):
    """Create token + learned positional embeddings."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [B, T], got {tuple(input_ids.shape)}")
        _, time = input_ids.shape
        if time > self.config.block_size:
            raise ValueError(
                f"sequence length {time} exceeds block_size {self.config.block_size}"
            )
        positions = torch.arange(0, time, dtype=torch.long, device=input_ids.device)
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(positions)[None, :, :]
        return self.dropout(token_emb + pos_emb)
