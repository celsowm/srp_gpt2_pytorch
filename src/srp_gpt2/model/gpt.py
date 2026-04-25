"""GPT language model assembly."""

from __future__ import annotations

from dataclasses import asdict
from typing import NamedTuple

import torch
from torch import nn

from srp_gpt2.config import ModelConfig
from srp_gpt2.model.block import TransformerBlock
from srp_gpt2.model.embeddings import TokenPositionEmbeddings
from srp_gpt2.model.init import GPTWeightInitializer
from srp_gpt2.model.loss import causal_lm_loss


class GPTOutput(NamedTuple):
    """Output container for logits and optional loss."""

    logits: torch.Tensor
    loss: torch.Tensor | None


class GPTLanguageModel(nn.Module):
    """Decoder-only GPT model for causal language modeling."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = TokenPositionEmbeddings(config)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.embeddings.token_embedding.weight
        GPTWeightInitializer(config).initialize(self)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> GPTOutput:
        x = self.embeddings(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = causal_lm_loss(logits, targets) if targets is not None else None
        return GPTOutput(logits=logits, loss=loss)

    @torch.no_grad()
    def crop_block_size(self, block_size: int) -> None:
        """Reduce maximum context length for inference or smaller fine-tuning."""
        if block_size > self.config.block_size:
            raise ValueError("new block_size must be <= current block_size")
        self.config = ModelConfig(**{**asdict(self.config), "block_size": block_size})
        self.embeddings.config = self.config
        self.embeddings.position_embedding.weight = nn.Parameter(
            self.embeddings.position_embedding.weight[:block_size]
        )
        for block in self.blocks:
            block.attention.config = self.config
            block.attention.causal_mask = block.attention.causal_mask[:, :, :block_size, :block_size]

    def count_parameters(self, trainable_only: bool = True) -> int:
        params = self.parameters()
        if trainable_only:
            return sum(p.numel() for p in params if p.requires_grad)
        return sum(p.numel() for p in params)
