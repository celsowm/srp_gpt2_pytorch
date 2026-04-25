"""Autoregressive text generation."""

from __future__ import annotations

import torch
from torch import nn

from srp_gpt2.data.tokenizer import TokenizerProtocol
from srp_gpt2.inference.sampler import Sampler, SamplingConfig


class TextGenerator:
    """Generate text with a causal language model."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: TokenizerProtocol,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = torch.device(device)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        sampling: SamplingConfig | None = None,
        stop_on_eos: bool = True,
    ) -> str:
        token_ids = self.tokenizer.encode(prompt)
        if not token_ids:
            eos = self.tokenizer.eos_token_id
            token_ids = [eos if eos is not None else 0]
        generated = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        sampler = Sampler(sampling or SamplingConfig())
        block_size = self.model.config.block_size

        for _ in range(max_new_tokens):
            context = generated[:, -block_size:]
            output = self.model(context)
            next_logits = output.logits[:, -1, :]
            next_token = sampler.sample(next_logits, generated)
            generated = torch.cat((generated, next_token), dim=1)
            if stop_on_eos and self.tokenizer.eos_token_id is not None:
                if int(next_token.item()) == self.tokenizer.eos_token_id:
                    break

        return self.tokenizer.decode(generated[0].tolist())
