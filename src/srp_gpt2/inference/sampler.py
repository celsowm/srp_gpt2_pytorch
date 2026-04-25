"""Next-token sampling strategies."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SamplingConfig:
    """Controls stochastic decoding."""

    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    repetition_penalty: float = 1.0

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be > 0")


class Sampler:
    """Sample a next token from logits."""

    def __init__(self, config: SamplingConfig) -> None:
        self.config = config

    def sample(self, logits: torch.Tensor, generated: torch.Tensor | None = None) -> torch.Tensor:
        logits = logits / self.config.temperature
        logits = self._apply_repetition_penalty(logits, generated)
        logits = self._apply_top_k(logits)
        logits = self._apply_top_p(logits)
        probabilities = F.softmax(logits, dim=-1)
        return torch.multinomial(probabilities, num_samples=1)

    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        generated: torch.Tensor | None,
    ) -> torch.Tensor:
        if generated is None or self.config.repetition_penalty == 1.0:
            return logits
        if generated.ndim != 2:
            raise ValueError("generated must have shape (batch, sequence_length)")

        adjusted = logits.clone()
        penalty = self.config.repetition_penalty
        for batch_idx in range(generated.size(0)):
            seen_tokens = torch.unique(generated[batch_idx])
            token_logits = adjusted[batch_idx, seen_tokens]
            adjusted[batch_idx, seen_tokens] = torch.where(
                token_logits < 0,
                token_logits * penalty,
                token_logits / penalty,
            )
        return adjusted

    def _apply_top_k(self, logits: torch.Tensor) -> torch.Tensor:
        if self.config.top_k is None:
            return logits
        top_k = min(self.config.top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k)
        threshold = values[..., -1, None]
        return logits.masked_fill(logits < threshold, float("-inf"))

    def _apply_top_p(self, logits: torch.Tensor) -> torch.Tensor:
        if self.config.top_p is None or self.config.top_p >= 1.0:
            return logits
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        sorted_indices_to_remove = cumulative_probs > self.config.top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        return logits.masked_fill(indices_to_remove, float("-inf"))
