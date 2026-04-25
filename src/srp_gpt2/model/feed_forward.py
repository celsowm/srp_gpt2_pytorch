"""GPT-style feed-forward network."""

from __future__ import annotations

from torch import nn

from srp_gpt2.config import ModelConfig


class FeedForward(nn.Module):
    """Position-wise MLP used inside a GPT block."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):  # type: ignore[no-untyped-def]
        return self.net(x)
