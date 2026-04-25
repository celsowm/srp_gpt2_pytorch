"""Weight initialization for GPT-style modules."""

from __future__ import annotations

import math

from torch import nn

from srp_gpt2.config import ModelConfig


class GPTWeightInitializer:
    """Apply GPT-2 style parameter initialization."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.base_std = 0.02
        self.residual_std = self.base_std / math.sqrt(2 * config.n_layer)

    def initialize(self, module: nn.Module) -> None:
        module.apply(self._initialize_module)
        self._scale_residual_projections(module)

    def _initialize_module(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.base_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.base_std)

    def _scale_residual_projections(self, module: nn.Module) -> None:
        for name, parameter in module.named_parameters():
            if name.endswith("out_projection.weight") or name.endswith("feed_forward.net.2.weight"):
                nn.init.normal_(parameter, mean=0.0, std=self.residual_std)
