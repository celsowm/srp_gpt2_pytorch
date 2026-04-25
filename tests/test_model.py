from __future__ import annotations

import torch

from srp_gpt2.config import ModelConfig, TrainingConfig
from srp_gpt2.model.gpt import GPTLanguageModel
from srp_gpt2.training.optimizer import build_adamw


def tiny_config() -> ModelConfig:
    return ModelConfig(vocab_size=257, block_size=16, n_layer=2, n_head=4, n_embd=32, dropout=0.0)


def test_forward_logits_and_loss_shape() -> None:
    config = tiny_config()
    model = GPTLanguageModel(config)
    x = torch.randint(0, config.vocab_size, (2, config.block_size))
    y = torch.randint(0, config.vocab_size, (2, config.block_size))
    output = model(x, y)
    assert output.logits.shape == (2, config.block_size, config.vocab_size)
    assert output.loss is not None
    assert output.loss.ndim == 0


def test_weight_tying() -> None:
    model = GPTLanguageModel(tiny_config())
    assert model.lm_head.weight.data_ptr() == model.embeddings.token_embedding.weight.data_ptr()


def test_optimizer_covers_trainable_parameters_once() -> None:
    model = GPTLanguageModel(tiny_config())
    optimizer = build_adamw(model, TrainingConfig(max_steps=1, warmup_steps=1))
    grouped = [p for group in optimizer.param_groups for p in group["params"]]
    grouped_ids = {id(p) for p in grouped}
    trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
    assert grouped_ids == trainable_ids
    assert len(grouped_ids) == len(grouped)
