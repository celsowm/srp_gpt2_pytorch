from __future__ import annotations

import torch

from srp_gpt2.inference.sampler import Sampler, SamplingConfig


def test_sampler_returns_one_token_per_batch() -> None:
    torch.manual_seed(123)
    sampler = Sampler(SamplingConfig(temperature=1.0, top_k=5, top_p=0.9))
    logits = torch.randn(3, 20)
    tokens = sampler.sample(logits)
    assert tokens.shape == (3, 1)
    assert tokens.dtype == torch.long


def test_repetition_penalty_reduces_seen_token_logits() -> None:
    sampler = Sampler(SamplingConfig(repetition_penalty=1.5))
    logits = torch.tensor([[4.0, -2.0, 1.0]], dtype=torch.float32)
    generated = torch.tensor([[0, 1, 0]], dtype=torch.long)

    adjusted = sampler._apply_repetition_penalty(logits, generated)

    assert torch.isclose(adjusted[0, 0], torch.tensor(2.6667), atol=1e-4)
    assert torch.isclose(adjusted[0, 1], torch.tensor(-3.0), atol=1e-4)
    assert torch.isclose(adjusted[0, 2], torch.tensor(1.0), atol=1e-4)
