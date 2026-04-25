from __future__ import annotations

import builtins

import pytest
import torch

from srp_gpt2.config import ModelConfig
from srp_gpt2.config import TrainingConfig
from srp_gpt2.data.tokenizer import ByteTokenizer, GPT2BPETokenizer
from srp_gpt2.model.gpt import GPTLanguageModel
from srp_gpt2.training.checkpoint import CheckpointManager, TrainState
from srp_gpt2.xray import (
    TinyLiveGenerationSession,
    TinyLiveTrainingSession,
    build_xray_tokenizer,
    causal_attention_maps,
    display_token_text,
    inspect_logits,
    resolve_xray_device,
    token_table,
    trace_transformer_forward,
)


def tiny_config() -> ModelConfig:
    return ModelConfig(vocab_size=257, block_size=8, n_layer=2, n_head=2, n_embd=16, dropout=0.0)


def test_inspect_logits_returns_sorted_probabilities_and_entropy() -> None:
    tokenizer = ByteTokenizer()
    logits = torch.tensor([0.0, 3.0, 1.0, -1.0], dtype=torch.float32)

    inspection = inspect_logits(logits, tokenizer, top_k=3)

    probabilities = [token.probability for token in inspection.top_tokens]
    assert [token.token_id for token in inspection.top_tokens] == [1, 2, 0]
    assert probabilities == sorted(probabilities, reverse=True)
    assert 0 < inspection.confidence < 1
    assert inspection.entropy > 0


def test_causal_attention_maps_have_expected_shape_and_mask() -> None:
    torch.manual_seed(123)
    model = GPTLanguageModel(tiny_config())
    input_ids = torch.randint(0, model.config.vocab_size, (1, 5))

    maps = causal_attention_maps(model, input_ids)

    assert len(maps) == model.config.n_layer
    for layer_map in maps:
        assert layer_map.shape == (model.config.n_head, 5, 5)
        future_mask = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
        assert torch.allclose(layer_map[:, future_mask], torch.zeros_like(layer_map[:, future_mask]))


def test_resolve_xray_device_prefers_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert resolve_xray_device("auto").type == "cuda"


def test_resolve_xray_device_uses_mps_when_cuda_is_missing(monkeypatch) -> None:
    class FakeMps:
        @staticmethod
        def is_available() -> bool:
            return True

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends, "mps", FakeMps())

    assert resolve_xray_device("auto").type == "mps"


def test_resolve_xray_device_falls_back_to_cpu(monkeypatch) -> None:
    class FakeMps:
        @staticmethod
        def is_available() -> bool:
            return False

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends, "mps", FakeMps())

    assert resolve_xray_device("auto").type == "cpu"


def test_trace_transformer_forward_captures_pipeline_shapes() -> None:
    torch.manual_seed(123)
    tokenizer = ByteTokenizer()
    model = GPTLanguageModel(tiny_config())
    input_ids = torch.randint(0, model.config.vocab_size, (1, 5))

    trace = trace_transformer_forward(model, input_ids, tokenizer)

    assert trace.embeddings.shape == (1, 5, model.config.n_embd)
    assert len(trace.blocks) == model.config.n_layer
    assert trace.blocks[0].attention.shape == (1, 5, model.config.n_embd)
    assert trace.blocks[0].mlp.shape == (1, 5, model.config.n_embd)
    assert trace.final_norm.shape == (1, 5, model.config.n_embd)
    assert trace.logits.shape == (1, 5, model.config.vocab_size)
    assert len(trace.next_token.top_tokens) == 5


def test_gpt2_bpe_token_table_uses_subword_tokens() -> None:
    tokenizer = GPT2BPETokenizer()
    text = "o rato roeu a roupa do rei de roma"

    rows = token_table(tokenizer, text)

    assert len(rows) < len(text)
    assert any(row["char_end"] - row["char_start"] > 1 for row in rows)
    assert any(str(row["text"]).startswith("▁") for row in rows)


def test_build_xray_tokenizer_does_not_silently_fallback_when_tiktoken_is_missing(monkeypatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "tiktoken":
            raise ImportError("missing tiktoken")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="will not silently fall back"):
        build_xray_tokenizer("gpt2")


def test_byte_debug_tokenizer_still_exposes_byte_hex_labels() -> None:
    tokenizer = build_xray_tokenizer("byte-debug")

    assert tokenizer.vocab_size == 257
    assert display_token_text(tokenizer, 195) == "0xC3"


def test_tiny_live_training_session_defaults_to_gpt2(tmp_path) -> None:
    text_file = tmp_path / "tiny.txt"
    text_file.write_text("O rato roeu a roupa do rei de Roma.\n" * 20, encoding="utf-8")
    session = TinyLiveTrainingSession(text_file=text_file, device="cpu", seed=123)

    assert session.tokenizer.vocab_size == 50257
    assert session.model_config.vocab_size == 50257


def test_tiny_live_training_session_step_updates_parameters(tmp_path) -> None:
    text_file = tmp_path / "tiny.txt"
    text_file.write_text("O rato roeu a roupa do rei de Roma.\n" * 20, encoding="utf-8")
    session = TinyLiveTrainingSession(
        text_file=text_file,
        device="cpu",
        seed=123,
        tokenizer_name="byte-debug",
    )
    before = next(session.model.parameters()).detach().clone()

    result = session.step()

    after = next(session.model.parameters()).detach()
    assert result.step == 1
    assert result.loss > 0
    assert result.grad_norm > 0
    assert not torch.allclose(before, after)


def test_tiny_live_generation_session_adds_one_token_and_reset_is_deterministic(tmp_path) -> None:
    tokenizer = ByteTokenizer()
    model = GPTLanguageModel(tiny_config())
    checkpoint = CheckpointManager(tmp_path)
    checkpoint_path = checkpoint.save(
        "last.pt",
        model,
        optimizer=None,
        scheduler=None,
        train_state=TrainState(),
        model_config=tiny_config(),
        training_config=TrainingConfig(max_steps=1, warmup_steps=1),
    )
    session = TinyLiveGenerationSession(
        checkpoint_path,
        prompt="O rato",
        device="cpu",
        tokenizer_name="byte-debug",
    )
    initial_length = session.generated.size(1)

    first = session.step()
    session.reset("O rato")
    second = session.step()

    assert session.generated.size(1) == initial_length + 1
    assert first.chosen_id == second.chosen_id
    assert first.accumulated_text == second.accumulated_text
    assert tokenizer.decode(session.generated[0].tolist()) == second.accumulated_text


def test_tiny_live_generation_session_rejects_incompatible_checkpoint(tmp_path) -> None:
    model = GPTLanguageModel(tiny_config())
    checkpoint = CheckpointManager(tmp_path)
    checkpoint_path = checkpoint.save(
        "last.pt",
        model,
        optimizer=None,
        scheduler=None,
        train_state=TrainState(),
        model_config=tiny_config(),
        training_config=TrainingConfig(max_steps=1, warmup_steps=1),
    )

    with pytest.raises(ValueError, match="Checkpoint/tokenizer incompatibilidade"):
        TinyLiveGenerationSession(checkpoint_path, prompt="O rato", device="cpu", tokenizer_name="gpt2")
