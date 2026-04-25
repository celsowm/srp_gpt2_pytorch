"""Didactic inspection helpers for tiny training and generation demos."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from itertools import cycle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from srp_gpt2.config import ModelConfig, TrainingConfig
from srp_gpt2.data.dataset import TextFileLanguageModelDataset
from srp_gpt2.data.tokenizer import TokenizerProtocol
from srp_gpt2.data.tokenizer import build_tokenizer
from srp_gpt2.inference.sampler import Sampler, SamplingConfig
from srp_gpt2.model.gpt import GPTLanguageModel
from srp_gpt2.training.checkpoint import CheckpointManager
from srp_gpt2.training.optimizer import build_adamw
from srp_gpt2.training.scheduler import WarmupCosineScheduler


@dataclass(frozen=True)
class TopToken:
    token_id: int
    text: str
    probability: float
    logit: float


@dataclass(frozen=True)
class LogitInspection:
    entropy: float
    confidence: float
    top_tokens: list[TopToken]


@dataclass(frozen=True)
class AttentionFocus:
    layer: int
    query_position: int
    top_positions: list[tuple[int, str, float]]


@dataclass(frozen=True)
class TensorSummary:
    name: str
    shape: tuple[int, ...]
    mean: float
    std: float
    norm: float
    min_value: float
    max_value: float


@dataclass(frozen=True)
class BlockTrace:
    layer: int
    ln1: TensorSummary
    attention: TensorSummary
    residual_after_attention: TensorSummary
    ln2: TensorSummary
    mlp: TensorSummary
    residual_after_mlp: TensorSummary
    attention_map: list[list[float]]


@dataclass(frozen=True)
class TransformerTrace:
    input_tokens: list[dict[str, Any]]
    embeddings: TensorSummary
    blocks: list[BlockTrace]
    final_norm: TensorSummary
    logits: TensorSummary
    next_token: LogitInspection


@dataclass(frozen=True)
class LiveTrainingStep:
    step: int
    loss: float
    perplexity: float
    learning_rate: float
    grad_norm: float
    param_norm: float
    trace_before: TransformerTrace
    trace_after: TransformerTrace


@dataclass(frozen=True)
class LiveGenerationStep:
    step: int
    chosen_id: int
    chosen_text: str
    accumulated_text: str
    trace: TransformerTrace


TOKENIZER_CHOICES = ("gpt2", "byte-debug")


def resolve_xray_device(device: str = "auto") -> torch.device:
    """Resolve the best available xray device."""
    normalized = device.strip().lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if normalized == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise ValueError("MPS/Metal was requested but is not available")
    if normalized != "cpu" and normalized not in {"cuda", "mps"}:
        raise ValueError("--device must be one of: auto, cuda, mps, cpu")
    return torch.device(normalized)


def normalize_xray_tokenizer_name(tokenizer_name: str) -> str:
    """Normalize public xray tokenizer names or paths."""
    normalized = tokenizer_name.strip()
    if normalized.lower() in {"gpt2", "gpt-2", "bpe"}:
        return "gpt2"
    if normalized.lower() in {"byte", "byte-debug", "debug"}:
        return "byte"
    return normalized


def build_xray_tokenizer(tokenizer_name: str = "gpt2") -> TokenizerProtocol:
    """Build the tokenizer used by didactic xray tools."""
    normalized = normalize_xray_tokenizer_name(tokenizer_name)
    try:
        return build_tokenizer(normalized)
    except (ImportError, FileNotFoundError) as exc:
        if normalized == "gpt2":
            raise ImportError(
                "GPT-2 BPE tokenization requires tiktoken. Install it with: "
                'pip install tiktoken. '
                "The xray app will not silently fall back to byte-debug."
            ) from exc
        if normalized.endswith(".model") or Path(normalized).exists():
            raise FileNotFoundError(
                f"SentencePiece model not found at: {normalized}. "
                "Did you run scripts/train_tokenizer.py?"
            ) from exc
        raise


def xray_tokenizer_label(tokenizer_name: str) -> str:
    normalized = normalize_xray_tokenizer_name(tokenizer_name)
    if normalized == "gpt2":
        return "GPT-2 BPE"
    if normalized == "byte":
        return "byte/debug (nao representa tokenizacao GPT)"
    return f"Custom ({normalized})"


def token_text(tokenizer: TokenizerProtocol, token_id: int) -> str:
    """Decode one token into a compact printable label."""
    if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
        return "<eos>"
    if tokenizer.vocab_size == 257 and 0 <= token_id < 256:
        if token_id == 10:
            return "\\n"
        if token_id == 13:
            return "\\r"
        if token_id == 9:
            return "\\t"
        if 32 <= token_id <= 126:
            return chr(token_id)
        return f"0x{token_id:02X}"
    text = tokenizer.decode([token_id])
    if text == "\n":
        return "\\n"
    if text == "\r":
        return "\\r"
    if text == "\t":
        return "\\t"
    if not text:
        return "<empty>"
    return text


def display_token_text(tokenizer: TokenizerProtocol, token_id: int) -> str:
    """Return a UI-safe token label that makes spaces visible."""
    text = token_text(tokenizer, token_id)
    if tokenizer.vocab_size == 257:
        return text
    if text == "<eos>":
        return text
    visible = text.replace(" ", "▁")
    visible = visible.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    return visible or "<empty>"


def token_table(tokenizer: TokenizerProtocol, text: str, max_tokens: int = 32) -> list[dict[str, Any]]:
    """Return rows that show how text becomes token IDs."""
    token_ids = tokenizer.encode(text)[:max_tokens]
    rows: list[dict[str, Any]] = []
    cursor = 0
    for position, token_id in enumerate(token_ids):
        raw_text = token_text(tokenizer, token_id)
        raw_for_span = "" if raw_text == "<eos>" else raw_text
        start = cursor
        end = min(len(text), cursor + len(raw_for_span))
        cursor = end
        rows.append(
            {
                "position": position,
                "token_id": token_id,
                "text": display_token_text(tokenizer, token_id),
                "raw_text": raw_text,
                "char_start": start,
                "char_end": end,
            }
        )
    return rows


def shifted_token_table(
    tokenizer: TokenizerProtocol,
    x: torch.Tensor,
    y: torch.Tensor,
    max_tokens: int = 16,
) -> list[dict[str, Any]]:
    """Return rows that show the next-token training target."""
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be one-dimensional token sequences")
    rows = []
    for position, (input_id, target_id) in enumerate(zip(x.tolist(), y.tolist(), strict=True)):
        if position >= max_tokens:
            break
        rows.append(
            {
                "position": position,
                "input_id": int(input_id),
                "input_text": display_token_text(tokenizer, int(input_id)),
                "target_id": int(target_id),
                "target_text": display_token_text(tokenizer, int(target_id)),
            }
        )
    return rows


def inspect_logits(
    logits: torch.Tensor,
    tokenizer: TokenizerProtocol,
    top_k: int = 5,
    sampling: SamplingConfig | None = None,
    generated: torch.Tensor | None = None,
) -> LogitInspection:
    """Summarize a next-token distribution."""
    if logits.ndim == 1:
        batch_logits = logits[None, :]
    elif logits.ndim == 2:
        batch_logits = logits
    else:
        raise ValueError("logits must have shape [vocab] or [batch, vocab]")

    filtered_logits = batch_logits.float()
    if sampling is not None:
        sampler = Sampler(sampling)
        filtered_logits = filtered_logits / sampling.temperature
        filtered_logits = sampler._apply_repetition_penalty(filtered_logits, generated)
        filtered_logits = sampler._apply_top_k(filtered_logits)
        filtered_logits = sampler._apply_top_p(filtered_logits)

    probs = torch.softmax(filtered_logits[0], dim=-1)
    safe_probs = probs.clamp_min(1e-12)
    entropy = float(-(safe_probs * safe_probs.log()).sum().item())
    confidence = float(probs.max().item())
    top_values, top_indices = torch.topk(probs, k=min(top_k, probs.numel()))
    top_tokens = [
        TopToken(
            token_id=int(token_id),
            text=display_token_text(tokenizer, int(token_id)),
            probability=float(probability),
            logit=float(filtered_logits[0, int(token_id)].item()),
        )
        for probability, token_id in zip(top_values.tolist(), top_indices.tolist(), strict=True)
    ]
    return LogitInspection(entropy=entropy, confidence=confidence, top_tokens=top_tokens)


def gradient_norm(model: torch.nn.Module) -> float:
    """Return global L2 norm of gradients that are currently populated."""
    total_sq = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        norm = parameter.grad.detach().float().norm(2).item()
        total_sq += norm * norm
    return math.sqrt(total_sq)


def parameter_norm(model: torch.nn.Module) -> float:
    """Return global L2 norm of model parameters."""
    total_sq = 0.0
    for parameter in model.parameters():
        norm = parameter.detach().float().norm(2).item()
        total_sq += norm * norm
    return math.sqrt(total_sq)


def tensor_summary(name: str, tensor: torch.Tensor) -> TensorSummary:
    """Return compact scalar statistics for a tensor."""
    values = tensor.detach().float()
    return TensorSummary(
        name=name,
        shape=tuple(tensor.shape),
        mean=float(values.mean().item()),
        std=float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0,
        norm=float(values.norm(2).item()),
        min_value=float(values.min().item()),
        max_value=float(values.max().item()),
    )


@torch.no_grad()
def trace_transformer_forward(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    tokenizer: TokenizerProtocol,
    top_k: int = 5,
    max_attention_tokens: int = 24,
) -> TransformerTrace:
    """Run a didactic forward pass and capture each Transformer stage."""
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, time]")
    raw_model = getattr(model, "_orig_mod", model)
    was_training = raw_model.training
    raw_model.eval()
    try:
        x = raw_model.embeddings(input_ids)
        blocks: list[BlockTrace] = []
        for layer_idx, block in enumerate(raw_model.blocks):
            ln1 = block.ln_1(x)
            attention_weights = block.attention.attention_weights(ln1)
            attention_out = block.attention(ln1)
            after_attention = x + attention_out
            ln2 = block.ln_2(after_attention)
            mlp_out = block.feed_forward(ln2)
            x = after_attention + mlp_out
            blocks.append(
                BlockTrace(
                    layer=layer_idx,
                    ln1=tensor_summary("ln_1", ln1),
                    attention=tensor_summary("attention_out", attention_out),
                    residual_after_attention=tensor_summary(
                        "residual_after_attention", after_attention
                    ),
                    ln2=tensor_summary("ln_2", ln2),
                    mlp=tensor_summary("mlp_out", mlp_out),
                    residual_after_mlp=tensor_summary("residual_after_mlp", x),
                    attention_map=_compact_attention_map(attention_weights, max_attention_tokens),
                )
            )
        final_norm = raw_model.ln_f(x)
        logits = raw_model.lm_head(final_norm)
        next_logits = logits[:, -1, :]
        token_rows = [
            {
                "position": idx,
                "token_id": int(token_id),
                "text": display_token_text(tokenizer, int(token_id)),
                "raw_text": token_text(tokenizer, int(token_id)),
            }
            for idx, token_id in enumerate(input_ids[0].detach().cpu().tolist())
        ]
        return TransformerTrace(
            input_tokens=token_rows,
            embeddings=tensor_summary("embeddings", raw_model.embeddings(input_ids)),
            blocks=blocks,
            final_norm=tensor_summary("ln_f", final_norm),
            logits=tensor_summary("logits", logits),
            next_token=inspect_logits(next_logits, tokenizer, top_k=top_k),
        )
    finally:
        raw_model.train(was_training)


class TinyLiveTrainingSession:
    """Tiny text-file training session that advances one visible step at a time."""

    def __init__(
        self,
        text_file: str | Path,
        device: str | torch.device = "auto",
        seed: int = 1337,
        top_k: int = 5,
        tokenizer_name: str = "gpt2",
    ) -> None:
        self.text_file = Path(text_file)
        self.device = resolve_xray_device(str(device)) if isinstance(device, str) else device
        self.seed = seed
        self.top_k = top_k
        self.tokenizer_name = normalize_xray_tokenizer_name(tokenizer_name)
        self.tokenizer = build_xray_tokenizer(self.tokenizer_name)
        self.step_index = 0
        self._build_state()

    def reset(self) -> None:
        self.step_index = 0
        self._build_state()

    def step(self) -> LiveTrainingStep:
        x, y = next(self._batches)
        x = x.to(self.device)
        y = y.to(self.device)
        visible_x = x[:1]
        trace_before = trace_transformer_forward(
            self.model, visible_x, self.tokenizer, top_k=self.top_k
        )
        lr = self.scheduler.step(self.step_index)
        self.optimizer.zero_grad(set_to_none=True)
        output = self.model(x, y)
        if output.loss is None:
            raise RuntimeError("model returned no loss during xray training")
        loss = output.loss
        loss.backward()
        grad = gradient_norm(self.model)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.grad_clip)
        self.optimizer.step()
        self.step_index += 1
        trace_after = trace_transformer_forward(
            self.model, visible_x, self.tokenizer, top_k=self.top_k
        )
        loss_value = float(loss.item())
        return LiveTrainingStep(
            step=self.step_index,
            loss=loss_value,
            perplexity=safe_perplexity(loss_value),
            learning_rate=lr,
            grad_norm=grad,
            param_norm=parameter_norm(self.model),
            trace_before=trace_before,
            trace_after=trace_after,
        )

    def _build_state(self) -> None:
        _set_seed(self.seed)
        self.model_config = ModelConfig(
            vocab_size=self.tokenizer.vocab_size,
            block_size=48,
            n_layer=2,
            n_head=4,
            n_embd=96,
            dropout=0.0,
        )
        self.training_config = TrainingConfig(
            batch_size=8,
            max_steps=400,
            gradient_accumulation_steps=1,
            learning_rate=8e-4,
            min_learning_rate=8e-5,
            warmup_steps=10,
            grad_clip=1.0,
            num_workers=0,
            amp=False,
            compile=False,
            seed=self.seed,
        )
        dataset = TextFileLanguageModelDataset(
            self.text_file,
            tokenizer=self.tokenizer,
            block_size=self.model_config.block_size,
            stride=16,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
            drop_last=True,
        )
        self._batches = cycle(loader)
        self.model = GPTLanguageModel(self.model_config).to(self.device)
        self.optimizer = build_adamw(self.model, self.training_config)
        self.scheduler = WarmupCosineScheduler(self.optimizer, self.training_config)


class TinyLiveGenerationSession:
    """Checkpoint-backed generation session that advances one token at a time."""

    def __init__(
        self,
        checkpoint: str | Path,
        prompt: str,
        device: str | torch.device = "auto",
        strategy: str = "greedy",
        seed: int = 1337,
        top_k: int = 5,
        tokenizer_name: str = "gpt2",
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.prompt = prompt
        self.strategy = strategy
        self.seed = seed
        self.top_k = top_k
        self.device = resolve_xray_device(str(device)) if isinstance(device, str) else device
        self.tokenizer_name = normalize_xray_tokenizer_name(tokenizer_name)
        self.tokenizer = build_xray_tokenizer(self.tokenizer_name)
        self.sampling = SamplingConfig(temperature=0.8, top_k=20, top_p=0.9, repetition_penalty=1.05)
        self._load_model()
        self.reset(prompt)

    def reset(self, prompt: str | None = None) -> None:
        _set_seed(self.seed)
        if prompt is not None:
            self.prompt = prompt
        token_ids = self.tokenizer.encode(self.prompt)
        if not token_ids:
            token_ids = [self.tokenizer.eos_token_id]
        self.generated = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        self.step_index = 0

    def step(self) -> LiveGenerationStep:
        context = self.generated[:, -self.model.config.block_size :]
        trace = trace_transformer_forward(self.model, context, self.tokenizer, top_k=self.top_k)
        with torch.no_grad():
            output = self.model(context)
            next_logits = output.logits[:, -1, :]
            if self.strategy == "greedy":
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            elif self.strategy == "sample":
                next_token = Sampler(self.sampling).sample(next_logits, self.generated)
            else:
                raise ValueError("strategy must be 'greedy' or 'sample'")
        self.generated = torch.cat((self.generated, next_token), dim=1)
        self.step_index += 1
        chosen_id = int(next_token.item())
        return LiveGenerationStep(
            step=self.step_index,
            chosen_id=chosen_id,
            chosen_text=display_token_text(self.tokenizer, chosen_id),
            accumulated_text=self.tokenizer.decode(self.generated[0].tolist()),
            trace=trace,
        )

    def _load_model(self) -> None:
        checkpoint = CheckpointManager.load(self.checkpoint, map_location=self.device)
        model_config = ModelConfig(**checkpoint["model_config"])
        if model_config.vocab_size != self.tokenizer.vocab_size:
            raise ValueError(
                "Checkpoint/tokenizer incompatibilidade: "
                f"checkpoint vocab_size={model_config.vocab_size}, "
                f"tokenizer {xray_tokenizer_label(self.tokenizer_name)} "
                f"vocab_size={self.tokenizer.vocab_size}. "
                "Use o mesmo tokenizer usado no treino ou retreine o checkpoint."
            )
        self.model = GPTLanguageModel(model_config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()


def safe_perplexity(loss: float) -> float:
    if loss > 20:
        return float("inf")
    return math.exp(loss)


def _compact_attention_map(weights: torch.Tensor, max_tokens: int) -> list[list[float]]:
    mean_heads = weights[0].detach().float().mean(dim=0)
    if mean_heads.size(0) > max_tokens:
        mean_heads = mean_heads[-max_tokens:, -max_tokens:]
    return [
        [float(value) for value in row]
        for row in mean_heads.detach().cpu().tolist()
    ]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def causal_attention_maps(model: torch.nn.Module, input_ids: torch.Tensor) -> list[torch.Tensor]:
    """Return per-layer attention maps as CPU tensors shaped [heads, time, time]."""
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, time]")
    was_training = model.training
    model.eval()
    try:
        x = model.embeddings(input_ids)
        maps: list[torch.Tensor] = []
        for block in model.blocks:
            att_input = block.ln_1(x)
            weights = block.attention.attention_weights(att_input)
            maps.append(weights[0].detach().cpu())
            x = block(x)
    finally:
        model.train(was_training)
    return maps


def summarize_attention_focus(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    tokenizer: TokenizerProtocol,
    top_k: int = 4,
) -> list[AttentionFocus]:
    """Summarize which previous positions the final token attends to."""
    maps = causal_attention_maps(model, input_ids)
    token_ids = input_ids[0].detach().cpu().tolist()
    query_position = len(token_ids) - 1
    summaries: list[AttentionFocus] = []
    for layer_idx, layer_map in enumerate(maps):
        mean_heads = layer_map.mean(dim=0)
        row = mean_heads[query_position]
        values, indices = torch.topk(row, k=min(top_k, row.numel()))
        summaries.append(
            AttentionFocus(
                layer=layer_idx,
                query_position=query_position,
                top_positions=[
                    (int(position), display_token_text(tokenizer, int(token_ids[position])), float(weight))
                    for weight, position in zip(values.tolist(), indices.tolist(), strict=True)
                ],
            )
        )
    return summaries


def write_jsonl_event(path: str | Path, event: dict[str, Any]) -> None:
    """Append one JSON event to a JSONL file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_json_ready(event), ensure_ascii=False, sort_keys=True) + "\n")


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    """Render small diagnostic rows as a Markdown table."""
    if not rows:
        return ""
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def logit_markdown(inspection: LogitInspection) -> str:
    rows = [
        {
            "rank": rank,
            "token_id": token.token_id,
            "text": token.text,
            "prob": f"{token.probability:.4f}",
            "logit": f"{token.logit:.3f}",
        }
        for rank, token in enumerate(inspection.top_tokens, start=1)
    ]
    header = f"entropy={inspection.entropy:.3f} confidence={inspection.confidence:.3f}"
    return header + "\n" + markdown_table(rows, ["rank", "token_id", "text", "prob", "logit"])


def attention_markdown(focus: list[AttentionFocus]) -> str:
    rows: list[dict[str, Any]] = []
    for layer in focus:
        rows.extend(
            {
                "layer": layer.layer,
                "query_pos": layer.query_position,
                "key_pos": position,
                "text": text,
                "weight": f"{weight:.3f}",
            }
            for position, text, weight in layer.top_positions
        )
    return markdown_table(rows, ["layer", "query_pos", "key_pos", "text", "weight"])


def _json_ready(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "__dataclass_fields__"):
        return _json_ready(asdict(value))
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    return value
