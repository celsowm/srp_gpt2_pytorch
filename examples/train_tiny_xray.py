"""Train a tiny GPT with didactic xray snapshots."""

from __future__ import annotations

import argparse
import itertools
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from srp_gpt2.config import ModelConfig, TrainingConfig
from srp_gpt2.data.dataset import TextFileLanguageModelDataset
from srp_gpt2.data.tokenizer import TokenizerProtocol
from srp_gpt2.inference.generator import TextGenerator
from srp_gpt2.inference.sampler import SamplingConfig
from srp_gpt2.model.gpt import GPTLanguageModel
from srp_gpt2.training.checkpoint import CheckpointManager, TrainState
from srp_gpt2.training.optimizer import build_adamw
from srp_gpt2.training.scheduler import WarmupCosineScheduler
from srp_gpt2.xray import (
    attention_markdown,
    build_xray_tokenizer,
    gradient_norm,
    inspect_logits,
    logit_markdown,
    markdown_table,
    parameter_norm,
    shifted_token_table,
    summarize_attention_focus,
    token_table,
    write_jsonl_event,
    xray_tokenizer_label,
)


@dataclass(frozen=True)
class ModeSettings:
    max_steps: int
    batch_size: int
    block_size: int
    stride: int
    n_layer: int
    n_head: int
    n_embd: int
    learning_rate: float
    warmup_steps: int
    log_interval: int
    snapshot_interval: int
    sample_tokens: int


MODES = {
    "smoke": ModeSettings(
        max_steps=20,
        batch_size=8,
        block_size=32,
        stride=16,
        n_layer=2,
        n_head=2,
        n_embd=64,
        learning_rate=1e-3,
        warmup_steps=4,
        log_interval=1,
        snapshot_interval=5,
        sample_tokens=24,
    ),
    "classroom": ModeSettings(
        max_steps=80,
        batch_size=8,
        block_size=48,
        stride=24,
        n_layer=2,
        n_head=4,
        n_embd=96,
        learning_rate=8e-4,
        warmup_steps=8,
        log_interval=5,
        snapshot_interval=10,
        sample_tokens=40,
    ),
    "overfit": ModeSettings(
        max_steps=700,
        batch_size=12,
        block_size=64,
        stride=16,
        n_layer=3,
        n_head=4,
        n_embd=128,
        learning_rate=8e-4,
        warmup_steps=20,
        log_interval=10,
        snapshot_interval=50,
        sample_tokens=60,
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Didactic tiny GPT training with xray output.")
    parser.add_argument("--mode", choices=sorted(MODES), default="smoke")
    parser.add_argument("--text-file", type=Path, default=Path("data/tiny.txt"))
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/tiny_xray"))
    parser.add_argument("--prompt", type=str, default="O rato")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--tokenizer", choices=["gpt2", "byte-debug"], default="gpt2")
    parser.add_argument("--attention", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    run(args)


def run(args: argparse.Namespace) -> None:
    settings = MODES[args.mode]
    set_seed(args.seed)
    device = torch.device(args.device)
    tokenizer = build_xray_tokenizer(args.tokenizer)
    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=settings.block_size,
        n_layer=settings.n_layer,
        n_head=settings.n_head,
        n_embd=settings.n_embd,
        dropout=0.0,
    )
    train_config = TrainingConfig(
        batch_size=settings.batch_size,
        max_steps=settings.max_steps,
        gradient_accumulation_steps=1,
        learning_rate=settings.learning_rate,
        min_learning_rate=settings.learning_rate * 0.1,
        weight_decay=0.1,
        warmup_steps=settings.warmup_steps,
        eval_interval=settings.snapshot_interval,
        eval_batches=1,
        log_interval=settings.log_interval,
        save_interval=settings.snapshot_interval,
        num_workers=0,
        amp=False,
        compile=False,
        seed=args.seed,
    )

    dataset = TextFileLanguageModelDataset(
        args.text_file,
        tokenizer=tokenizer,
        block_size=model_config.block_size,
        stride=settings.stride,
    )
    loader = DataLoader(dataset, batch_size=train_config.batch_size, shuffle=True, drop_last=True)
    batches = itertools.cycle(loader)
    model = GPTLanguageModel(model_config).to(device)
    optimizer = build_adamw(model, train_config)
    scheduler = WarmupCosineScheduler(optimizer, train_config)
    checkpoints = CheckpointManager(args.out_dir)
    state = TrainState()
    xray_dir = args.out_dir / "xray"
    xray_dir.mkdir(parents=True, exist_ok=True)
    events_path = xray_dir / "events.jsonl"
    report_path = xray_dir / "report.md"
    events_path.unlink(missing_ok=True)

    first_x, first_y = dataset[0]
    report_sections = [
        "# Tiny GPT xray report",
        (
            f"mode={args.mode} text_file={args.text_file} device={device} "
            f"tokenizer={xray_tokenizer_label(args.tokenizer)}"
        ),
        "## Tokenizacao inicial",
        markdown_table(token_table(tokenizer, args.text_file.read_text(encoding='utf-8'), 24), ["position", "token_id", "text"]),
        "## Alvo do treino: x -> y",
        markdown_table(
            shifted_token_table(tokenizer, first_x, first_y, 16),
            ["position", "input_id", "input_text", "target_id", "target_text"],
        ),
    ]

    print(f"\n== Tokenizacao inicial ({xray_tokenizer_label(args.tokenizer)}) ==")
    print(report_sections[3])
    print("\n== Alvo do treino: x -> y ==")
    print(report_sections[5])
    write_jsonl_event(
        events_path,
        {
            "event": "start",
            "mode": args.mode,
            "text_file": str(args.text_file),
            "model_config": model_config,
            "training_config": train_config,
            "dataset_examples": len(dataset),
        },
    )

    for _ in range(train_config.max_steps):
        x, y = next(batches)
        x = x.to(device)
        y = y.to(device)
        lr = scheduler.step(state.step)
        optimizer.zero_grad(set_to_none=True)
        output = model(x, y)
        if output.loss is None:
            raise RuntimeError("model returned no loss during xray training")
        loss = output.loss
        loss.backward()
        grad = gradient_norm(model)
        if train_config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        optimizer.step()
        state.step += 1

        event = {
            "event": "train_step",
            "step": state.step,
            "loss": float(loss.item()),
            "perplexity": safe_perplexity(float(loss.item())),
            "lr": lr,
            "grad_norm": grad,
            "param_norm": parameter_norm(model),
        }
        write_jsonl_event(events_path, event)

        if state.step % train_config.log_interval == 0 or state.step == 1:
            print(
                f"step={state.step:04d} loss={event['loss']:.4f} "
                f"ppl={event['perplexity']:.2f} lr={lr:.2e} grad={grad:.2f}"
            )

        if state.step % settings.snapshot_interval == 0 or state.step == train_config.max_steps:
            section = snapshot(
                model=model,
                tokenizer=tokenizer,
                prompt=args.prompt,
                device=device,
                step=state.step,
                max_new_tokens=settings.sample_tokens,
                top_k=args.top_k,
                include_attention=args.attention,
            )
            report_sections.append(section)
            write_jsonl_event(events_path, {"event": "snapshot", "step": state.step, "markdown": section})
            checkpoints.save(
                "last.pt",
                model,
                optimizer,
                scheduler,
                state,
                model_config,
                train_config,
            )

    checkpoints.save("last.pt", model, optimizer, scheduler, state, model_config, train_config)
    report_path.write_text("\n\n".join(report_sections) + "\n", encoding="utf-8")
    print(f"\ncheckpoint: {args.out_dir / 'last.pt'}")
    print(f"xray events: {events_path}")
    print(f"xray report: {report_path}")


@torch.no_grad()
def snapshot(
    model: GPTLanguageModel,
    tokenizer: TokenizerProtocol,
    prompt: str,
    device: torch.device,
    step: int,
    max_new_tokens: int,
    top_k: int,
    include_attention: bool,
) -> str:
    model.eval()
    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        token_ids = [tokenizer.eos_token_id]
    input_ids = torch.tensor([token_ids[-model.config.block_size :]], dtype=torch.long, device=device)
    output = model(input_ids)
    next_logits = output.logits[:, -1, :]
    inspection = inspect_logits(next_logits, tokenizer, top_k=top_k)
    generator = TextGenerator(model, tokenizer, device=device)
    generated = generator.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        sampling=SamplingConfig(temperature=0.8, top_k=20, top_p=0.9, repetition_penalty=1.05),
    )
    sections = [
        f"## Snapshot step {step}",
        f"prompt: `{prompt}`",
        "### Proximos tokens provaveis",
        logit_markdown(inspection),
        "### Amostra gerada",
        "```text\n" + generated + "\n```",
    ]
    if include_attention:
        focus = summarize_attention_focus(model, input_ids, tokenizer)
        sections.extend(["### Atencao causal do ultimo token", attention_markdown(focus)])
    model.train()
    print(f"\n== Snapshot step {step} ==")
    print(logit_markdown(inspection))
    print("sample:", console_text(generated).replace("\n", "\\n"))
    return "\n\n".join(sections)


def safe_perplexity(loss: float) -> float:
    if loss > 20:
        return float("inf")
    return math.exp(loss)


def console_text(text: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="backslashreplace").decode(encoding, errors="replace")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
