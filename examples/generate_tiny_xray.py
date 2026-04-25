"""Generate text while showing each next-token decision."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

from srp_gpt2.config import ModelConfig
from srp_gpt2.inference.sampler import Sampler, SamplingConfig
from srp_gpt2.model.gpt import GPTLanguageModel
from srp_gpt2.training.checkpoint import CheckpointManager
from srp_gpt2.xray import (
    attention_markdown,
    build_xray_tokenizer,
    inspect_logits,
    logit_markdown,
    summarize_attention_focus,
    token_text,
    write_jsonl_event,
    xray_tokenizer_label,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Didactic tiny GPT generation with xray output.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--strategy", choices=["greedy", "sample"], default="greedy")
    parser.add_argument("--tokenizer", choices=["gpt2", "byte-debug"], default="gpt2")
    parser.add_argument("--show-top-k", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--attention", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    run(args)


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device)
    checkpoint = CheckpointManager.load(args.checkpoint, map_location=device)
    model_config = ModelConfig(**checkpoint["model_config"])
    model = GPTLanguageModel(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    tokenizer = build_xray_tokenizer(args.tokenizer)
    if model_config.vocab_size != tokenizer.vocab_size:
        raise ValueError(
            "Checkpoint/tokenizer incompatibilidade: "
            f"checkpoint vocab_size={model_config.vocab_size}, "
            f"tokenizer {xray_tokenizer_label(args.tokenizer)} vocab_size={tokenizer.vocab_size}. "
            "Use o mesmo tokenizer usado no treino ou retreine o checkpoint."
        )
    sampling = SamplingConfig(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    sampler = Sampler(sampling)
    out_dir = args.out_dir or args.checkpoint.parent / "xray"
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "generation_trace.md"
    events_path = out_dir / "generation_events.jsonl"
    events_path.unlink(missing_ok=True)

    token_ids = tokenizer.encode(args.prompt)
    if not token_ids:
        token_ids = [tokenizer.eos_token_id]
    generated = torch.tensor([token_ids], dtype=torch.long, device=device)
    sections = [
        "# Tiny GPT generation trace",
        f"checkpoint={args.checkpoint}",
        f"prompt: `{args.prompt}`",
        f"tokenizer: {xray_tokenizer_label(args.tokenizer)}",
        f"sampling: temperature={args.temperature} top_k={args.top_k} top_p={args.top_p} "
        f"repetition_penalty={args.repetition_penalty}",
    ]
    write_jsonl_event(
        events_path,
        {
            "event": "start",
            "checkpoint": str(args.checkpoint),
            "prompt": args.prompt,
            "sampling": sampling,
        },
    )

    print("== Generation xray ==")
    print(f"prompt: {console_text(args.prompt)}")
    for step in range(1, args.max_new_tokens + 1):
        context = generated[:, -model.config.block_size :]
        with torch.no_grad():
            output = model(context)
            next_logits = output.logits[:, -1, :]
            generated_for_sampling = generated if args.strategy == "sample" else None
            inspection = inspect_logits(
                next_logits,
                tokenizer,
                top_k=args.show_top_k,
                sampling=sampling if args.strategy == "sample" else None,
                generated=generated_for_sampling,
            )
            if args.strategy == "greedy":
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                next_token = sampler.sample(next_logits, generated)
        generated = torch.cat((generated, next_token), dim=1)
        chosen_id = int(next_token.item())
        chosen_text = token_text(tokenizer, chosen_id)
        accumulated = tokenizer.decode(generated[0].tolist())

        event = {
            "event": "generation_step",
            "step": step,
            "strategy": args.strategy,
            "context": tokenizer.decode(context[0].tolist()),
            "chosen_id": chosen_id,
            "chosen_text": chosen_text,
            "accumulated": accumulated,
            "inspection": inspection,
        }
        step_sections = [
            f"## Step {step}",
            f"context: `{event['context']}`",
            f"chosen: `{chosen_text}` token_id={chosen_id}",
            "### Top candidatos",
            logit_markdown(inspection),
            "### Texto acumulado",
            "```text\n" + accumulated + "\n```",
        ]
        if args.attention:
            focus = summarize_attention_focus(model, context, tokenizer)
            event["attention"] = focus
            step_sections.extend(["### Atencao causal do ultimo token", attention_markdown(focus)])
        sections.append("\n\n".join(step_sections))
        write_jsonl_event(events_path, event)

        print(
            f"step={step:02d} chosen={console_text(chosen_text)!r} id={chosen_id} "
            f"confidence={inspection.confidence:.3f} entropy={inspection.entropy:.3f}"
        )
        if tokenizer.eos_token_id is not None and chosen_id == tokenizer.eos_token_id:
            break

    trace_path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    print("\n== Final text ==")
    print(console_text(tokenizer.decode(generated[0].tolist())))
    print(f"\ntrace: {trace_path}")
    print(f"events: {events_path}")


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
