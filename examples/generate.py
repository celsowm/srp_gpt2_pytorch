"""Generate text from a saved checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from srp_gpt2.config import ModelConfig
from srp_gpt2.data.tokenizer import build_tokenizer
from srp_gpt2.inference.generator import TextGenerator
from srp_gpt2.inference.sampler import SamplingConfig
from srp_gpt2.model.gpt import GPTLanguageModel
from srp_gpt2.training.checkpoint import CheckpointManager


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", choices=["byte", "gpt2"], default="byte")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.85)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = CheckpointManager.load(args.checkpoint, map_location=args.device)
    model = GPTLanguageModel(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    tokenizer = build_tokenizer(args.tokenizer)
    generator = TextGenerator(model, tokenizer, device=args.device)
    print(
        generator.generate(
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            sampling=SamplingConfig(
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            ),
        )
    )


if __name__ == "__main__":
    main()
