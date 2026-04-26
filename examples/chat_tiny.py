"""Interactive chat with a checkpoint produced by :mod:`examples.sft_tiny`."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from srp_gpt2.chat.template import ChatMLTemplate
from srp_gpt2.chat.tokenizer import ChatTokenizer
from srp_gpt2.config import ModelConfig
from srp_gpt2.data.tokenizer import ByteTokenizer
from srp_gpt2.inference.chat_session import ChatSession
from srp_gpt2.inference.sampler import SamplingConfig
from srp_gpt2.model.gpt import GPTLanguageModel
from srp_gpt2.training.checkpoint import CheckpointManager


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--system", type=str, default="Voce e um assistente util.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = CheckpointManager.load(args.checkpoint, map_location=args.device)
    model = GPTLanguageModel(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])

    tokenizer = ChatTokenizer(ByteTokenizer())
    template = ChatMLTemplate(tokenizer)
    session = ChatSession(
        model=model,
        tokenizer=tokenizer,
        template=template,
        device=args.device,
        system_prompt=args.system,
    )
    sampling = SamplingConfig(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    print("Chat iniciado. Ctrl+C para sair.\n")
    try:
        while True:
            user = input("user> ").strip()
            if not user:
                continue
            reply = session.reply(user, max_new_tokens=args.max_new_tokens, sampling=sampling)
            print(f"assistant> {reply}\n")
    except (EOFError, KeyboardInterrupt):
        print("\nbye.")


if __name__ == "__main__":
    main()
