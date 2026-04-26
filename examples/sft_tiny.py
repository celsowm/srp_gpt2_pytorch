"""Tiny supervised fine-tuning (SFT) smoke test.

Treats a tiny chat JSONL as the training corpus and trains a small GPT to
reproduce the assistant turns. Mirrors :mod:`examples.train_tiny` but for
chat conversations instead of plain text.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from srp_gpt2.chat.template import ChatMLTemplate
from srp_gpt2.chat.tokenizer import ChatTokenizer
from srp_gpt2.config import ModelConfig, TrainingConfig
from srp_gpt2.data.chat_dataset import ChatJsonlDataset
from srp_gpt2.data.tokenizer import ByteTokenizer
from srp_gpt2.model.gpt import GPTLanguageModel
from srp_gpt2.training.optimizer import build_adamw
from srp_gpt2.training.scheduler import WarmupCosineScheduler
from srp_gpt2.training.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, default=Path("data/sample_chat.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/sft_tiny"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # 1) Build a chat-aware tokenizer on top of the byte tokenizer.
    base_tokenizer = ByteTokenizer()
    tokenizer = ChatTokenizer(base_tokenizer)
    template = ChatMLTemplate(tokenizer)

    # 2) Tiny model whose vocab matches the chat tokenizer (base + specials).
    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=128,
        n_layer=2,
        n_head=4,
        n_embd=128,
        dropout=0.0,
    )
    train_config = TrainingConfig(
        batch_size=4,
        max_steps=200,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        min_learning_rate=1e-4,
        warmup_steps=10,
        eval_interval=100,
        log_interval=10,
        save_interval=100,
        num_workers=0,
        amp=False,
    )

    # 3) SFT dataset (assistant-only loss).
    dataset = ChatJsonlDataset(
        path=args.jsonl,
        template=template,
        block_size=model_config.block_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        drop_last=True,
    )

    # 4) Model + standard trainer (no special path for SFT; -100 in targets does the job).
    model = GPTLanguageModel(model_config)
    optimizer = build_adamw(model, train_config)
    scheduler = WarmupCosineScheduler(optimizer, train_config)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=loader,
        val_loader=None,
        train_config=train_config,
        model_config=model_config,
        out_dir=args.out_dir,
        device=args.device,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
