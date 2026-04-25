"""Train a tiny byte-level GPT for a smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from srp_gpt2.config import DataConfig, ModelConfig, TrainingConfig
from srp_gpt2.data.dataset import TextFileDataset
from srp_gpt2.data.tokenizer import ByteTokenizer
from srp_gpt2.model.gpt import GPTLanguageModel
from srp_gpt2.training.optimizer import build_adamw
from srp_gpt2.training.scheduler import WarmupCosineScheduler
from srp_gpt2.training.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/tiny"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tokenizer = ByteTokenizer()
    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=64,
        n_layer=2,
        n_head=4,
        n_embd=128,
        dropout=0.0,
    )
    train_config = TrainingConfig(
        batch_size=16,
        max_steps=300,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        min_learning_rate=1e-4,
        warmup_steps=20,
        eval_interval=100,
        log_interval=20,
        save_interval=100,
        num_workers=0,
        amp=False,
    )
    data_config = DataConfig(stride=32)

    dataset = TextFileDataset(
        args.text,
        tokenizer,
        block_size=model_config.block_size,
        stride=data_config.stride,
    )
    loader = DataLoader(dataset, batch_size=train_config.batch_size, shuffle=True, drop_last=True)
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
