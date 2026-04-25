"""Command-line interface for SRP GPT-2."""

from __future__ import annotations

import argparse
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from srp_gpt2.config import ModelConfig, ProjectConfig
from srp_gpt2.data.dataset import ParquetTextDataset
from srp_gpt2.data.tokenizer import TokenizerProtocol, build_tokenizer
from srp_gpt2.inference.generator import TextGenerator
from srp_gpt2.inference.sampler import SamplingConfig
from srp_gpt2.model.gpt import GPTLanguageModel
from srp_gpt2.training.checkpoint import CheckpointManager, TrainState
from srp_gpt2.training.optimizer import build_adamw
from srp_gpt2.training.scheduler import WarmupCosineScheduler
from srp_gpt2.training.trainer import Trainer


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SRP GPT-2 PyTorch")
    subparsers = parser.add_subparsers(required=True)

    train = subparsers.add_parser("train", help="train a GPT language model")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--hf-dataset", type=str, required=True)
    train.add_argument("--hf-train-split", type=str, default="train")
    train.add_argument("--hf-val-split", type=str, default="validation")
    train.add_argument("--hf-text-column", type=str, default="text")
    train.add_argument("--hf-cache-dir", type=Path, default=None)
    train.add_argument("--tokenizer", choices=["byte", "gpt2"], default="gpt2")
    train.add_argument("--out-dir", type=Path, required=True)
    train.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train.add_argument("--gpu-index", type=int, default=None)
    train.add_argument("--resume", type=Path, default=None)
    train.set_defaults(func=train_command)

    generate = subparsers.add_parser("generate", help="generate text from a checkpoint")
    generate.add_argument("--checkpoint", type=Path, required=True)
    generate.add_argument("--tokenizer", choices=["byte", "gpt2"], default="gpt2")
    generate.add_argument("--prompt", type=str, required=True)
    generate.add_argument("--max-new-tokens", type=int, default=100)
    generate.add_argument("--temperature", type=float, default=1.0)
    generate.add_argument("--top-k", type=int, default=None)
    generate.add_argument("--top-p", type=float, default=None)
    generate.add_argument("--repetition-penalty", type=float, default=1.0)
    generate.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    generate.add_argument("--gpu-index", type=int, default=None)
    generate.set_defaults(func=generate_command)

    params = subparsers.add_parser("param-count", help="print model parameter count")
    params.add_argument("--config", type=Path, required=True)
    params.set_defaults(func=param_count_command)

    return parser


def train_command(args: argparse.Namespace) -> None:
    project_config = ProjectConfig.from_yaml(args.config)
    set_seed(project_config.training.seed)
    device = resolve_device(args.device, args.gpu_index)

    tokenizer = build_tokenizer(args.tokenizer)
    model_config = _config_with_tokenizer_vocab(project_config.model, tokenizer.vocab_size)
    model = GPTLanguageModel(model_config)
    checkpoint = CheckpointManager.load(args.resume, map_location="cpu") if args.resume else None
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state"])
    if project_config.training.compile:
        model = torch.compile(model)  # type: ignore[assignment]

    train_dataset, val_dataset = build_datasets(args, tokenizer, model_config, project_config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=project_config.training.batch_size,
        shuffle=True,
        num_workers=project_config.training.num_workers,
        pin_memory=args.device.startswith("cuda"),
        drop_last=True,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=project_config.training.batch_size,
            shuffle=False,
            num_workers=project_config.training.num_workers,
            pin_memory=args.device.startswith("cuda"),
            drop_last=False,
        )
        if val_dataset is not None
        else None
    )

    optimizer = build_adamw(model, project_config.training)
    scheduler = WarmupCosineScheduler(optimizer, project_config.training)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        train_config=project_config.training,
        model_config=model_config,
        out_dir=args.out_dir,
        device=device,
    )
    if checkpoint is not None:
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            _move_optimizer_state(optimizer, trainer.device)
        if "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        trainer.train_state = TrainState(**checkpoint["train_state"])
    trainer.fit()


def generate_command(args: argparse.Namespace) -> None:
    device = resolve_device(args.device, args.gpu_index)
    checkpoint = CheckpointManager.load(args.checkpoint, map_location=device)
    model_config = ModelConfig(**checkpoint["model_config"])
    model = GPTLanguageModel(model_config)
    model.load_state_dict(checkpoint["model_state"])
    tokenizer = build_tokenizer(args.tokenizer)
    generator = TextGenerator(model, tokenizer, device=device)
    text = generator.generate(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        sampling=SamplingConfig(
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        ),
    )
    print(text)


def param_count_command(args: argparse.Namespace) -> None:
    project_config = ProjectConfig.from_yaml(args.config)
    model = GPTLanguageModel(project_config.model)
    params = model.count_parameters()
    print(f"parameters={params:,}")


def _config_with_tokenizer_vocab(model_config: ModelConfig, vocab_size: int) -> ModelConfig:
    if model_config.vocab_size != vocab_size:
        return replace(model_config, vocab_size=vocab_size)
    return model_config


def build_datasets(
    args: argparse.Namespace,
    tokenizer: TokenizerProtocol,
    model_config: ModelConfig,
    project_config: ProjectConfig,
) -> tuple[ParquetTextDataset, ParquetTextDataset]:
    train_dataset = ParquetTextDataset(
        args.hf_dataset,
        split=args.hf_train_split,
        tokenizer=tokenizer,
        block_size=model_config.block_size,
        stride=project_config.data.stride,
        text_column=args.hf_text_column,
        cache_dir=args.hf_cache_dir,
    )
    val_dataset = ParquetTextDataset(
        args.hf_dataset,
        split=args.hf_val_split,
        tokenizer=tokenizer,
        block_size=model_config.block_size,
        stride=project_config.data.stride,
        text_column=args.hf_text_column,
        cache_dir=args.hf_cache_dir,
    )
    return train_dataset, val_dataset


def resolve_device(device: str, gpu_index: int | None) -> str:
    if gpu_index is None:
        return device
    if gpu_index < 0:
        raise ValueError("--gpu-index must be >= 0")
    if device != "cuda":
        raise ValueError("--gpu-index can only be used with --device cuda")
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    if gpu_index >= torch.cuda.device_count():
        raise ValueError(
            f"--gpu-index {gpu_index} is out of range; "
            f"available CUDA devices: {torch.cuda.device_count()}"
        )
    return f"cuda:{gpu_index}"


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
