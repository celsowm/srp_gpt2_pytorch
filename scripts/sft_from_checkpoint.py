"""SFT script that loads a pre-trained checkpoint."""

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
from srp_gpt2.training.checkpoint import CheckpointManager
from srp_gpt2.training.optimizer import build_adamw
from srp_gpt2.training.scheduler import WarmupCosineScheduler
from srp_gpt2.training.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, default=Path("data/sft_sample.jsonl"))
    parser.add_argument("--pretrain-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/sft"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    # 1) Build a chat-aware tokenizer on top of the byte tokenizer.
    base_tokenizer = ByteTokenizer()
    tokenizer = ChatTokenizer(base_tokenizer)
    template = ChatMLTemplate(tokenizer)

    # 2) Load pre-trained model
    print(f"Loading pre-trained checkpoint from {args.pretrain_checkpoint}")
    checkpoint = CheckpointManager.load(args.pretrain_checkpoint, map_location="cpu")
    base_model_config = ModelConfig(**checkpoint["model_config"])

    # We need to adjust vocab_size if we added chat special tokens
    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=base_model_config.block_size,
        n_layer=base_model_config.n_layer,
        n_head=base_model_config.n_head,
        n_embd=base_model_config.n_embd,
        dropout=0.0,
    )

    train_config = TrainingConfig(
        batch_size=4,
        max_steps=args.steps,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        min_learning_rate=1e-5,
        warmup_steps=5,
        eval_interval=args.steps,
        log_interval=10,
        save_interval=args.steps,
        num_workers=0,
        amp=False,
    )

    # 3) SFT dataset
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

    # 4) Initialize model and load weights
    model = GPTLanguageModel(model_config)

    # Load base weights, ignoring the head and embedding if vocab size changed
    # In this case, vocab size definitely changed (257 -> 259)
    print("Loading base weights...")
    state_dict = checkpoint["model_state"]
    model_state = model.state_dict()

    # Filter out mismatching keys (embedding and head)
    # Actually, ByteTokenizer was 257. ChatTokenizer adds 2 specials = 259.
    for name, param in state_dict.items():
        if name in model_state:
            if param.shape == model_state[name].shape:
                model_state[name].copy_(param)
            elif "token_embedding" in name or "lm_head" in name:
                # Partial load for vocab expansion
                old_vocab_size = param.shape[0]
                new_vocab_size = model_state[name].shape[0]
                print(f"Partial loading {name}: {old_vocab_size} -> {new_vocab_size}")
                with torch.no_grad():
                    model_state[name][:old_vocab_size].copy_(param)
            else:
                print(f"Skipping {name} due to shape mismatch: {param.shape} vs {model_state[name].shape}")

    model.load_state_dict(model_state)

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
