"""Training-loop orchestration."""

from __future__ import annotations

import contextlib
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from srp_gpt2.config import ModelConfig, TrainingConfig
from srp_gpt2.training.checkpoint import CheckpointManager, TrainState
from srp_gpt2.training.scheduler import WarmupCosineScheduler


class Trainer:
    """Train and evaluate a GPT language model."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: WarmupCosineScheduler,
        train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None,
        train_config: TrainingConfig,
        model_config: ModelConfig,
        out_dir: str | Path,
        device: str | torch.device,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_config = train_config
        self.model_config = model_config
        self.device = torch.device(device)
        self.checkpoints = CheckpointManager(out_dir)
        self.train_state = TrainState()
        self.scaler = self._build_grad_scaler()

    def fit(self) -> TrainState:
        self.model.train()
        running_loss = 0.0
        tokens_seen = 0
        start_time = time.time()
        step_iterator = tqdm(
            total=self.train_config.max_steps,
            initial=self.train_state.step,
            desc="training",
            unit="step",
        )

        while self.train_state.step < self.train_config.max_steps:
            for x, y in self.train_loader:
                if self.train_state.step >= self.train_config.max_steps:
                    break

                lr = self.scheduler.step(self.train_state.step)
                loss = self._training_step(x, y)
                running_loss += loss
                tokens_seen += x.numel()

                if self._should_log():
                    elapsed = max(time.time() - start_time, 1e-9)
                    avg_loss = running_loss / self.train_config.log_interval
                    tok_per_sec = tokens_seen / elapsed
                    tqdm.write(
                        f"step={self.train_state.step:07d} "
                        f"loss={avg_loss:.4f} lr={lr:.3e} tok/s={tok_per_sec:.0f}"
                    )
                    running_loss = 0.0
                    tokens_seen = 0
                    start_time = time.time()

                if self._should_eval():
                    val_loss = self.evaluate()
                    tqdm.write(f"eval step={self.train_state.step:07d} val_loss={val_loss:.4f}")
                    self._save_best_if_needed(val_loss)
                    self.model.train()

                if self._should_save():
                    self._save_last()
                step_iterator.update(1)
            self.train_state.epoch += 1

        self._save_last()
        step_iterator.close()
        return self.train_state

    def evaluate(self) -> float:
        if self.val_loader is None:
            return float("nan")
        self.model.eval()
        losses: list[float] = []
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(self.val_loader):
                if batch_idx >= self.train_config.eval_batches:
                    break
                x = x.to(self.device)
                y = y.to(self.device)
                with self._autocast_context():
                    output = self.model(x, y)
                if output.loss is not None:
                    losses.append(float(output.loss.item()))
        return sum(losses) / max(1, len(losses))

    def _build_grad_scaler(self):  # type: ignore[no-untyped-def]
        enabled = self.train_config.amp and self.device.type == "cuda"
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            try:
                return torch.amp.GradScaler(self.device.type, enabled=enabled)
            except TypeError:
                pass
        return torch.cuda.amp.GradScaler(enabled=enabled)

    def _training_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        self.optimizer.zero_grad(set_to_none=True)
        micro_loss_total = 0.0
        chunks_x = x.chunk(self.train_config.gradient_accumulation_steps)
        chunks_y = y.chunk(self.train_config.gradient_accumulation_steps)
        accumulation_count = len(chunks_x)

        for micro_x, micro_y in zip(chunks_x, chunks_y, strict=True):
            micro_x = micro_x.to(self.device)
            micro_y = micro_y.to(self.device)
            with self._autocast_context():
                output = self.model(micro_x, micro_y)
                if output.loss is None:
                    raise RuntimeError("model returned no loss during training")
                loss = output.loss / accumulation_count
            micro_loss_total += float(loss.item())
            self.scaler.scale(loss).backward()

        if self.train_config.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.train_state.step += 1
        return micro_loss_total

    def _autocast_context(self):  # type: ignore[no-untyped-def]
        if self.train_config.amp and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _should_log(self) -> bool:
        return self.train_state.step > 0 and self.train_state.step % self.train_config.log_interval == 0

    def _should_eval(self) -> bool:
        return (
            self.val_loader is not None
            and self.train_state.step > 0
            and self.train_state.step % self.train_config.eval_interval == 0
        )

    def _should_save(self) -> bool:
        return self.train_state.step > 0 and self.train_state.step % self.train_config.save_interval == 0

    def _save_last(self) -> None:
        self.checkpoints.save(
            "last.pt",
            self.model,
            self.optimizer,
            self.scheduler,
            self.train_state,
            self.model_config,
            self.train_config,
        )

    def _save_best_if_needed(self, val_loss: float) -> None:
        if val_loss != val_loss:  # NaN check
            return
        best = self.train_state.best_val_loss
        if best is None or val_loss < best:
            self.train_state.best_val_loss = val_loss
            self.checkpoints.save(
                "best.pt",
                self.model,
                self.optimizer,
                self.scheduler,
                self.train_state,
                self.model_config,
                self.train_config,
            )
