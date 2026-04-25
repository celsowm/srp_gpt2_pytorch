"""Autoregressive text datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from srp_gpt2.data.tokenizer import TokenizerProtocol


class ParquetTextDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Parquet-backed Hugging Face dataset sliced into next-token examples."""

    def __init__(
        self,
        dataset_name: str,
        split: str,
        tokenizer: TokenizerProtocol,
        block_size: int,
        stride: int | None = None,
        text_column: str = "text",
        cache_dir: str | Path | None = None,
        append_eos: bool = True,
        **load_dataset_kwargs: Any,
    ) -> None:
        self.dataset_name = dataset_name
        self.split = split
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.stride = stride or block_size
        self.text_column = text_column
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.append_eos = append_eos
        self.load_dataset_kwargs = load_dataset_kwargs
        self.tokens = self._load_tokens()
        self.starts = self._build_starts()

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.starts[index]
        chunk = self.tokens[start : start + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

    def _load_tokens(self) -> list[int]:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "ParquetTextDataset requires datasets. Install with: pip install -e '.[hf]'"
            ) from exc

        dataset = load_dataset(
            self.dataset_name,
            split=self.split,
            cache_dir=str(self.cache_dir) if self.cache_dir is not None else None,
            **self.load_dataset_kwargs,
        )
        if self.text_column not in dataset.column_names:
            raise ValueError(
                f"dataset split '{self.split}' has no column '{self.text_column}'. "
                f"Available columns: {', '.join(dataset.column_names)}"
            )

        tokens: list[int] = []
        for row in dataset:
            text = row[self.text_column]
            if not isinstance(text, str) or not text.strip():
                continue
            tokens.extend(self.tokenizer.encode(text))
            if self.append_eos and self.tokenizer.eos_token_id is not None:
                tokens.append(self.tokenizer.eos_token_id)

        if len(tokens) <= self.block_size:
            raise ValueError(
                f"dataset has {len(tokens)} tokens, but block_size={self.block_size}; "
                "provide more text or reduce block_size"
            )
        return tokens

    def _build_starts(self) -> list[int]:
        max_start = len(self.tokens) - self.block_size - 1
        if max_start < 0:
            return []
        return list(range(0, max_start + 1, self.stride))
