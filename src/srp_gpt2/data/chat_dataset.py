"""Supervised fine-tuning (SFT) dataset for chat conversations.

Single responsibility: read a JSONL file where each line is a conversation
of the form ``{"messages": [{"role": ..., "content": ...}, ...]}`` and
produce next-token training examples whose loss is computed **only** over
the assistant's tokens.

Loss masking is implemented in the standard PyTorch way: positions that
should be ignored are set to ``-100`` in the ``targets`` tensor, which
is the default ``ignore_index`` of ``F.cross_entropy``.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from srp_gpt2.chat.template import ChatMessage, ChatMLTemplate

IGNORE_INDEX = -100


class ChatJsonlDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """JSONL-backed chat dataset with assistant-only loss masking."""

    def __init__(
        self,
        path: str | Path,
        template: ChatMLTemplate,
        block_size: int,
        pad_token_id: int = 0,
    ) -> None:
        self.path = Path(path)
        self.template = template
        self.block_size = block_size
        self.pad_token_id = pad_token_id
        self.examples = self._load_examples()

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids, target_ids = self.examples[index]
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
        )

    # --- Internals ----------------------------------------------------

    def _load_examples(self) -> list[tuple[list[int], list[int]]]:
        examples: list[tuple[list[int], list[int]]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                conversation = json.loads(line)
                messages = [ChatMessage(**m) for m in conversation["messages"]]
                example = self._build_example(messages)
                if example is not None:
                    examples.append(example)
        if not examples:
            raise ValueError(f"no chat conversations found in {self.path}")
        return examples

    def _build_example(
        self, messages: list[ChatMessage]
    ) -> tuple[list[int], list[int]] | None:
        # Flatten template segments into a token stream + per-token supervision flag.
        tokens: list[int] = []
        supervised: list[bool] = []
        for segment in self.template.render_for_training(messages):
            tokens.extend(segment.token_ids)
            supervised.extend([segment.supervised] * len(segment.token_ids))

        # Causal LM: predict tokens[1:] from tokens[:-1].
        # A position contributes to the loss only if its *target* token was supervised.
        if len(tokens) < 2:
            return None
        input_ids = tokens[:-1]
        target_ids = [
            tok if sup else IGNORE_INDEX
            for tok, sup in zip(tokens[1:], supervised[1:], strict=True)
        ]

        # Truncate or right-pad to fixed block_size so a vanilla DataLoader can batch.
        if len(input_ids) >= self.block_size:
            input_ids = input_ids[: self.block_size]
            target_ids = target_ids[: self.block_size]
        else:
            pad = self.block_size - len(input_ids)
            input_ids = input_ids + [self.pad_token_id] * pad
            target_ids = target_ids + [IGNORE_INDEX] * pad

        # Drop conversations where nothing is supervised (no assistant turn).
        if all(t == IGNORE_INDEX for t in target_ids):
            return None
        return input_ids, target_ids
