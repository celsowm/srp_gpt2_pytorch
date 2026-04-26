"""Chat templating.

Single responsibility: turn a list of role/content messages into a flat
sequence of token IDs *together with* a parallel mask that says which
positions belong to the assistant's reply (the only positions where we
want supervision during SFT).

The chosen format is a minimal ChatML:

    <|im_start|>system\n{content}<|im_end|>
    <|im_start|>user\n{content}<|im_end|>
    <|im_start|>assistant\n{content}<|im_end|>

Each role header (``<|im_start|>{role}\n``) is part of the *prompt* and
is **not** supervised. The assistant's body and the closing
``<|im_end|>`` of the assistant turn **are** supervised, so the model
learns both what to say and when to stop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from srp_gpt2.chat.tokenizer import ChatTokenizer


Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


@dataclass(frozen=True)
class ChatSegment:
    """A contiguous run of tokens with a single supervision flag."""

    token_ids: list[int]
    supervised: bool


class ChatMLTemplate:
    """Render chat conversations into token IDs + supervision flags."""

    def __init__(self, tokenizer: ChatTokenizer) -> None:
        self.tokenizer = tokenizer
        self._im_start = tokenizer.special_id(tokenizer.specials.im_start)
        self._im_end = tokenizer.special_id(tokenizer.specials.im_end)

    # --- Training-time rendering -------------------------------------

    def render_for_training(self, messages: list[ChatMessage]) -> list[ChatSegment]:
        """Produce alternating prompt/response segments for one conversation."""
        segments: list[ChatSegment] = []
        for msg in messages:
            header_ids = [self._im_start, *self.tokenizer.encode(f"{msg.role}\n")]
            body_ids = self.tokenizer.encode(msg.content)
            if msg.role == "assistant":
                # Header is context (not supervised); body + im_end are supervised.
                segments.append(ChatSegment(header_ids, supervised=False))
                segments.append(ChatSegment([*body_ids, self._im_end], supervised=True))
            else:
                segments.append(
                    ChatSegment([*header_ids, *body_ids, self._im_end], supervised=False)
                )
        return segments

    # --- Inference-time rendering ------------------------------------

    def render_for_generation(self, messages: list[ChatMessage]) -> list[int]:
        """Render the prompt and open the assistant turn for sampling."""
        ids: list[int] = []
        for msg in messages:
            ids.append(self._im_start)
            ids.extend(self.tokenizer.encode(f"{msg.role}\n"))
            ids.extend(self.tokenizer.encode(msg.content))
            ids.append(self._im_end)
        # Open the assistant's turn so the model continues from here.
        ids.append(self._im_start)
        ids.extend(self.tokenizer.encode("assistant\n"))
        return ids
