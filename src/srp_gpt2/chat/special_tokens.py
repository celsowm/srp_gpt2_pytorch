"""Reserved chat special tokens.

Single responsibility: hold the canonical names of the special tokens that
delimit chat turns. Following the ChatML convention used by OpenAI and
many open models:

    <|im_start|>{role}\n{content}<|im_end|>

The exact integer IDs are assigned at runtime by :class:`ChatTokenizer`,
because they depend on the underlying base tokenizer's vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChatSpecialTokens:
    """Names of the chat-turn delimiter tokens (ChatML)."""

    im_start: str = "<|im_start|>"
    im_end: str = "<|im_end|>"

    def as_list(self) -> list[str]:
        return [self.im_start, self.im_end]
