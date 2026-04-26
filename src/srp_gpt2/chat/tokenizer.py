"""Chat tokenizer wrapper.

Single responsibility: take any base tokenizer that follows
:class:`TokenizerProtocol` and expose it as a tokenizer that *also* knows
about chat special tokens (``<|im_start|>``, ``<|im_end|>``, ...) without
modifying the base tokenizer at all.

The wrapper reserves fresh IDs above ``base.vocab_size`` for each special
token. Encoding plain text is delegated to the base tokenizer; the
specials are inserted by :class:`ChatJsonlDataset` and
:class:`ChatSession` directly via :meth:`special_id`.
"""

from __future__ import annotations

from srp_gpt2.chat.special_tokens import ChatSpecialTokens
from srp_gpt2.data.tokenizer import TokenizerProtocol


class ChatTokenizer:
    """Wrap a base tokenizer and reserve IDs for chat special tokens."""

    def __init__(
        self,
        base: TokenizerProtocol,
        specials: ChatSpecialTokens | None = None,
    ) -> None:
        self.base = base
        self.specials = specials or ChatSpecialTokens()
        base_vocab = base.vocab_size
        self._special_to_id: dict[str, int] = {
            piece: base_vocab + offset
            for offset, piece in enumerate(self.specials.as_list())
        }
        self._id_to_special: dict[int, str] = {
            i: piece for piece, i in self._special_to_id.items()
        }

    # --- Vocabulary --------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return self.base.vocab_size + len(self._special_to_id)

    @property
    def eos_token_id(self) -> int:
        """For chat models the end-of-turn marker plays the role of EOS."""
        return self._special_to_id[self.specials.im_end]

    # --- Special access ----------------------------------------------

    def special_id(self, piece: str) -> int:
        return self._special_to_id[piece]

    def is_special(self, token_id: int) -> bool:
        return token_id in self._id_to_special

    # --- Plain encode/decode -----------------------------------------

    def encode(self, text: str) -> list[int]:
        """Encode plain text (no special tokens are inserted)."""
        return list(self.base.encode(text))

    def decode(self, token_ids: list[int]) -> str:
        """Decode a mixed sequence; specials are rendered verbatim."""
        out: list[str] = []
        buffer: list[int] = []
        for tok in token_ids:
            if tok in self._id_to_special:
                if buffer:
                    out.append(self.base.decode(buffer))
                    buffer = []
                out.append(self._id_to_special[tok])
            else:
                buffer.append(tok)
        if buffer:
            out.append(self.base.decode(buffer))
        return "".join(out)
