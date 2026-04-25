"""Tokenizer abstractions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from srp_gpt2.data.bpe import SimpleSentencePieceBPE


class TokenizerProtocol(Protocol):
    """Minimal tokenizer interface used by datasets and generation."""

    vocab_size: int
    eos_token_id: int | None

    def encode(self, text: str) -> list[int]:
        """Convert text into token IDs."""

    def decode(self, token_ids: list[int]) -> str:
        """Convert token IDs into text."""


@dataclass
class ByteTokenizer:
    """Dependency-free UTF-8 byte tokenizer for demos and tests."""

    vocab_size: int = 257
    eos_token_id: int = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, token_ids: list[int]) -> str:
        data = bytes([token for token in token_ids if 0 <= token < 256])
        return data.decode("utf-8", errors="replace")


class GPT2BPETokenizer:
    """GPT-2 byte-pair tokenizer backed by ``tiktoken``.
    
    .. deprecated:: 0.1.0
        Use :class:`SentencePieceTokenizer` for custom datasets.
    """

    def __init__(self) -> None:
        try:
            import tiktoken
        except ImportError as exc:
            raise ImportError(
                "GPT2BPETokenizer requires tiktoken. Install with: pip install tiktoken"
            ) from exc
        self._encoding = tiktoken.get_encoding("gpt2")
        self.vocab_size = self._encoding.n_vocab
        self.eos_token_id = self._encoding.eot_token

    def encode(self, text: str) -> list[int]:
        return self._encoding.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, token_ids: list[int]) -> str:
        return self._encoding.decode(token_ids)


class SentencePieceTokenizer:
    """Tokenizer backed by a hand-made BPE implementation.
    
    This implementation is fully didactic and dependency-free.
    """

    def __init__(self, model_path: str | Path) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"BPE model not found at: {path}")

        self._bpe = SimpleSentencePieceBPE()
        self._bpe.load(path)
        self.vocab_size = self._bpe.vocab_size
        self.eos_token_id = self._bpe.eos_id

    def encode(self, text: str) -> list[int]:
        return self._bpe.encode(text, out_type=int)

    def decode(self, token_ids: list[int]) -> str:
        return self._bpe.decode(token_ids)


def build_tokenizer(name: str) -> TokenizerProtocol:
    normalized = name.strip()
    
    # Check if it's a known shortcut
    lower_name = normalized.lower()
    if lower_name == "byte":
        return ByteTokenizer()
    if lower_name in {"gpt2", "gpt-2", "bpe"}:
        return GPT2BPETokenizer()
    
    # Check for the default custom model path
    default_ptbr = Path("data/tokenizer/ptbr_32k.model")
    if lower_name == "ptbr":
        if default_ptbr.exists():
            return SentencePieceTokenizer(default_ptbr)
        raise FileNotFoundError(
            f"Default PT-BR model not found at {default_ptbr}. "
            "Train it first using 'python scripts/train_tokenizer.py'"
        )

    # Check if it's a path to a model
    path = Path(normalized)
    if path.exists():
        return SentencePieceTokenizer(path)
        
    # Fallback to ptbr if it's the default and file exists
    if normalized == "data/tokenizer/ptbr_32k.model" and default_ptbr.exists():
        return SentencePieceTokenizer(default_ptbr)
        
    raise ValueError(
        f"unknown tokenizer '{name}'. Use 'byte', 'gpt2', 'ptbr', or path to a .model file."
    )
