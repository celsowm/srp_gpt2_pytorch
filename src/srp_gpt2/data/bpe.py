# simple_sentencepiece_bpe.py
# Python 3.10+
# Somente biblioteca padrão. Otimizado com suporte de Claude (Incremental Updates + Heap-based Encoding).

from __future__ import annotations

import json
import re
import unicodedata
import heapq
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Literal, overload


SP_SPACE = " "

_TOKEN_RE = re.compile(
    rf"{SP_SPACE}?\w+|{SP_SPACE}?[^\w{SP_SPACE}]",
    re.UNICODE,
)


def normalize_text(text: str) -> str:
    """Normalização SentencePiece-style."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    return SP_SPACE + text.replace(" ", SP_SPACE)


def pretokenize(normalized: str) -> list[str]:
    """Divide o texto normalizado em unidades iniciais."""
    if not normalized:
        return []
    return _TOKEN_RE.findall(normalized)


@dataclass(frozen=True)
class BPEMerge:
    left: str
    right: str
    merged: str


class SimpleSentencePieceBPE:
    pad_id = 0
    unk_id = 1
    bos_id = 2
    eos_id = 3

    pad_piece = "<pad>"
    unk_piece = "<unk>"
    bos_piece = "<s>"
    eos_piece = "</s>"

    def __init__(self) -> None:
        self.piece_to_id: dict[str, int] = {}
        self.id_to_piece: dict[int, str] = {}
        self.merges: list[BPEMerge] = []
        self.char_vocab: set[str] = set()
        # Lookup rápido para encoding: (left, right) -> merged
        self._merge_lookup: dict[tuple[str, str], str] = {}
        # Ranking de merges para prioridade no heap
        self._merge_rank: dict[tuple[str, str], int] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.piece_to_id)

    def pieces(self) -> list[str]:
        return [self.id_to_piece[i] for i in range(len(self.id_to_piece))]

    def train(
        self,
        source: str | Path | Iterable[str],
        model_prefix: str | Path = "tokenizer_ptbr",
        vocab_size: int = 32_000,
        character_coverage: float = 0.9995,
        min_pair_freq: int = 2,
        max_piece_len: int = 64,
        max_lines: int | None = None,
        save_model: bool = True,
        verbose: bool = False,
        progress_callback: callable | None = None,
    ) -> None:
        self._validate_train_args(
            vocab_size=vocab_size,
            character_coverage=character_coverage,
            min_pair_freq=min_pair_freq,
            max_piece_len=max_piece_len,
        )

        word_freq: Counter[str] = Counter()
        char_freq: Counter[str] = Counter()

        for line in self._iter_texts(source, max_lines=max_lines):
            normalized = normalize_text(line)
            tokens = pretokenize(normalized)
            for token in tokens:
                word_freq[token] += 1
                char_freq.update(token)

        if not word_freq:
            raise ValueError("Corpus vazio ou sem texto útil.")

        char_vocab = self._select_chars(char_freq, character_coverage)
        char_vocab.add(SP_SPACE)

        vocab: list[str] = [
            self.pad_piece,
            self.unk_piece,
            self.bos_piece,
            self.eos_piece,
        ]
        seen = set(vocab)

        for ch, _ in char_freq.most_common():
            if ch in char_vocab and ch not in seen:
                vocab.append(ch)
                seen.add(ch)

        # Representação inicial: word_tuple -> freq
        splits: dict[tuple[str, ...], int] = {}
        for token, freq in word_freq.items():
            symbols = tuple(
                ch if ch in char_vocab else self.unk_piece
                for ch in token
            )
            splits[symbols] = splits.get(symbols, 0) + freq

        # --- Tabelas Incrementais ---
        pair_freq: Counter[tuple[str, str]] = Counter()
        # pair -> set de chaves em 'splits' que contêm esse par
        pair_index: dict[tuple[str, str], set[tuple[str, ...]]] = {}

        for symbols, freq in splits.items():
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pair_freq[pair] += freq
                if pair not in pair_index:
                    pair_index[pair] = set()
                pair_index[pair].add(symbols)

        merges: list[BPEMerge] = []
        unk = self.unk_piece

        while len(vocab) < vocab_size:
            best = self._best_pair_from_counter(
                pair_freq=pair_freq,
                seen=seen,
                min_pair_freq=min_pair_freq,
                max_piece_len=max_piece_len,
                unk=unk,
            )

            if best is None:
                break

            left, right, merged, freq = best

            # --- Atualização Incremental ---
            affected_symbols = pair_index.pop((left, right), set())
            new_splits_buffer: dict[tuple[str, ...], int] = {}

            for old_symbols in affected_symbols:
                count = splits.pop(old_symbols)
                
                # 1. Remover contagens antigas dos pares desta palavra
                for i in range(len(old_symbols) - 1):
                    p = (old_symbols[i], old_symbols[i + 1])
                    if p == (left, right): continue # Já removido do pair_index acima
                    
                    pair_freq[p] -= count
                    if pair_freq[p] <= 0:
                        del pair_freq[p]
                    if p in pair_index:
                        pair_index[p].discard(old_symbols)
                        if not pair_index[p]: del pair_index[p]

                # 2. Aplicar o merge
                new_symbols = _merge_tuple(old_symbols, left, right, merged)
                new_splits_buffer[new_symbols] = new_splits_buffer.get(new_symbols, 0) + count

            # 3. Re-inserir novas formas no estado global
            for new_symbols, total_new_freq in new_splits_buffer.items():
                # Se new_symbols já existir (colisão), removemos temporariamente para re-adicionar
                if new_symbols in splits:
                    existing_freq = splits[new_symbols]
                    for i in range(len(new_symbols) - 1):
                        p = (new_symbols[i], new_symbols[i + 1])
                        pair_freq[p] -= existing_freq
                        if pair_freq[p] <= 0: del pair_freq[p]
                        pair_index[p].discard(new_symbols)
                        if not pair_index[p]: del pair_index[p]
                    splits[new_symbols] += total_new_freq
                else:
                    splits[new_symbols] = total_new_freq
                
                # Adicionar novos pares ao index
                final_freq = splits[new_symbols]
                for i in range(len(new_symbols) - 1):
                    p = (new_symbols[i], new_symbols[i + 1])
                    pair_freq[p] += final_freq
                    if p not in pair_index: pair_index[p] = set()
                    pair_index[p].add(new_symbols)

            merges.append(BPEMerge(left=left, right=right, merged=merged))
            vocab.append(merged)
            seen.add(merged)

            if progress_callback:
                progress_callback({
                    "vocab_size": len(vocab),
                    "merges": len(merges),
                    "last_merge": (left, right, merged),
                    "freq": freq
                })

            if verbose and len(vocab) % 100 == 0:
                print(f"vocab={len(vocab)} merges={len(merges)} freq={freq}")

        # Finalizar estado
        self.char_vocab = char_vocab
        self.merges = merges
        self.piece_to_id = {piece: i for i, piece in enumerate(vocab)}
        self.id_to_piece = {i: piece for piece, i in self.piece_to_id.items()}
        self._merge_lookup = {(m.left, m.right): m.merged for m in self.merges}
        self._merge_rank = {pair: i for i, pair in enumerate(self._merge_lookup)}

        if save_model:
            self.save(model_prefix)

    @overload
    def encode(self, text: str, out_type: type[int] = int, add_bos: bool = False, add_eos: bool = False) -> list[int]: ...
    @overload
    def encode(self, text: str, out_type: type[str], add_bos: bool = False, add_eos: bool = False) -> list[str]: ...

    def encode(
        self,
        text: str,
        out_type: type[int] | type[str] = int,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int] | list[str]:
        self._ensure_ready()
        normalized = normalize_text(text)
        tokens = pretokenize(normalized)

        pieces: list[str] = []
        if add_bos: pieces.append(self.bos_piece)

        for token in tokens:
            symbols = [ch if ch in self.char_vocab else self.unk_piece for ch in token]
            pieces.extend(_apply_merges_fast(symbols, self._merge_lookup, self._merge_rank))

        if add_eos: pieces.append(self.eos_piece)

        if out_type is str: return pieces
        unk_id = self.unk_id
        pid = self.piece_to_id
        return [pid.get(p, unk_id) for p in pieces]

    def decode(self, ids_or_pieces: Iterable[int | str]) -> str:
        self._ensure_ready()
        pieces: list[str] = []
        for item in ids_or_pieces:
            piece = self.id_to_piece.get(item, self.unk_piece) if isinstance(item, int) else item
            if piece in {self.pad_piece, self.bos_piece, self.eos_piece}: continue
            pieces.append("?" if piece == self.unk_piece else piece)
        return "".join(pieces).replace(SP_SPACE, " ").strip()

    def save(self, model_prefix: str | Path) -> None:
        self._ensure_ready()
        model_path, vocab_path = self._model_paths(model_prefix)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model = {
            "type": "simple_sentencepiece_bpe",
            "space_symbol": SP_SPACE,
            "pad_id": self.pad_id, "unk_id": self.unk_id, "bos_id": self.bos_id, "eos_id": self.eos_id,
            "pad_piece": self.pad_piece, "unk_piece": self.unk_piece, "bos_piece": self.bos_piece, "eos_piece": self.eos_piece,
            "char_vocab": sorted(self.char_vocab),
            "pieces": self.pieces(),
            "merges": [asdict(m) for m in self.merges],
        }
        self._atomic_write_text(model_path, json.dumps(model, ensure_ascii=False, indent=2))
        vocab_lines = [f"{self.id_to_piece[i]}\t{-float(i)}" for i in sorted(self.id_to_piece)]
        self._atomic_write_text(vocab_path, "\n".join(vocab_lines) + "\n")

    def load(self, model_path: str | Path) -> None:
        data = json.loads(Path(model_path).read_text(encoding="utf-8"))
        if data.get("type") != "simple_sentencepiece_bpe":
            raise ValueError("Arquivo não parece ser um modelo simple_sentencepiece_bpe.")
        
        pieces = data["pieces"]
        self.pad_id, self.unk_id, self.bos_id, self.eos_id = data["pad_id"], data["unk_id"], data["bos_id"], data["eos_id"]
        self.pad_piece, self.unk_piece, self.bos_piece, self.eos_piece = data["pad_piece"], data["unk_piece"], data["bos_piece"], data["eos_piece"]
        self.char_vocab = set(data["char_vocab"])
        self.piece_to_id = {p: i for i, p in enumerate(pieces)}
        self.id_to_piece = {i: p for p, i in self.piece_to_id.items()}
        self.merges = [BPEMerge(l=item["left"], r=item["right"], merged=item["merged"]) if "l" not in item else BPEMerge(**item) for item in data["merges"]]
        # Normalização dos campos de merge caso venham de versões diferentes
        self.merges = [BPEMerge(left=m.left, right=m.right, merged=m.merged) for m in self.merges]
        self._merge_lookup = {(m.left, m.right): m.merged for m in self.merges}
        self._merge_rank = {pair: i for i, pair in enumerate(self._merge_lookup)}
        self._ensure_ready()

    @staticmethod
    def _validate_train_args(vocab_size: int, character_coverage: float, min_pair_freq: int, max_piece_len: int) -> None:
        if vocab_size < 10: raise ValueError("vocab_size < 10")
        if not 0.0 < character_coverage <= 1.0: raise ValueError("coverage invalid")
        if min_pair_freq < 1: raise ValueError("min_freq < 1")

    def _ensure_ready(self) -> None:
        if not self.piece_to_id: raise RuntimeError("Modelo não carregado")

    @staticmethod
    def _iter_texts(source: str | Path | Iterable[str], max_lines: int | None = None) -> Iterator[str]:
        count = 0
        if isinstance(source, (str, Path)):
            with Path(source).open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if max_lines is not None and count >= max_lines: break
                    count += 1
                    yield line.rstrip("\n")
        else:
            for line in source:
                if max_lines is not None and count >= max_lines: break
                count += 1
                yield line

    @staticmethod
    def _select_chars(char_freq: Counter[str], character_coverage: float) -> set[str]:
        total = sum(char_freq.values())
        if total == 0: return set()
        target, running, selected = total * character_coverage, 0, set()
        for ch, freq in char_freq.most_common():
            selected.add(ch)
            running += freq
            if running >= target: break
        return selected

    @staticmethod
    def _best_pair_from_counter(pair_freq: Counter[tuple[str, str]], seen: set[str], min_pair_freq: int, max_piece_len: int, unk: str) -> tuple[str, str, str, int] | None:
        best, best_key = None, None
        for (left, right), freq in pair_freq.items():
            if freq < min_pair_freq or left == unk or right == unk: continue
            merged = left + right
            if len(merged) > max_piece_len or merged in seen: continue
            key = (freq, len(merged), merged)
            if best_key is None or key > best_key:
                best, best_key = (left, right, merged, freq), key
        return best

    @staticmethod
    def _model_paths(model_prefix: str | Path) -> tuple[Path, Path]:
        p = Path(model_prefix)
        return p.parent / f"{p.name}.model", p.parent / f"{p.name}.vocab"

    @staticmethod
    def _atomic_write_text(path: Path, content: str) -> None:
        tmp = path.with_name(f"{path.name}.tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(path)


def _merge_tuple(symbols: tuple[str, ...], left: str, right: str, merged: str) -> tuple[str, ...]:
    res, i, n = [], 0, len(symbols)
    while i < n:
        if i + 1 < n and symbols[i] == left and symbols[i + 1] == right:
            res.append(merged); i += 2
        else:
            res.append(symbols[i]); i += 1
    return tuple(res)


def _apply_merges_fast(symbols: list[str], merge_lookup: dict[tuple[str, str], str], merge_rank: dict[tuple[str, str], int]) -> list[str]:
    n = len(symbols)
    if n <= 1: return symbols

    nxt = list(range(1, n + 1))
    prv = list(range(-1, n - 1))
    heap = []

    for i in range(n - 1):
        pair = (symbols[i], symbols[i + 1])
        if pair in merge_rank:
            heapq.heappush(heap, (merge_rank[pair], i))

    alive = [True] * n
    while heap:
        rank, i = heapq.heappop(heap)
        if not alive[i]: continue
        j = nxt[i]
        if j >= n or not alive[j]: continue

        pair = (symbols[i], symbols[j])
        if merge_rank.get(pair) != rank: continue

        # Apply merge
        symbols[i] = merge_lookup[pair]
        alive[j] = False

        # Relink
        nxt[i] = nxt[j]
        if nxt[j] < n: prv[nxt[j]] = i

        # Check neighbors
        if prv[i] >= 0:
            li = prv[i]
            p = (symbols[li], symbols[i])
            if p in merge_rank: heapq.heappush(heap, (merge_rank[p], li))
        if nxt[i] < n:
            ri = nxt[i]
            p = (symbols[i], symbols[ri])
            if p in merge_rank: heapq.heappush(heap, (merge_rank[p], i))

    return [symbols[i] for i in range(n) if alive[i]]
