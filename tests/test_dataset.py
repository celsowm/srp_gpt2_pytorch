from __future__ import annotations

import pytest
import torch

from srp_gpt2.data.dataset import ParquetTextDataset
from srp_gpt2.data.dataset import TextFileLanguageModelDataset
from srp_gpt2.data.tokenizer import ByteTokenizer


def test_parquet_text_dataset_loads_parquet(tmp_path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    pytest.importorskip("datasets")

    path = tmp_path / "train.parquet"
    table = pa.Table.from_pylist(
        [
            {"text": "Primeiro documento em portugues."},
            {"text": "Segundo documento para treino."},
        ]
    )
    pq.write_table(table, path)

    dataset = ParquetTextDataset(
        "parquet",
        split="train",
        tokenizer=ByteTokenizer(),
        block_size=8,
        stride=8,
        data_files=str(path),
    )

    x, y = dataset[0]
    assert len(dataset) > 0
    assert torch.equal(y[:-1], x[1:])
    assert x.shape == (8,)
    assert y.shape == (8,)


def test_text_file_language_model_dataset_loads_shifted_tokens(tmp_path) -> None:
    path = tmp_path / "tiny.txt"
    path.write_text("abcdef", encoding="utf-8")

    dataset = TextFileLanguageModelDataset(
        path,
        tokenizer=ByteTokenizer(),
        block_size=4,
        stride=2,
        append_eos=False,
    )

    x, y = dataset[0]
    assert len(dataset) == 1
    assert torch.equal(y[:-1], x[1:])
    assert x.tolist() == [97, 98, 99, 100]
    assert y.tolist() == [98, 99, 100, 101]


def test_text_file_language_model_dataset_rejects_too_little_text(tmp_path) -> None:
    path = tmp_path / "tiny.txt"
    path.write_text("abc", encoding="utf-8")

    with pytest.raises(ValueError, match="provide more text or reduce block_size"):
        TextFileLanguageModelDataset(
            path,
            tokenizer=ByteTokenizer(),
            block_size=4,
            append_eos=False,
        )
