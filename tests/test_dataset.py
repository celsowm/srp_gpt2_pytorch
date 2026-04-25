from __future__ import annotations

import pytest
import torch

from srp_gpt2.data.dataset import ParquetTextDataset
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
