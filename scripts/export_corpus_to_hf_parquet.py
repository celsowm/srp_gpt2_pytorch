"""Export text corpus files to Hugging Face-friendly Parquet shards."""

from __future__ import annotations

import argparse
from pathlib import Path

END_OF_TEXT = "<|endoftext|>"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-text", type=Path, required=True)
    parser.add_argument("--val-text", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("hf_dataset"))
    parser.add_argument("--dataset-name", default="srp-gpt2-ptbr-corpus")
    parser.add_argument("--source-name", default="mixed_ptbr")
    parser.add_argument("--shard-size-mb", type=int, default=128)
    parser.add_argument("--repo-id", default=None, help="optional Hugging Face dataset repo id")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    if args.shard_size_mb <= 0:
        raise ValueError("--shard-size-mb must be positive")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_count = export_split(args.train_text, args.out_dir, "train", args.source_name, args.shard_size_mb)
    val_count = export_split(args.val_text, args.out_dir, "validation", args.source_name, args.shard_size_mb)
    write_readme(args.out_dir, args.dataset_name, args.source_name, train_count, val_count)

    print(f"out={args.out_dir}")
    print(f"train_docs={train_count:,}")
    print(f"validation_docs={val_count:,}")

    if args.repo_id:
        upload_to_hub(args.out_dir, args.repo_id, args.private)


def export_split(
    text_path: Path,
    out_dir: Path,
    split: str,
    source_name: str,
    shard_size_mb: int,
) -> int:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("Install optional HF deps: python -m pip install -e '.[hf]'") from exc

    split_dir = out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    max_chars = shard_size_mb * 1024 * 1024
    docs = iter_documents(text_path)
    rows: list[dict[str, str]] = []
    shard_chars = 0
    shard_idx = 0
    total_docs = 0

    for doc_idx, text in enumerate(docs):
        row = {
            "id": f"{source_name}-{split}-{doc_idx:09d}",
            "text": text,
            "source": source_name,
            "split": split,
        }
        rows.append(row)
        shard_chars += len(text)
        total_docs += 1
        if shard_chars >= max_chars:
            write_shard(pq, pa, split_dir, split, shard_idx, rows)
            shard_idx += 1
            rows = []
            shard_chars = 0

    if rows:
        write_shard(pq, pa, split_dir, split, shard_idx, rows)
    return total_docs


def iter_documents(path: Path):
    text = path.read_text(encoding="utf-8")
    for part in text.split(END_OF_TEXT):
        doc = part.strip()
        if doc:
            yield doc


def write_shard(pq, pa, split_dir: Path, split: str, shard_idx: int, rows: list[dict[str, str]]) -> None:
    table = pa.Table.from_pylist(rows)
    path = split_dir / f"{split}-{shard_idx:05d}.parquet"
    pq.write_table(table, path, compression="zstd")
    print(f"wrote {path} rows={len(rows):,}")


def write_readme(
    out_dir: Path,
    dataset_name: str,
    source_name: str,
    train_count: int,
    val_count: int,
) -> None:
    readme = f"""---
configs:
- config_name: default
  data_files:
  - split: train
    path: train/*.parquet
  - split: validation
    path: validation/*.parquet
---

# {dataset_name}

Text corpus exported from `{source_name}` for causal language model training.

## Columns

- `id`: stable document identifier
- `text`: UTF-8 document text
- `source`: corpus source label
- `split`: train or validation

## Size

- train documents: {train_count:,}
- validation documents: {val_count:,}
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def upload_to_hub(out_dir: Path, repo_id: str, private: bool) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError("Install optional HF deps: python -m pip install -e '.[hf]'") from exc

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=out_dir,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload corpus parquet shards",
    )
    print(f"uploaded=https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
