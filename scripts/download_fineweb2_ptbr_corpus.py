"""Stream Portuguese FineWeb2 documents and keep Brazilian-domain samples."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from urllib.parse import urlparse

BLANK_LINES = re.compile(r"\n{3,}")
WHITESPACE = re.compile(r"[ \t]+")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/fineweb_ptbr"))
    parser.add_argument("--max-docs", type=int, default=50_000)
    parser.add_argument("--max-chars", type=int, default=200_000_000)
    parser.add_argument("--min-chars", type=int, default=500)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument(
        "--allow-all-portuguese",
        action="store_true",
        help="do not restrict to .br URLs; useful when you want PT generally, not PT-BR-ish",
    )
    args = parser.parse_args()

    if args.max_docs <= 0:
        raise ValueError("--max-docs must be positive")
    if args.max_chars <= 0:
        raise ValueError("--max-chars must be positive")
    if not 0 < args.val_ratio < 1:
        raise ValueError("--val-ratio must be between 0 and 1")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    docs = collect_documents(args)
    if not docs:
        raise RuntimeError("no FineWeb2 documents were collected")

    write_outputs(args.out_dir, docs, args.val_ratio)


def collect_documents(args: argparse.Namespace) -> list[dict[str, str]]:
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install Hugging Face datasets: python -m pip install datasets") from exc

    dataset = load_dataset(
        "HuggingFaceFW/fineweb-2",
        name="por_Latn",
        split="train",
        streaming=True,
    )
    if args.shuffle_buffer > 0:
        dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    docs: list[dict[str, str]] = []
    total_chars = 0
    scanned = 0
    kept = 0

    for item in dataset:
        scanned += 1
        url = str(item.get("url", ""))
        if not args.allow_all_portuguese and not is_brazilian_url(url):
            continue

        text = clean_text(str(item.get("text", "")))
        if len(text) < args.min_chars:
            continue

        docs.append({"text": text, "url": url, "id": str(item.get("id", ""))})
        total_chars += len(text)
        kept += 1

        if kept % 1000 == 0:
            print(f"scanned={scanned:,} kept={kept:,} chars={total_chars:,}")
        if kept >= args.max_docs or total_chars >= args.max_chars:
            break

    print(f"scanned={scanned:,}")
    print(f"kept={kept:,}")
    print(f"chars={total_chars:,}")
    return docs


def is_brazilian_url(url: str) -> bool:
    hostname = urlparse(url).hostname
    return hostname is not None and hostname.lower().rstrip(".").endswith(".br")


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(WHITESPACE.sub(" ", line).strip() for line in text.splitlines())
    return BLANK_LINES.sub("\n\n", text).strip()


def write_outputs(out_dir: Path, docs: list[dict[str, str]], val_ratio: float) -> None:
    raw_path = out_dir / "documents.jsonl"
    with raw_path.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    val_count = max(1, round(len(docs) * val_ratio))
    train_docs = docs[:-val_count] or docs[:1]
    val_docs = docs[-val_count:]
    separator = "\n\n<|endoftext|>\n\n"

    train_text = separator.join(doc["text"] for doc in train_docs) + "\n"
    val_text = separator.join(doc["text"] for doc in val_docs) + "\n"
    (out_dir / "train.txt").write_text(train_text, encoding="utf-8")
    (out_dir / "val.txt").write_text(val_text, encoding="utf-8")

    print(f"raw={raw_path} docs={len(docs):,}")
    print(f"train={out_dir / 'train.txt'} chars={len(train_text):,}")
    print(f"val={out_dir / 'val.txt'} chars={len(val_text):,}")


if __name__ == "__main__":
    main()
