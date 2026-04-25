"""Combine train/val files from separated corpus source folders."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        type=Path,
        help="folder containing train.txt and val.txt; pass multiple times",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/mixed"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_parts = read_parts(args.source, "train.txt")
    val_parts = read_parts(args.source, "val.txt")
    separator = "\n\n<|endoftext|>\n\n"

    train_text = separator.join(train_parts) + "\n"
    val_text = separator.join(val_parts) + "\n"
    train_path = args.out_dir / "train.txt"
    val_path = args.out_dir / "val.txt"
    train_path.write_text(train_text, encoding="utf-8")
    val_path.write_text(val_text, encoding="utf-8")

    print(f"sources={len(args.source)}")
    print(f"train={train_path} chars={len(train_text):,}")
    print(f"val={val_path} chars={len(val_text):,}")


def read_parts(sources: list[Path], filename: str) -> list[str]:
    parts: list[str] = []
    for source in sources:
        path = source / filename
        if not path.exists():
            raise FileNotFoundError(f"missing {path}")
        text = path.read_text(encoding="utf-8").strip()
        if text:
            parts.append(text)
    return parts


if __name__ == "__main__":
    main()
