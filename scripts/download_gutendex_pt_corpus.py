"""Download Portuguese books from Gutendex and build train/val text files."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE_URL = "https://gutendex.com/books/"
USER_AGENT = "srp-gpt2-pytorch/0.1 corpus-prep"

START_MARKER = re.compile(
    r"\*\*\*\s*START OF (?:TH(?:E|IS) )?PROJECT GUTENBERG.*?\*\*\*",
    re.IGNORECASE | re.DOTALL,
)
END_MARKER = re.compile(
    r"\*\*\*\s*END OF (?:TH(?:E|IS) )?PROJECT GUTENBERG.*?\*\*\*",
    re.IGNORECASE | re.DOTALL,
)
BAD_FILENAME_CHARS = re.compile(r'[\\/*?:"<>|]')
WHITESPACE = re.compile(r"[ \t]+")
BLANK_LINES = re.compile(r"\n{3,}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("dataset_livros_ptbr"))
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--max-books", type=int, default=50)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--min-chars", type=int, default=5_000)
    parser.add_argument(
        "--skip-title-regex",
        default=r"\b(?:biblia|bíblia|dicion[aá]rio|diccionario|dictionary)\b",
        help="case-insensitive regex for titles to skip",
    )
    args = parser.parse_args()

    if args.max_books <= 0:
        raise ValueError("--max-books must be positive")
    if not 0 < args.val_ratio < 1:
        raise ValueError("--val-ratio must be between 0 and 1")

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    books = download_books(
        raw_dir=args.raw_dir,
        max_books=args.max_books,
        delay=args.delay,
        min_chars=args.min_chars,
        skip_title_regex=args.skip_title_regex,
    )
    if not books:
        raise RuntimeError("no books were downloaded; try lowering --min-chars")

    train_text, val_text = split_books(books, args.val_ratio)
    train_path = args.out_dir / "train.txt"
    val_path = args.out_dir / "val.txt"
    train_path.write_text(train_text, encoding="utf-8")
    val_path.write_text(val_text, encoding="utf-8")

    print(f"books={len(books)}")
    print(f"train={train_path} chars={len(train_text):,}")
    print(f"val={val_path} chars={len(val_text):,}")


def download_books(
    raw_dir: Path,
    max_books: int,
    delay: float,
    min_chars: int,
    skip_title_regex: str,
) -> list[str]:
    url = f"{BASE_URL}?{urlencode({'languages': 'pt'})}"
    books: list[str] = []
    skip_title = re.compile(skip_title_regex, re.IGNORECASE) if skip_title_regex else None

    while url and len(books) < max_books:
        payload = fetch_json(url)
        for book in payload.get("results", []):
            if len(books) >= max_books:
                break

            text_url = choose_text_url(book.get("formats", {}))
            if text_url is None:
                continue

            book_id = book["id"]
            title = clean_filename(book["title"])[:80] or "sem_titulo"
            if skip_title is not None and skip_title.search(title):
                print(f"skipping: {title}")
                continue

            filepath = raw_dir / f"ID{book_id}_{title}.txt"

            if filepath.exists():
                text = filepath.read_text(encoding="utf-8")
                books.append(text)
                print(f"cached: {title}")
                continue

            print(f"downloading: {title}")
            text = clean_gutenberg_text(fetch_text(text_url))
            if len(text) >= min_chars:
                filepath.write_text(text, encoding="utf-8")
                books.append(text)
            time.sleep(delay)

        url = payload.get("next")

    return books


def fetch_json(url: str) -> dict[str, object]:
    return json.loads(fetch_bytes(url).decode("utf-8"))


def fetch_text(url: str) -> str:
    data = fetch_bytes(url)
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def fetch_bytes(url: str, retries: int = 3) -> bytes:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=30) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            wait_seconds = 5 * (attempt + 1)
            print(f"request failed, retrying in {wait_seconds}s: {exc}")
            time.sleep(wait_seconds)
    raise RuntimeError(f"failed to fetch {url}") from last_error


def choose_text_url(formats: dict[str, str]) -> str | None:
    candidates = [(mime, url) for mime, url in formats.items() if "text/plain" in mime]
    if not candidates:
        return None
    for mime, url in candidates:
        if "utf-8" in mime.lower():
            return url
    return candidates[0][1]


def clean_gutenberg_text(text: str) -> str:
    start = START_MARKER.search(text)
    end = END_MARKER.search(text)
    if start is not None:
        text = text[start.end() :]
    if end is not None:
        text = text[: end.start()]
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(WHITESPACE.sub(" ", line).strip() for line in text.splitlines())
    return BLANK_LINES.sub("\n\n", text).strip()


def clean_filename(value: str) -> str:
    return BAD_FILENAME_CHARS.sub("", value).strip()


def split_books(books: list[str], val_ratio: float) -> tuple[str, str]:
    val_count = max(1, round(len(books) * val_ratio))
    val_books = books[-val_count:]
    train_books = books[:-val_count] or books[:1]
    separator = "\n\n<|endoftext|>\n\n"
    return separator.join(train_books) + "\n", separator.join(val_books) + "\n"


if __name__ == "__main__":
    main()
