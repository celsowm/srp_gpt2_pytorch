.PHONY: test lint format tiny-data

test:
	pytest

lint:
	ruff check src tests examples

format:
	ruff format src tests examples

tiny-data:
	python scripts/download_tiny_corpus.py
