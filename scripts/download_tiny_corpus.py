"""Create a tiny local corpus for smoke tests without internet access."""

from __future__ import annotations

from pathlib import Path

TEXT = """
O rato roeu a roupa do rei de Roma.
Transformers usam atenção causal para prever o próximo token.
Modelos GPT são decoders autoregressivos.
Este corpus é pequeno e serve apenas para testar o pipeline.
""".strip()


def main() -> None:
    out = Path("data/tiny.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text((TEXT + "\n") * 200, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
