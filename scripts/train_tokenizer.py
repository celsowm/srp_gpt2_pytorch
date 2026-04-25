"""Script to train a hand-made BPE tokenizer on local text files."""

import argparse
import glob
import logging
from pathlib import Path

from srp_gpt2.data.bpe import SimpleSentencePieceBPE

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def train_tokenizer(
    input_pattern: str,
    model_prefix: str,
    vocab_size: int,
    verbose: bool = True,
) -> None:
    """Train a SimpleSentencePieceBPE model."""
    files = glob.glob(input_pattern)
    if not files:
        raise ValueError(f"No files found matching pattern: {input_pattern}")

    logger.info(f"Found {len(files)} files for training.")
    
    # Create output directory if it doesn't exist
    output_dir = Path(model_prefix).parent
    if str(output_dir) != "." and not output_dir.exists():
        output_dir.mkdir(parents=True)

    # We iterate texts using a generator to save memory
    def iter_all_files():
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        yield line
            except Exception as e:
                logger.warning(f"Could not read file {file_path}: {e}")

    logger.info(f"Training BPE model with vocab_size={vocab_size}...")
    
    bpe = SimpleSentencePieceBPE()
    bpe.train(
        source=iter_all_files(),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        verbose=verbose
    )
    
    logger.info(f"Training complete. Files saved: {model_prefix}.model and {model_prefix}.vocab")


def main():
    parser = argparse.ArgumentParser(description="Train a hand-made BPE tokenizer.")
    parser.add_argument(
        "--input", 
        type=str, 
        default="dataset_livros_ptbr/*.txt",
        help="Glob pattern for input text files."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="data/tokenizer/ptbr_32k",
        help="Prefix for the generated model files."
    )
    parser.add_argument(
        "--vocab_size", 
        type=int, 
        default=32000, 
        help="Vocabulary size."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show training progress."
    )

    args = parser.parse_args()
    
    try:
        train_tokenizer(
            input_pattern=args.input,
            model_prefix=args.output,
            vocab_size=args.vocab_size,
        )
    except Exception as e:
        logger.error(f"Failed to train tokenizer: {e}")
        exit(1)


if __name__ == "__main__":
    main()
