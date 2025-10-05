#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import List

# Ensure project root is on sys.path so 'src' is importable
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.bpetok.bpe import BPETrainer


def read_files(paths: List[str]) -> List[str]:
    texts: List[str] = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            texts.append(f.read())
    return texts


def main():
    parser = argparse.ArgumentParser(description="Train byte-level BPE tokenizer")
    parser.add_argument(
        "--train-files", nargs="+", required=True, help="Paths to training text files"
    )
    parser.add_argument(
        "--vocab-size", type=int, default=4096, help="Target vocabulary size (>=259)"
    )
    parser.add_argument(
        "--min-pair-count", type=int, default=2, help="Minimum pair frequency to merge"
    )
    parser.add_argument(
        "--model-out", type=str, default="bpe_model.json", help="Output model path"
    )

    args = parser.parse_args()
    texts = read_files(args.train_files)
    trainer = BPETrainer(vocab_size=args.vocab_size, min_pair_count=args.min_pair_count)
    tokenizer = trainer.train(texts)

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(args.model_out)
    print(f"Saved model to {args.model_out}")
    print(f"Vocab size: {len(tokenizer.model.id_to_bytes)}")
    print(f"Merges learned: {len(tokenizer.model.merges)}")


if __name__ == "__main__":
    main()
