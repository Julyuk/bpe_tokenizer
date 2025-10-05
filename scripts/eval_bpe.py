#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import List

# Ensure project root is on sys.path so 'src' is importable
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.bpetok.bpe import BPETokenizer, count_words


def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def evaluate_file(tokenizer: BPETokenizer, text_path: str):
    text = read_file(text_path)
    tokens = tokenizer.encode(text)
    token_count = len(tokens)
    words = count_words(text)
    bytes_count = len(text.encode("utf-8"))
    chars_count = len(text)

    fertility = token_count / max(1, words)
    bytes_per_token = bytes_count / max(1, token_count)
    chars_per_token = chars_count / max(1, token_count)

    return {
        "file": text_path,
        "tokens": token_count,
        "words": words,
        "bytes": bytes_count,
        "chars": chars_count,
        "tokens_per_word": fertility,
        "bytes_per_token": bytes_per_token,
        "chars_per_token": chars_per_token,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate BPE tokenizer on validation files")
    parser.add_argument("--model", required=True, help="Path to saved model JSON")
    parser.add_argument("--val-files", nargs="+", required=True, help="Validation text files")
    args = parser.parse_args()

    tokenizer = BPETokenizer.load(args.model)

    print("file\ttokens\twords\tbytes\tchars\ttokens/word\tbytes/token\tchars/token")
    totals = {
        "tokens": 0,
        "words": 0,
        "bytes": 0,
        "chars": 0,
    }
    for fp in args.val_files:
        m = evaluate_file(tokenizer, fp)
        print(
            f"{m['file']}\t{m['tokens']}\t{m['words']}\t{m['bytes']}\t{m['chars']}\t"
            f"{m['tokens_per_word']:.4f}\t{m['bytes_per_token']:.4f}\t{m['chars_per_token']:.4f}"
        )
        for k in totals:
            totals[k] += m[k]

    if totals["tokens"] > 0:
        agg_tpw = totals["tokens"] / max(1, totals["words"])
        agg_bpt = totals["bytes"] / totals["tokens"]
        agg_cpt = totals["chars"] / totals["tokens"]
        print(
            f"TOTAL\t{totals['tokens']}\t{totals['words']}\t{totals['bytes']}\t{totals['chars']}\t"
            f"{agg_tpw:.4f}\t{agg_bpt:.4f}\t{agg_cpt:.4f}"
        )


if __name__ == "__main__":
    main()
