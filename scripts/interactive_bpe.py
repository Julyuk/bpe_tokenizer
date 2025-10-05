#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path so 'src' is importable
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.bpetok.bpe import BPETokenizer


def main():
    parser = argparse.ArgumentParser(description="Interactive BPE encode/decode")
    parser.add_argument("--model", required=True, help="Path to saved model JSON")
    args = parser.parse_args()

    tok = BPETokenizer.load(args.model)
    print("Loaded model. Type a line to encode; empty line to exit.")
    while True:
        try:
            line = input("> ")
        except EOFError:
            break
        if not line:
            break
        ids = tok.encode(line)
        print(f"tokens: {ids}")
        print(f"decoded: {tok.decode(ids)}")


if __name__ == "__main__":
    main()
