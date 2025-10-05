#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import List, Dict

# Ensure project root is on sys.path so 'src' is importable
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.bpetok.bpe import BPETrainer, BPETokenizer, count_words, BASE_VOCAB_SIZE


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def evaluate_file(tokenizer: BPETokenizer, path: str) -> Dict[str, float]:
    text = read_text(path)
    tokens = tokenizer.encode(text)
    token_count = len(tokens)
    words = count_words(text)
    bytes_count = len(text.encode("utf-8"))
    chars_count = len(text)
    return {
        "file": path,
        "tokens": token_count,
        "words": words,
        "bytes": bytes_count,
        "chars": chars_count,
        "tokens_per_word": (token_count / max(1, words)),
        "bytes_per_token": (bytes_count / max(1, token_count)) if token_count else 0.0,
        "chars_per_token": (chars_count / max(1, token_count)) if token_count else 0.0,
    }


def build_training_code_snippet(vocab_size: int, min_pair_count: int, model_out: str, train_files: List[str]) -> str:
    quoted_files = " ".join(train_files)
    return f"""```bash
python scripts/train_bpe.py \
  --train-files {quoted_files} \
  --vocab-size {vocab_size} \
  --min-pair-count {min_pair_count} \
  --model-out {model_out}
```
```python
from src.bpetok.bpe import BPETrainer
texts = [open(p, 'r', encoding='utf-8').read() for p in {train_files!r}]
trainer = BPETrainer(vocab_size={vocab_size}, min_pair_count={min_pair_count})
tok = trainer.train(texts)
tok.save({model_out!r})
```
"""


def build_usage_code_snippet(model_path: str) -> str:
    return f"""```python
from src.bpetok.bpe import BPETokenizer

# Load trained model
tok = BPETokenizer.load({model_path!r})

# Tokenize and detokenize
text = "Hello, world! Привіт, світе!"
ids = tok.encode(text)
print(ids)
print(tok.decode(ids))
```
"""


def write_report(
    out_path: str,
    model_path: str,
    train_files: List[str],
    vocab_size: int,
    merges_count: int,
    min_pair_count: int,
    eval_rows: List[Dict[str, float]],
):
    # Build metrics table
    header = (
        "| файл | tokens | words | bytes | chars | tokens/word | bytes/token | chars/token |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    lines = []
    totals = {"tokens": 0, "words": 0, "bytes": 0, "chars": 0}
    for m in eval_rows:
        lines.append(
            f"| {Path(m['file']).name} | {int(m['tokens'])} | {int(m['words'])} | {int(m['bytes'])} | {int(m['chars'])} | "
            f"{m['tokens_per_word']:.4f} | {m['bytes_per_token']:.4f} | {m['chars_per_token']:.4f} |"
        )
        for k in totals:
            totals[k] += int(m[k])
    agg = ""
    if totals["tokens"] > 0:
        agg_tpw = totals["tokens"] / max(1, totals["words"])
        agg_bpt = totals["bytes"] / totals["tokens"]
        agg_cpt = totals["chars"] / totals["tokens"]
        agg = (
            f"\n**Сукупно:** tokens={totals['tokens']}, words={totals['words']}, bytes={totals['bytes']}, chars={totals['chars']}; "
            f"tokens/word={agg_tpw:.4f}, bytes/token={agg_bpt:.4f}, chars/token={agg_cpt:.4f}\n"
        )

    training_cmd = build_training_code_snippet(vocab_size, min_pair_count, model_path, train_files)
    usage_snippet = build_usage_code_snippet(model_path)

    content = f"""
# Звіт: Власний BPE токенізатор (byte-fallback, EN+UK)

## Навчальний датасет
- Файли (не з валідації):\n{''.join([f'- `{p}`\n' for p in train_files])}

## Розмір словника
- Вибраний розмір словника: **{vocab_size}**
- Кількість мерджів: **{merges_count}** (враховуючи базу {BASE_VOCAB_SIZE})

## Код навчання
{training_cmd}

## Код токенізації і детокенізації
{usage_snippet}

## Метрики на валідаційних наборах
{header}{chr(10).join(lines)}
{agg}

## Нотатки
- Базовою абеткою є 256 байтів → byte-fallback гарантує покриття будь-яких символів.
- Моделі/корпуси для валідації (Brown/Laws) не використовуються у навчанні.
- Метрики: fertility (tokens/word), bytes/token, chars/token.
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate Markdown report for BPE tokenizer")
    parser.add_argument("--val-files", nargs="+", required=True, help="Validation text files (Brown/Laws EN/UK)")
    parser.add_argument("--out", required=True, help="Output Markdown report path")
    parser.add_argument("--model", help="Existing model JSON path (skip training)")
    parser.add_argument("--train-files", nargs="+", help="Training text files (not validation)")
    parser.add_argument("--vocab-size", type=int, default=4096, help="Target vocab size if training")
    parser.add_argument("--min-pair-count", type=int, default=2, help="Minimum pair frequency for merges")
    parser.add_argument("--model-out", default="bpe_model.json", help="Where to save trained model")

    args = parser.parse_args()

    if args.model:
        model_path = args.model
        tok = BPETokenizer.load(model_path)
        vocab_size = len(tok.model.id_to_bytes)
        merges_count = len(tok.model.merges)
        train_files = args.train_files or []
    else:
        if not args.train_files:
            raise SystemExit("Either --model or --train-files must be provided.")
        texts = [read_text(p) for p in args.train_files]
        trainer = BPETrainer(vocab_size=args.vocab_size, min_pair_count=args.min_pair_count)
        tok = trainer.train(texts)
        Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
        tok.save(args.model_out)
        model_path = args.model_out
        vocab_size = len(tok.model.id_to_bytes)
        merges_count = len(tok.model.merges)
        train_files = args.train_files

    rows = [evaluate_file(tok, p) for p in args.val_files]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_report(
        out_path=args.out,
        model_path=model_path,
        train_files=train_files,
        vocab_size=vocab_size,
        merges_count=merges_count,
        min_pair_count=args.min_pair_count,
        eval_rows=rows,
    )
    print(f"Report written to {args.out}")
    print(f"Model: {model_path} | Vocab size: {vocab_size} | Merges: {merges_count}")


if __name__ == "__main__":
    main()
