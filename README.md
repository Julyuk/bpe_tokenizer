## Byte-fallback BPE Tokenizer (EN + UKR)

This repository contains a from-scratch, dependency-free Byte-Pair Encoding (BPE) tokenizer with byte-fallback suitable for multilingual text (English and Ukrainian). It trains on your own dataset (not the provided validation corpora), saves a compact JSON model, and provides scripts to evaluate fertility and compression metrics on validation sets.

### Features
- Byte-level base alphabet (256 bytes) → automatic byte-fallback for any Unicode text
- Deterministic BPE merges; reproducible training
- Save/load to a single JSON file
- Metrics: tokens/words ratio (fertility), bytes/token, chars/token
- Simple CLI: train, evaluate, interactive encode/decode

### Install
- Python 3.9+
- No external dependencies

### Layout
- `src/bpetok/bpe.py` — core BPETrainer and BPETokenizer
- `src/bpetok/__init__.py` — package export
- `scripts/train_bpe.py` — train and save model
- `scripts/eval_bpe.py` — compute metrics on validation files
- `scripts/interactive_bpe.py` — REPL to test encoding/decoding

### Quick Start
Train on your own corpus (not Brown or Laws used for validation):
```bash
python scripts/train_bpe.py \
  --train-files /path/to/your/train1.txt /path/to/your/train2.txt \
  --vocab-size 4096 \
  --model-out bpe_model.json
```

Evaluate on validation corpora (Brown EN/UKR, Laws EN/UKR):
```bash
python scripts/eval_bpe.py \
  --model bpe_model.json \
  --val-files /path/to/brown_en.txt /path/to/brown_uk.txt /path/to/laws_en.txt /path/to/laws_uk.txt
```

Interactive test:
```bash
python scripts/interactive_bpe.py --model bpe_model.json
```

### Notes
- Special tokens: `<pad>`=0, `<bos>`=1, `<eos>`=2
- Base vocab: 256 bytes + 3 specials = 259
- Target vocab size must be ≥ 259; merges = vocab_size − 259
- Byte-fallback: any unseen character is encoded via its UTF-8 bytes

### Why these choices?
- Byte base guarantees coverage for multilingual text without unknowns
- Simple, deterministic training improves reproducibility
- No external libs → transparency and educational value (KISS/DRY/YAGNI)
