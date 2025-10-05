"""
У проєкті реалізовано власний байт-рівневий BPE з byte‑fallback
для англ. та укр. мов: тренування (мержі частотних пар), збереження/завантаження
моделі в JSON, кодування/декодування, а також метрики (tokens/word,
bytes/token, chars/token) для валідації на незнайомих корпусах. Дизайн — простий,
прозорий, без зовнішніх залежностей (KISS/DRY/YAGNI).
"""

# This project implements a from-scratch byte-level BPE tokenizer with byte-fallback
# for English and Ukrainian. It includes training, saving/loading, encoding and
# decoding, plus evaluation scripts for fertility and compression metrics.

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


# Special token ids are fixed for simplicity and reproducibility
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
FIRST_BYTE_ID = 3  # bytes 0..255 map to ids 3..258
BASE_VOCAB_SIZE = FIRST_BYTE_ID + 256  # 259 total including 3 specials


@dataclass
class BPEModel:
    special_token_to_id: Dict[str, int]
    id_to_bytes: List[bytes]
    merges: List[Tuple[int, int, int]]  # (left_id, right_id, new_id) in order

    def to_json(self) -> str:
        payload = {
            "version": 1,
            "special_token_to_id": self.special_token_to_id,
            "id_to_bytes_hex": [b.hex() for b in self.id_to_bytes],
            "merges": self.merges,
        }
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def from_json(data: str) -> "BPEModel":
        obj = json.loads(data)
        id_to_bytes = [bytes.fromhex(h) for h in obj["id_to_bytes_hex"]]
        return BPEModel(
            special_token_to_id=obj["special_token_to_id"],
            id_to_bytes=id_to_bytes,
            merges=[tuple(x) for x in obj["merges"]],
        )


class BPETokenizer:
    def __init__(self, model: BPEModel):
        self.model = model
        # Build fast lookup structures
        self.pair_rank: Dict[Tuple[int, int], int] = {}
        for rank, (l, r, new_id) in enumerate(self.model.merges):
            self.pair_rank[(l, r)] = rank

    @staticmethod
    def byte_id(b: int) -> int:
        return FIRST_BYTE_ID + b

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        data = text.encode("utf-8")
        seq: List[int] = [self.byte_id(b) for b in data]
        if not seq:
            tokens: List[int] = []
        else:
            tokens = self._apply_merges(seq)
        if add_bos:
            tokens = [BOS_ID] + tokens
        if add_eos:
            tokens = tokens + [EOS_ID]
        return tokens

    def _apply_merges(self, seq: List[int]) -> List[int]:
        # Greedy BPE apply: repeatedly merge the best-ranked pair present
        if not seq or not self.pair_rank:
            return seq
        while True:
            best_pair = None
            best_rank = None
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                rank = self.pair_rank.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pair = pair
            if best_pair is None:
                break
            seq = self._merge_pair_in_seq(seq, best_pair)
        return seq

    @staticmethod
    def _merge_pair_in_seq(seq: List[int], pair: Tuple[int, int]) -> List[int]:
        l, r = pair
        new_seq: List[int] = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == l and seq[i + 1] == r:
                new_seq.append(BPETokenizer._new_id_for_pair_static(l, r))
                i += 2
            else:
                new_seq.append(seq[i])
                i += 1
        return new_seq

    @staticmethod
    def _new_id_for_pair_static(l: int, r: int) -> int:
        # Placeholder; this is replaced at runtime by monkey-patching per instance
        raise RuntimeError("_new_id_for_pair_static should be bound per instance")

    def _new_id_for_pair(self, l: int, r: int) -> int:
        rank = self.pair_rank[(l, r)]
        _, _, new_id = self.model.merges[rank]
        return new_id

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        out_bytes = bytearray()
        for tid in ids:
            if skip_special and tid in (PAD_ID, BOS_ID, EOS_ID):
                continue
            token_bytes = self.model.id_to_bytes[tid]
            out_bytes.extend(token_bytes)
        return out_bytes.decode("utf-8", errors="replace")

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.model.to_json())

    @staticmethod
    def load(path: str) -> "BPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            model = BPEModel.from_json(f.read())
        tok = BPETokenizer(model)
        # Bind static method to use instance-specific lookup without passing self in hot path
        def _bound_new_id(l: int, r: int, self_ref=tok):
            return self_ref._new_id_for_pair(l, r)
        BPETokenizer._new_id_for_pair_static = staticmethod(_bound_new_id)
        return tok


class BPETrainer:
    def __init__(self, vocab_size: int, min_pair_count: int = 2):
        if vocab_size < BASE_VOCAB_SIZE:
            raise ValueError(
                f"vocab_size must be >= {BASE_VOCAB_SIZE} (got {vocab_size})."
            )
        self.vocab_size = vocab_size
        self.min_pair_count = max(1, min_pair_count)

    def train(self, texts: Iterable[str]) -> BPETokenizer:
        # Initialize special tokens and byte tokens
        special_token_to_id = {"<pad>": PAD_ID, "<bos>": BOS_ID, "<eos>": EOS_ID}
        id_to_bytes: List[bytes] = [b"", b"", b""]  # specials carry empty bytes
        for b in range(256):
            id_to_bytes.append(bytes([b]))

        # Prepare corpus as sequences of byte-token ids
        sequences: List[List[int]] = []
        for text in texts:
            data = text.encode("utf-8")
            if not data:
                continue
            sequences.append([FIRST_BYTE_ID + b for b in data])

        merges: List[Tuple[int, int, int]] = []
        current_vocab_size = BASE_VOCAB_SIZE

        while current_vocab_size < self.vocab_size:
            pair_counts = self._count_pair_frequencies(sequences)
            if not pair_counts:
                break
            # Choose most frequent pair above threshold
            best_pair, best_count = None, 0
            for pair, cnt in pair_counts.items():
                if cnt >= self.min_pair_count and cnt > best_count:
                    best_pair, best_count = pair, cnt
            if best_pair is None:
                break
            new_id = current_vocab_size
            # Define new token bytes as concatenation of children
            l, r = best_pair
            new_bytes = id_to_bytes[l] + id_to_bytes[r]
            id_to_bytes.append(new_bytes)
            # Replace in all sequences
            total_replacements = 0
            for i, seq in enumerate(sequences):
                new_seq, rep = self._merge_pair_in_sequence(seq, best_pair, new_id)
                if rep:
                    sequences[i] = new_seq
                    total_replacements += rep
            if total_replacements == 0:
                # No effective change (should be rare due to counting), stop
                id_to_bytes.pop()
                break
            merges.append((l, r, new_id))
            current_vocab_size += 1
        model = BPEModel(
            special_token_to_id=special_token_to_id,
            id_to_bytes=id_to_bytes,
            merges=merges,
        )
        tok = BPETokenizer(model)
        # Bind static helper for performance in encode path
        def _bound_new_id(l: int, r: int, self_ref=tok):
            return self_ref._new_id_for_pair(l, r)
        BPETokenizer._new_id_for_pair_static = staticmethod(_bound_new_id)
        return tok

    @staticmethod
    def _count_pair_frequencies(sequences: List[List[int]]) -> Dict[Tuple[int, int], int]:
        counts: Dict[Tuple[int, int], int] = {}
        for seq in sequences:
            if len(seq) < 2:
                continue
            prev = seq[0]
            for cur in seq[1:]:
                pair = (prev, cur)
                counts[pair] = counts.get(pair, 0) + 1
                prev = cur
        return counts

    @staticmethod
    def _merge_pair_in_sequence(
        seq: List[int], pair: Tuple[int, int], new_id: int
    ):
        l, r = pair
        out: List[int] = []
        i = 0
        replacements = 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == l and seq[i + 1] == r:
                out.append(new_id)
                i += 2
                replacements += 1
            else:
                out.append(seq[i])
                i += 1
        return out, replacements


# Utility metric helpers used by eval script
_word_re = re.compile(r"\b\w+\b", flags=re.UNICODE)


def count_words(text: str) -> int:
    return len(_word_re.findall(text))
