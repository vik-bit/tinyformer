"""Byte-level BPE tokenizer, written from scratch.

Design follows GPT-2 / tiktoken so it can be checked against them:

* text is split into chunks with the GPT-2 pre-tokenisation regex, so merges
  never cross word / number / punctuation boundaries;
* every chunk starts as its UTF-8 bytes (ids 0..255), so any string round-trips
  and there is no unknown token;
* the vocabulary is a ``bytes -> rank`` table. Encoding repeatedly merges the
  adjacent pair whose concatenation has the lowest rank. That is the algorithm
  tiktoken uses, which is why ``Tokenizer.from_tiktoken("gpt2")`` reproduces
  tiktoken's ids exactly (tested in tests/test_tokenizer.py).

Training is the classic greedy algorithm (Sennrich et al. 2016) with
incremental pair-count updates, so each merge only touches the words that
contain the merged pair.
"""
from __future__ import annotations

import heapq
import json
from collections import Counter, defaultdict
from pathlib import Path

import regex as re

GPT2_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
EOT = "<|endoftext|>"


class Tokenizer:
    def __init__(self, ranks: dict[bytes, int], special: dict[str, int] | None = None,
                 pattern: str = GPT2_PATTERN):
        self.ranks = ranks
        self.special = dict(special or {})
        self.pattern = pattern
        self._pat = re.compile(pattern)
        self.decoder = {i: b for b, i in ranks.items()}
        for s, i in self.special.items():
            self.decoder[i] = s.encode("utf-8")
        self._special_pat = (re.compile("(" + "|".join(re.escape(s) for s in self.special) + ")")
                             if self.special else None)
        self._cache: dict[bytes, list[int]] = {}

    # ------------------------------------------------------------------ train
    @classmethod
    def train(cls, text: str, vocab_size: int, pattern: str = GPT2_PATTERN,
              special: tuple[str, ...] = (EOT,), verbose: bool = False) -> "Tokenizer":
        """Learn ``vocab_size - 256 - len(special)`` merges from ``text``."""
        n_merges = vocab_size - 256 - len(special)
        assert n_merges >= 0, "vocab_size must be >= 256 + number of special tokens"
        pat = re.compile(pattern)
        # special tokens never take part in merges
        pieces = re.split("(" + "|".join(re.escape(s) for s in special) + ")", text) if special else [text]
        word_freq: Counter[bytes] = Counter()
        for p in pieces:
            if p in special:
                continue
            word_freq.update(m.encode("utf-8") for m in pat.findall(p))

        words = [list(w) for w in word_freq]             # symbol ids per unique word
        freqs = list(word_freq.values())
        token_bytes: list[bytes] = [bytes([i]) for i in range(256)]

        pair_count: Counter[tuple[int, int]] = Counter()
        pair_where: defaultdict[tuple[int, int], set[int]] = defaultdict(set)
        for wi, w in enumerate(words):
            f = freqs[wi]
            for a, b in zip(w, w[1:]):
                pair_count[(a, b)] += f
                pair_where[(a, b)].add(wi)

        # max-heap with lazy invalidation; tie-break on lowest pair ids
        heap = [(-c, a, b) for (a, b), c in pair_count.items()]
        heapq.heapify(heap)
        for step in range(n_merges):
            best = None
            while heap:
                negc, a, b = heapq.heappop(heap)
                if pair_count.get((a, b), 0) == -negc and negc < 0:
                    best = (a, b)
                    break
            if best is None:
                break
            changed: set[tuple[int, int]] = set()
            new_id = len(token_bytes)
            token_bytes.append(token_bytes[best[0]] + token_bytes[best[1]])
            for wi in list(pair_where[best]):
                w, f = words[wi], freqs[wi]
                # remove this word's old pair counts
                for a, b in zip(w, w[1:]):
                    pair_count[(a, b)] -= f
                    if pair_count[(a, b)] <= 0:
                        del pair_count[(a, b)]
                    pair_where[(a, b)].discard(wi)
                    changed.add((a, b))
                # merge left to right
                out, i = [], 0
                while i < len(w):
                    if i + 1 < len(w) and w[i] == best[0] and w[i + 1] == best[1]:
                        out.append(new_id)
                        i += 2
                    else:
                        out.append(w[i])
                        i += 1
                words[wi] = out
                for a, b in zip(out, out[1:]):
                    pair_count[(a, b)] += f
                    pair_where[(a, b)].add(wi)
                    changed.add((a, b))
            pair_where.pop(best, None)
            for pr in changed:
                c = pair_count.get(pr, 0)
                if c > 0:
                    heapq.heappush(heap, (-c, pr[0], pr[1]))
            if verbose and (step + 1) % 500 == 0:
                print(f"merge {step + 1}/{n_merges}: {token_bytes[new_id]!r}")

        ranks = {b: i for i, b in enumerate(token_bytes)}
        spec = {s: len(ranks) + k for k, s in enumerate(special)}
        return cls(ranks, spec, pattern)

    # ----------------------------------------------------------------- encode
    def _bpe(self, chunk: bytes) -> list[int]:
        if chunk in self.ranks:
            return [self.ranks[chunk]]
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        parts = [bytes([b]) for b in chunk]
        ranks = self.ranks
        while len(parts) > 1:
            best_i, best_r = -1, None
            for i in range(len(parts) - 1):
                r = ranks.get(parts[i] + parts[i + 1])
                if r is not None and (best_r is None or r < best_r):
                    best_i, best_r = i, r
            if best_r is None:
                break
            parts[best_i:best_i + 2] = [parts[best_i] + parts[best_i + 1]]
        ids = [ranks[p] for p in parts]
        if len(self._cache) < 200_000:
            self._cache[chunk] = ids
        return ids

    def encode_ordinary(self, text: str) -> list[int]:
        out: list[int] = []
        for m in self._pat.findall(text):
            out.extend(self._bpe(m.encode("utf-8")))
        return out

    def encode(self, text: str, allow_special: bool = True) -> list[int]:
        if not (allow_special and self._special_pat):
            return self.encode_ordinary(text)
        out: list[int] = []
        for piece in self._special_pat.split(text):
            if piece in self.special:
                out.append(self.special[piece])
            elif piece:
                out.extend(self.encode_ordinary(piece))
        return out

    def decode_bytes(self, ids) -> bytes:
        return b"".join(self.decoder[int(i)] for i in ids)

    def decode(self, ids) -> str:
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    @property
    def vocab_size(self) -> int:
        return len(self.ranks) + len(self.special)

    @property
    def eot_id(self) -> int | None:
        return self.special.get(EOT)

    # ------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        items = sorted(self.ranks.items(), key=lambda kv: kv[1])
        Path(path).write_text(json.dumps({
            "pattern": self.pattern,
            "tokens": [b.hex() for b, _ in items],
            "special": self.special,
        }))

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        d = json.loads(Path(path).read_text())
        ranks = {bytes.fromhex(h): i for i, h in enumerate(d["tokens"])}
        return cls(ranks, d["special"], d["pattern"])

    @classmethod
    def from_tiktoken(cls, name: str = "gpt2") -> "Tokenizer":
        """Wrap tiktoken's published rank table in *this* encoder (reference check)."""
        import tiktoken
        enc = tiktoken.get_encoding(name)
        return cls(dict(enc._mergeable_ranks), dict(enc._special_tokens), enc._pat_str)
