"""Corpus helpers: TinyStories split, token files, random-window batches."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .tokenizer import EOT, Tokenizer

DATA = Path(__file__).resolve().parents[1] / "data"
RAW = DATA / "TinyStoriesV2-GPT4-valid.txt"


def tinystories_split(path: Path = RAW, val_frac: float = 0.1) -> tuple[str, str]:
    """Stories in file order; the first (possibly truncated) one is dropped.
    Last ``val_frac`` of stories is the held-out split. Deterministic."""
    stories = [s.strip() for s in path.read_text(encoding="utf-8").split(EOT)][1:]
    stories = [s for s in stories if s]
    n_val = int(len(stories) * val_frac)
    join = lambda xs: "".join(s + "\n" + EOT + "\n" for s in xs)  # noqa: E731
    return join(stories[:-n_val]), join(stories[-n_val:])


def encode_to_file(tok: Tokenizer, text: str, out: Path) -> int:
    ids = np.array(tok.encode(text), dtype=np.uint16 if tok.vocab_size < 65536 else np.uint32)
    ids.tofile(out)
    return len(ids)


class TokenStream:
    def __init__(self, path: Path, vocab_size: int):
        dtype = np.uint16 if vocab_size < 65536 else np.uint32
        self.ids = torch.from_numpy(np.fromfile(path, dtype=dtype).astype(np.int64))

    def __len__(self):
        return len(self.ids)

    def batch(self, B: int, T: int, gen: torch.Generator):
        ix = torch.randint(len(self.ids) - T - 1, (B,), generator=gen)
        x = torch.stack([self.ids[i:i + T] for i in ix])
        y = torch.stack([self.ids[i + 1:i + T + 1] for i in ix])
        return x, y

    def sequential(self, B: int, T: int, max_batches: int | None = None):
        """Non-overlapping windows over the whole stream (for eval)."""
        n = (len(self.ids) - 1) // T
        starts = torch.arange(n) * T
        for k in range(0, n, B):
            if max_batches is not None and k // B >= max_batches:
                return
            s = starts[k:k + B]
            yield (torch.stack([self.ids[i:i + T] for i in s]),
                   torch.stack([self.ids[i + 1:i + T + 1] for i in s]))
