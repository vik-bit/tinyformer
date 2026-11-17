"""Tokenizer + token files for an arbitrary <|endoftext|>-separated text file (used by the GPU notebook).

  python scripts/prepare_corpus.py corpus.txt --vocab 8192 --out data/big [--max_bytes 300000000]
Writes {out}/tok{V}.json, ts_v{V}_train.bin, ts_v{V}_val.bin (last 1% of stories = val).
The tokenizer is trained on at most --tok_bytes of the train split.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tinyformer.data import encode_to_file, tinystories_split  # noqa: E402
from tinyformer.tokenizer import Tokenizer  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("text")
    p.add_argument("--vocab", type=int, default=8192)
    p.add_argument("--out", required=True)
    p.add_argument("--val_frac", type=float, default=0.01)
    p.add_argument("--tok_bytes", type=int, default=50_000_000)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    train, val = tinystories_split(Path(a.text), a.val_frac)
    tok = Tokenizer.train(train[:a.tok_bytes], a.vocab)
    tok.save(out / f"tok{a.vocab}.json")
    print("val tokens", encode_to_file(tok, val, out / f"ts_v{a.vocab}_val.bin"))
    print("train tokens", encode_to_file(tok, train, out / f"ts_v{a.vocab}_train.bin"))


if __name__ == "__main__":
    main()
