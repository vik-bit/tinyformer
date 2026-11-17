"""Train BPE at several vocab sizes on the TinyStories train split and compare
compression on the held-out split with tiktoken's GPT-2 tokenizer.
Also writes data/tok{V}.json and data/ts_v{V}_{train,val}.bin for training.
"""
import json
import sys
import time
from pathlib import Path

import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tinyformer.data import DATA, encode_to_file, tinystories_split  # noqa: E402
from tinyformer.tokenizer import Tokenizer  # noqa: E402

VOCABS = [int(v) for v in sys.argv[1:]] or [512, 1024, 2048, 4096, 8192]


def main():
    train, val = tinystories_split()
    vb = len(val.encode("utf-8"))
    print(f"train {len(train.encode()):,} bytes, val {vb:,} bytes")
    gpt2 = tiktoken.get_encoding("gpt2")
    t0 = time.perf_counter()
    n_ref = len(gpt2.encode(val, allowed_special="all"))
    ref_t = time.perf_counter() - t0
    mine_gpt2 = Tokenizer.from_tiktoken("gpt2")
    t0 = time.perf_counter()
    n_mine_gpt2 = len(mine_gpt2.encode(val))
    mine_gpt2_t = time.perf_counter() - t0
    rows = [{"tokenizer": "tiktoken gpt2 (reference)", "vocab": gpt2.n_vocab, "val_tokens": n_ref,
             "bytes_per_token": vb / n_ref, "encode_MB_per_s": vb / ref_t / 1e6},
            {"tokenizer": "tinyformer encoder + GPT-2 ranks", "vocab": mine_gpt2.vocab_size,
             "val_tokens": n_mine_gpt2, "bytes_per_token": vb / n_mine_gpt2,
             "encode_MB_per_s": vb / mine_gpt2_t / 1e6}]
    for V in VOCABS:
        t0 = time.perf_counter()
        tok = Tokenizer.train(train, V)
        tt = time.perf_counter() - t0
        tok.save(DATA / f"tok{V}.json")
        t0 = time.perf_counter()
        n_val = encode_to_file(tok, val, DATA / f"ts_v{V}_val.bin")
        et = time.perf_counter() - t0
        n_train = encode_to_file(tok, train, DATA / f"ts_v{V}_train.bin")
        assert tok.decode(tok.encode(val[:20000])) == val[:20000]
        rows.append({"tokenizer": f"tinyformer BPE trained on TinyStories", "vocab": V, "val_tokens": n_val,
                     "train_tokens": n_train, "bytes_per_token": vb / n_val,
                     "train_seconds": tt, "encode_MB_per_s": vb / et / 1e6})
        print(rows[-1])
    out = {"corpus": "TinyStoriesV2-GPT4-valid.txt, 90/10 story split", "val_bytes": vb,
           "same_ids_as_tiktoken": n_mine_gpt2 == n_ref, "rows": rows}
    Path("results").mkdir(exist_ok=True)
    Path("results/tokenizer.json").write_text(json.dumps(out, indent=2))
    for r in rows:
        print(f"{r['tokenizer']:40s} V={r['vocab']:6d} bytes/tok={r['bytes_per_token']:.3f}")


if __name__ == "__main__":
    main()
