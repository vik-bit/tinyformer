"""Greedy generation speed with vs without the KV cache (same tokens out, checked).

GPT-2 small (124M, weights converted from HF) and, if given, a trained
checkpoint. Batch 1, fp32, CPU; median of 3 repeats per setting.
  python scripts/bench_kvcache.py [runs/ablate/rope_mha_v4096_s0/model.pt]
"""
import json
import statistics as st
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tinyformer.model import GPT, Config  # noqa: E402


def bench(model, prompt_len, new, reps=3):
    torch.manual_seed(0)
    x = torch.randint(0, model.cfg.vocab_size, (1, prompt_len))
    out, times = {}, {}
    for cache in (True, False):
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            y = model.generate(x, new, use_cache=cache)
            ts.append(time.perf_counter() - t0)
        out[cache], times[cache] = y, st.median(ts)
    assert torch.equal(out[True], out[False]), "cache changed the generated tokens"
    return {"prompt": prompt_len, "new_tokens": new, "s_cache": times[True], "s_nocache": times[False],
            "tok_per_s_cache": new / times[True], "tok_per_s_nocache": new / times[False],
            "speedup": times[False] / times[True]}


def main():
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    rows = {}
    from transformers import GPT2LMHeadModel
    from tinyformer.convert import gpt_from_hf
    gpt2 = gpt_from_hf(GPT2LMHeadModel.from_pretrained("gpt2"))
    rows["gpt2_124M"] = [bench(gpt2, p, n) for p, n in ((16, 64), (16, 256), (512, 128))]
    if len(sys.argv) > 1:
        ck = torch.load(sys.argv[1], map_location="cpu")
        m = GPT(Config(**ck["config"]))
        m.load_state_dict(ck["model"])
        m.eval()
        rows[f"trained_{sum(p.numel() for p in m.parameters()) / 1e6:.1f}M"] = [bench(m, 8, 112)]
    for k, v in rows.items():
        for r in v:
            print(k, {a: round(b, 3) if isinstance(b, float) else b for a, b in r.items()})
    Path("results").mkdir(exist_ok=True)
    Path("results/kvcache.json").write_text(json.dumps({"device": "CPU i7-1255U, 4 threads, fp32, batch 1",
                                                        "torch": torch.__version__, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
