"""Load GPT-2 small into tinyformer.GPT and compare with Hugging Face.

Reports max |logit difference| over a set of prompts, top-1 agreement, and
whether greedy generations are token-identical (with and without KV cache).
Writes results/verify_gpt2.json.
"""
import json
import sys
import time
from pathlib import Path

import tiktoken
import torch
from transformers import GPT2LMHeadModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tinyformer.convert import gpt_from_hf  # noqa: E402
from tinyformer.tokenizer import Tokenizer  # noqa: E402

PROMPTS = [
    "The capital of France is",
    "In 1905, Albert Einstein published",
    "def fibonacci(n):\n    ",
    "Once upon a time, there was a little girl who",
    "The patient was admitted with chest pain and",
    "Transformers use self-attention to",
    "1, 2, 3, 4, 5,",
    "To be, or not to be,",
]
N_NEW = 40


def main():
    torch.set_grad_enabled(False)
    hf = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    mine = gpt_from_hf(hf)
    tok = Tokenizer.from_tiktoken("gpt2")          # our encoder with GPT-2's ranks
    ref_tok = tiktoken.get_encoding("gpt2")
    rows, max_diff, agree, total = [], 0.0, 0, 0
    for p in PROMPTS:
        ids = tok.encode(p)
        assert ids == ref_tok.encode(p)
        x = torch.tensor([ids])
        a, b = hf(x).logits, mine(x)
        d = (a - b).abs().max().item()
        max_diff = max(max_diff, d)
        agree += (a.argmax(-1) == b.argmax(-1)).sum().item()
        total += x.numel()
        g_hf = hf.generate(x, max_new_tokens=N_NEW, do_sample=False, pad_token_id=50256)[0, len(ids):]
        g_c = mine.generate(x, N_NEW, use_cache=True)[0, len(ids):]
        g_nc = mine.generate(x, N_NEW, use_cache=False)[0, len(ids):]
        rows.append({"prompt": p, "max_abs_logit_diff": d,
                     "greedy_identical_cache": bool(torch.equal(g_hf, g_c)),
                     "greedy_identical_nocache": bool(torch.equal(g_hf, g_nc)),
                     "first_divergence": next((i for i, (u, v) in enumerate(zip(g_hf.tolist(), g_c.tolist()))
                                               if u != v), None),
                     "completion": tok.decode(g_c.tolist())})
        print(f"{d:.2e}  hf==mine(cache)={rows[-1]['greedy_identical_cache']}  {p!r} ->"
              f" {rows[-1]['completion'][:60]!r}")
    # one long sequence at full context to exercise all 1024 learned positions
    torch.manual_seed(0)
    x = torch.randint(0, 50257, (1, 1024))
    long_diff = (hf(x).logits - mine(x)).abs().max().item()
    out = {
        "model": "gpt2 (124M), fp32, CPU",
        "torch": torch.__version__,
        "prompts": len(PROMPTS),
        "max_abs_logit_diff": max_diff,
        "max_abs_logit_diff_random_1024_tokens": long_diff,
        "top1_agreement": f"{agree}/{total}",
        "greedy_tokens_per_prompt": N_NEW,
        "greedy_identical_all": all(r["greedy_identical_cache"] and r["greedy_identical_nocache"] for r in rows),
        "rows": rows,
        "measured": time.strftime("%Y-%m-%d"),
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/verify_gpt2.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
