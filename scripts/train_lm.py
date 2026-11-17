"""Train a decoder LM on TinyStories token files produced by tokenizer_compare.py.

Logs JSON lines (step, train loss, val loss, val bits-per-byte) so runs with
different tokenizers can be compared on bits per byte rather than per token.

  python scripts/train_lm.py --vocab 4096 --positions rope --n_kv_head 2 --steps 2000 --out runs/x
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tinyformer.data import DATA, TokenStream  # noqa: E402
from tinyformer.model import GPT, Config  # noqa: E402
from tinyformer.tokenizer import Tokenizer  # noqa: E402


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab", type=int, default=4096)
    p.add_argument("--data_dir", default=str(DATA))
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--n_kv_head", type=int, default=None)
    p.add_argument("--positions", default="learned")
    p.add_argument("--norm", default="layer")
    p.add_argument("--mlp", default="gelu")
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--eval_batches", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", action="store_true", help="fp16 autocast + GradScaler (CUDA only)")
    p.add_argument("--out", required=True)
    return p.parse_args()


@torch.no_grad()
def evaluate(model, val, tok_bytes, args):
    """Mean val loss (nats/token) and bits per UTF-8 byte over a fixed set of windows."""
    model.eval()
    nll, ntok, nbytes = 0.0, 0, 0
    for x, y in val.sequential(args.batch, args.seq, args.eval_batches):
        x, y = x.to(args.device), y.to(args.device)
        logits = model(x)
        l = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        nll += l.item()
        ntok += y.numel()
        nbytes += tok_bytes[y.cpu()].sum().item()
    model.train()
    return nll / ntok, nll / math.log(2) / nbytes


def main():
    args = parse()
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d = Path(args.data_dir)
    tok = Tokenizer.load(d / f"tok{args.vocab}.json")
    tok_bytes = torch.tensor([len(tok.decoder[i]) if tok.decoder[i] != b"<|endoftext|>" else 1
                              for i in range(tok.vocab_size)])
    train = TokenStream(d / f"ts_v{args.vocab}_train.bin", tok.vocab_size)
    val = TokenStream(d / f"ts_v{args.vocab}_val.bin", tok.vocab_size)
    cfg = Config(vocab_size=tok.vocab_size, max_seq_len=args.seq, d_model=args.d_model, n_layer=args.n_layer,
                 n_head=args.n_head, n_kv_head=args.n_kv_head, positions=args.positions, norm=args.norm,
                 mlp=args.mlp, bias=False, tie_weights=True)
    model = GPT(cfg).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.95))
    lr_at = lambda s: args.lr * min(1, (s + 1) / args.warmup) * (  # noqa: E731
        0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    gen = torch.Generator().manual_seed(args.seed)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    meta = {"args": vars(args), "config": cfg.to_dict(), "params": n_params,
            "train_tokens_available": len(train), "torch": torch.__version__}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"params {n_params / 1e6:.2f}M, train tokens {len(train):,}")
    log = open(out / "log.jsonl", "w")
    t0 = time.perf_counter()
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            vl, bpb = evaluate(model, val, tok_bytes, args)
            rec = {"step": step, "val_loss": vl, "val_bpb": bpb, "elapsed_s": time.perf_counter() - t0,
                   "tokens_seen": step * args.batch * args.seq}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(rec)
        if step == args.steps:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y = train.batch(args.batch, args.seq, gen)
        with torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
            _, loss = model(x.to(args.device), targets=y.to(args.device))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if step % 50 == 0:
            log.write(json.dumps({"step": step, "train_loss": loss.item()}) + "\n")
    torch.save({"config": cfg.to_dict(), "model": model.state_dict(), "vocab": args.vocab}, out / "model.pt")
    log.close()


if __name__ == "__main__":
    main()
