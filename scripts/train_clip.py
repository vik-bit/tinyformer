"""Train the small from-scratch CLIP on Flickr8k train; validation recall each epoch.

  python scripts/train_clip.py --epochs 30 --out runs/clip
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tinyformer.clip_train import (LOAD_RES, RES, build, make_tokenize, normalize, resize_short,  # noqa: E402
                                   to_tensor)
from tinyformer.flickr import load_split, open_image  # noqa: E402
from tinyformer.tokenizer import EOT, Tokenizer  # noqa: E402

CACHE = Path(__file__).resolve().parents[1] / "data" / "flickr8k_cache"


def images_u8(split, size):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{split}_{size}.pt"
    if f.exists():
        return torch.load(f)
    imgs, _ = load_split(split)
    x = torch.stack([to_tensor(resize_short(open_image(b), size)) for b in imgs])
    torch.save(x, f)
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--vocab", type=int, default=2048)
    p.add_argument("--d", type=int, default=192)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    _, caps = load_split("train")
    _, vcaps = load_split("validation")
    tok = Tokenizer.train("\n".join(c.strip().lower() for cs in caps for c in cs), a.vocab)
    tokenize = make_tokenize(tok)
    tr = images_u8("train", LOAD_RES)
    va = normalize(images_u8("validation", RES))
    arch = dict(d=a.d, layers=a.layers, heads=4, embed_dim=128)
    model = build(tok.vocab_size, **arch)
    params = sum(p.numel() for p in model.parameters())
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    other = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": other, "weight_decay": 0}],
                            lr=a.lr, betas=(0.9, 0.98))
    steps_per_epoch = len(tr) // a.batch
    total = steps_per_epoch * a.epochs
    warm = steps_per_epoch
    gen = torch.Generator().manual_seed(a.seed)
    log = open(out / "log.jsonl", "w")
    print(f"params {params / 1e6:.2f}M, {steps_per_epoch} steps/epoch")
    t0, step = time.perf_counter(), 0
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(tr), generator=gen)
        for b in range(steps_per_epoch):
            ix = perm[b * a.batch:(b + 1) * a.batch]
            # random 64-crop of the 72 image + horizontal flip; one random caption of five
            ox, oy = torch.randint(0, LOAD_RES - RES + 1, (2,), generator=gen).tolist()
            img = tr[ix, :, oy:oy + RES, ox:ox + RES]
            if torch.rand(1, generator=gen).item() < 0.5:
                img = img.flip(-1)
            ci = torch.randint(0, 5, (len(ix),), generator=gen)
            ids, eot = tokenize([caps[i][c] for i, c in zip(ix.tolist(), ci.tolist())])
            lr = a.lr * min(1, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * step / total))
            for g in opt.param_groups:
                g["lr"] = lr
            _, loss = model(normalize(img), ids, eot)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
        r = val_recall(model, va, vcaps, tokenize)
        rec = {"epoch": ep + 1, "step": step, "train_loss_last": loss.item(), "val_t2i": r,
               "temperature": 1 / model.logit_scale.exp().item(), "elapsed_s": time.perf_counter() - t0}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(rec, flush=True)
    tokd = {"tokens": [b.hex() for b, _ in sorted(tok.ranks.items(), key=lambda kv: kv[1])],
            "special": tok.special, "pattern": tok.pattern}
    torch.save({"model": model.state_dict(), "arch": arch, "tok": tokd, "args": vars(a), "params": params},
               out / "model.pt")


@torch.no_grad()
def val_recall(model, va, vcaps, tokenize):
    model.eval()
    im = torch.cat([model.encode_image(va[i:i + 250]) for i in range(0, len(va), 250)])
    texts = [c for cs in vcaps for c in cs]
    tx = torch.cat([model.encode_text(*tokenize(texts[i:i + 500])) for i in range(0, len(texts), 500)])
    gt = torch.arange(len(va)).repeat_interleave(5)
    top = (tx @ im.t()).topk(10, dim=1).indices
    return {f"R@{k}": (top[:, :k] == gt[:, None]).any(1).float().mean().item() for k in (1, 5, 10)}


if __name__ == "__main__":
    main()
