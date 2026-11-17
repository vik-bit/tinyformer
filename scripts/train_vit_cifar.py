"""Train a small ViT (same Block/Trunk as the LM) on CIFAR-10 from scratch.

Patch 4 on 32x32 -> 64 patches + CLS. Random crop (pad 4) + flip, AdamW,
cosine LR, label smoothing 0.1. Reports test accuracy (10k images) per epoch.
Uses torchvision's CIFAR-10 loader if installed, else the python pickle archive.
  python scripts/train_vit_cifar.py --epochs 15 --out runs/vit_cifar
"""
import argparse
import json
import math
import pickle
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tinyformer.model import Config, ViT  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "cifar10"
URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


def load_cifar():
    d = DATA / "cifar-10-batches-py"
    if not d.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        tgz = DATA / "cifar-10-python.tar.gz"
        if not tgz.exists():
            urllib.request.urlretrieve(URL, tgz)
        with tarfile.open(tgz) as t:
            t.extractall(DATA, filter="data")

    def read(names):
        xs, ys = [], []
        for n in names:
            with open(d / n, "rb") as f:
                b = pickle.load(f, encoding="bytes")
            xs.append(b[b"data"])
            ys += b[b"labels"]
        return torch.from_numpy(np.concatenate(xs).reshape(-1, 3, 32, 32)), torch.tensor(ys)

    return read([f"data_batch_{i}" for i in range(1, 6)]), read(["test_batch"])


def augment(x, gen):
    B = len(x)
    pad = F.pad(x, (4, 4, 4, 4), mode="reflect")
    i, j = torch.randint(0, 9, (2, B), generator=gen)
    rows = (i[:, None] + torch.arange(32))[:, :, None].expand(B, 32, 32)
    cols = (j[:, None] + torch.arange(32))[:, None, :].expand(B, 32, 32)
    out = pad[torch.arange(B)[:, None, None], :, rows, cols].permute(0, 3, 1, 2)
    flip = torch.rand(B, generator=gen) < 0.5
    out[flip] = out[flip].flip(-1)
    return out


def norm(u8):
    return (u8.float() / 255 - MEAN) / STD


@torch.no_grad()
def accuracy(model, x, y, bs=500):
    model.eval()
    correct = sum((model(norm(x[i:i + bs])).argmax(1) == y[i:i + bs]).sum().item() for i in range(0, len(x), bs))
    model.train()
    return correct / len(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--d", type=int, default=192)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    (xtr, ytr), (xte, yte) = load_cifar()
    cfg = Config(d_model=a.d, n_layer=a.layers, n_head=a.heads, positions="none", causal=False,
                 mlp="gelu", norm_eps=1e-6, max_seq_len=65)
    model = ViT(cfg, image_size=32, patch=4, n_classes=10).to(a.device)
    params = sum(p.numel() for p in model.parameters())
    decay = [p for p in model.parameters() if p.dim() >= 2]
    other = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": other, "weight_decay": 0}],
                            lr=a.lr)
    spe = len(xtr) // a.batch
    total, warm = spe * a.epochs, spe
    gen = torch.Generator().manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.jsonl", "w")
    print(f"params {params / 1e6:.2f}M")
    t0, step = time.perf_counter(), 0
    for ep in range(a.epochs):
        perm = torch.randperm(len(xtr), generator=gen)
        tl = 0.0
        for b in range(spe):
            ix = perm[b * a.batch:(b + 1) * a.batch]
            x = norm(augment(xtr[ix], gen)).to(a.device)
            lr = a.lr * min(1, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * step / total))
            for g in opt.param_groups:
                g["lr"] = lr
            loss = F.cross_entropy(model(x), ytr[ix].to(a.device), label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item()
            step += 1
        acc = accuracy(model, xte.to(a.device), yte.to(a.device))
        rec = {"epoch": ep + 1, "train_loss": tl / spe, "test_acc": acc, "elapsed_s": time.perf_counter() - t0}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(rec, flush=True)
    torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "args": vars(a), "params": params},
               out / "model.pt")


if __name__ == "__main__":
    main()
