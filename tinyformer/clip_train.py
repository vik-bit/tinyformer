"""Small CLIP trained from scratch on Flickr8k train (6000 images x 5 captions).

Image tower: ViT on 64x64 crops, patch 8. Text tower: causal transformer over
this repo's BPE (trained on the train captions only), pooled at <|endoftext|>.
Loss: symmetric InfoNCE with a learnable temperature (CLIP.forward).
"""
from __future__ import annotations

import torch

from .model import CLIP, Config
from .tokenizer import EOT, Tokenizer

MEAN = torch.tensor([0.4815, 0.4578, 0.4082]).view(3, 1, 1)
STD = torch.tensor([0.2686, 0.2613, 0.2758]).view(3, 1, 1)
RES, LOAD_RES, MAX_LEN = 64, 72, 32


def resize_short(img, size):
    w, h = img.size
    s = size / min(w, h)
    img = img.resize((max(size, round(w * s)), max(size, round(h * s))), resample=3)
    w, h = img.size
    l, t = (w - size) // 2, (h - size) // 2
    return img.crop((l, t, l + size, t + size))


def to_tensor(img):
    import numpy as np
    return torch.from_numpy(np.asarray(img, dtype="uint8").copy()).permute(2, 0, 1)


def normalize(u8):
    return (u8.float() / 255 - MEAN) / STD


def eval_transform(img):
    return normalize(to_tensor(resize_short(img, RES)))


def make_tokenize(tok: Tokenizer, max_len: int = MAX_LEN):
    eot = tok.eot_id

    def tokenize(texts):
        rows = [tok.encode(t.strip().lower())[:max_len - 1] + [eot] for t in texts]
        T = max(len(r) for r in rows)
        ids = torch.zeros(len(rows), T, dtype=torch.long)
        for i, r in enumerate(rows):
            ids[i, :len(r)] = torch.tensor(r)
        return ids, torch.tensor([len(r) - 1 for r in rows])
    return tokenize


def build(vocab_size: int, d: int = 192, layers: int = 4, heads: int = 4, embed_dim: int = 128) -> CLIP:
    v = Config(d_model=d, n_layer=layers, n_head=heads, positions="none", causal=False, mlp="gelu",
               max_seq_len=(RES // 8) ** 2 + 1)
    t = Config(vocab_size=vocab_size, max_seq_len=MAX_LEN, d_model=d, n_layer=layers, n_head=heads,
               mlp="gelu", tie_weights=False)
    return CLIP(v, t, embed_dim=embed_dim, image_size=RES, patch=8, vision_pre_norm=True)


def load_scratch(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    tok = Tokenizer(**{"ranks": {bytes.fromhex(h): i for i, h in enumerate(ck["tok"]["tokens"])},
                       "special": ck["tok"]["special"], "pattern": ck["tok"]["pattern"]})
    m = build(tok.vocab_size, **ck["arch"])
    m.load_state_dict(ck["model"])
    return m.eval(), eval_transform, make_tokenize(tok)
