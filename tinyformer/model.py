"""One set of transformer blocks, used for a decoder LM, a ViT and a CLIP model.

Everything is plain PyTorch modules: attention is written out as
softmax(QK^T / sqrt(d)) V (no ``F.scaled_dot_product_attention``) so the maths
is visible; tests/test_model.py checks it against the fused kernel.

Options covered by the config:
  positions   learned | rope | none (ViT/CLIP add their own learned table)
  norm        layer | rms
  mlp         gelu | gelu_tanh | quick_gelu | swiglu
  n_kv_head   < n_head gives grouped-query attention (1 = multi-query)
  causal      True for the LM / CLIP text tower, False for ViT
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 50257
    max_seq_len: int = 1024
    d_model: int = 768
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int | None = None      # None -> n_head (plain MHA)
    d_ff: int | None = None           # None -> 4 * d_model (or 8/3 * d_model for swiglu)
    positions: str = "learned"        # learned | rope | none
    norm: str = "layer"               # layer | rms
    norm_eps: float = 1e-5
    mlp: str = "gelu_tanh"            # gelu | gelu_tanh | quick_gelu | swiglu
    bias: bool = True
    causal: bool = True
    tie_weights: bool = True
    dropout: float = 0.0
    rope_theta: float = 10000.0

    def __post_init__(self):
        if self.n_kv_head is None:
            self.n_kv_head = self.n_head
        assert self.d_model % self.n_head == 0
        assert self.n_head % self.n_kv_head == 0
        if self.d_ff is None:
            if self.mlp == "swiglu":   # keep params ~equal to a 4x GELU MLP
                self.d_ff = 64 * math.ceil(8 * self.d_model / 3 / 64)
            else:
                self.d_ff = 4 * self.d_model

    def to_dict(self):
        return asdict(self)


# --------------------------------------------------------------------- parts
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def make_norm(cfg: Config) -> nn.Module:
    if cfg.norm == "rms":
        return RMSNorm(cfg.d_model, cfg.norm_eps)
    return nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps, bias=cfg.bias)


def rope_cache(head_dim: int, max_len: int, theta: float, device=None):
    inv = 1.0 / theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    t = torch.arange(max_len, device=device).float()
    freqs = torch.outer(t, inv)                       # (T, hd/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    """x: (B, H, T, hd). Rotates consecutive pairs (x0,x1), (x2,x3), ..."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.n_head, self.n_kv = cfg.n_head, cfg.n_kv_head
        self.hd = cfg.d_model // cfg.n_head
        self.q = nn.Linear(cfg.d_model, cfg.n_head * self.hd, bias=cfg.bias)
        self.k = nn.Linear(cfg.d_model, cfg.n_kv_head * self.hd, bias=cfg.bias)
        self.v = nn.Linear(cfg.d_model, cfg.n_kv_head * self.hd, bias=cfg.bias)
        self.o = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, rope=None, cache=None, pos: int = 0, attn_mask=None):
        """x: (B, T, C). ``cache`` is a dict holding this layer's past k/v;
        ``pos`` is the absolute position of x[:, 0]. ``attn_mask`` is an
        optional (B, T_k) bool tensor, True = may attend (padding mask)."""
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.n_head, self.hd).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        if rope is not None:
            cos, sin = rope[0][pos:pos + T], rope[1][pos:pos + T]
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        if self.n_kv != self.n_head:                 # GQA: share each kv head
            rep = self.n_head // self.n_kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        Tk = k.shape[2]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)       # (B, H, T, Tk)
        if self.cfg.causal:
            # query i sits at absolute position pos+i and may see keys 0..pos+i
            qi = torch.arange(T, device=x.device)[:, None] + (Tk - T)
            kj = torch.arange(Tk, device=x.device)[None, :]
            att = att.masked_fill(kj > qi, float("-inf"))
        if attn_mask is not None:
            att = att.masked_fill(~attn_mask[:, None, None, :], float("-inf"))
        att = self.drop(F.softmax(att.float(), dim=-1).type_as(q))
        y = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.o(y)


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.kind = cfg.mlp
        if cfg.mlp == "swiglu":
            self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        if self.kind == "swiglu":
            h = F.silu(self.gate(x)) * self.up(x)
        elif self.kind == "gelu":
            h = F.gelu(self.up(x))
        elif self.kind == "gelu_tanh":
            h = F.gelu(self.up(x), approximate="tanh")
        elif self.kind == "quick_gelu":       # used by OpenAI CLIP
            u = self.up(x)
            h = u * torch.sigmoid(1.702 * u)
        else:
            raise ValueError(self.kind)
        return self.drop(self.down(h))


class Block(nn.Module):
    """Pre-norm residual block: x + attn(norm(x)), then x + mlp(norm(x))."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.norm1 = make_norm(cfg)
        self.attn = Attention(cfg)
        self.norm2 = make_norm(cfg)
        self.mlp = MLP(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, rope=None, cache=None, pos=0, attn_mask=None):
        x = x + self.drop(self.attn(self.norm1(x), rope, cache, pos, attn_mask))
        return x + self.mlp(self.norm2(x))


class Trunk(nn.Module):
    """The shared part: N blocks + final norm. Positions are handled here only for RoPE."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = make_norm(cfg)
        if cfg.positions == "rope":
            cos, sin = rope_cache(cfg.d_model // cfg.n_head, cfg.max_seq_len, cfg.rope_theta)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x, caches=None, pos=0, attn_mask=None):
        rope = (self.rope_cos, self.rope_sin) if self.cfg.positions == "rope" else None
        for i, blk in enumerate(self.blocks):
            x = blk(x, rope, None if caches is None else caches[i], pos, attn_mask)
        return self.norm(x)


def init_weights(module: nn.Module, n_layer: int):
    """GPT-2 style init: N(0, 0.02), residual output projections scaled by 1/sqrt(2N)."""
    for name, p in module.named_parameters():
        if p.dim() >= 2:
            std = 0.02 / math.sqrt(2 * n_layer) if name.endswith(("attn.o.weight", "mlp.down.weight")) else 0.02
            nn.init.normal_(p, 0.0, std)
        elif name.endswith("bias"):
            nn.init.zeros_(p)


# ---------------------------------------------------------------------- LM
class GPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.causal
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        if cfg.positions == "learned":
            self.pos = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.trunk = Trunk(cfg)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        init_weights(self, cfg.n_layer)
        if cfg.tie_weights:
            self.head.weight = self.tok.weight

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and self.cfg.positions == "learned":
            n -= self.pos.weight.numel()
        return n

    def forward(self, idx, targets=None, caches=None, pos: int = 0):
        B, T = idx.shape
        assert pos + T <= self.cfg.max_seq_len, "sequence longer than max_seq_len"
        x = self.tok(idx)
        if self.cfg.positions == "learned":
            x = x + self.pos(torch.arange(pos, pos + T, device=idx.device))
        x = self.trunk(self.drop(x), caches, pos)
        logits = self.head(x)
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, use_cache: bool = True,
                 temperature: float = 0.0, top_k: int | None = None, generator=None):
        """temperature 0 = greedy. With use_cache the prompt is run once and
        each new token costs one position; without it the full prefix is re-run."""
        caches = [dict() for _ in self.trunk.blocks] if use_cache else None
        pos = 0
        inp = idx
        for _ in range(max_new_tokens):
            if use_cache:
                logits = self(inp, caches=caches, pos=pos)[:, -1]
                pos += inp.shape[1]
            else:
                logits = self(idx[:, -self.cfg.max_seq_len:])[:, -1]
            if temperature == 0.0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                nxt = torch.multinomial(F.softmax(logits, -1), 1, generator=generator)
            idx = torch.cat([idx, nxt], dim=1)
            inp = nxt
            if use_cache and pos + 1 > self.cfg.max_seq_len:
                break
        return idx


# --------------------------------------------------------------------- ViT
class ViT(nn.Module):
    """Patch embedding (a strided conv) + CLS token + learned positions + Trunk."""

    def __init__(self, cfg: Config, image_size: int = 224, patch: int = 16, in_ch: int = 3,
                 n_classes: int = 1000, pre_norm: bool = False, patch_bias: bool = True,
                 out_dim: int | None = None):
        super().__init__()
        assert not cfg.causal and cfg.positions == "none"
        self.cfg = cfg
        self.n_patches = (image_size // patch) ** 2
        self.patch = nn.Conv2d(in_ch, cfg.d_model, patch, stride=patch, bias=patch_bias)
        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches + 1, cfg.d_model))
        self.pre_norm = make_norm(cfg) if pre_norm else nn.Identity()   # CLIP has ln_pre
        self.trunk = Trunk(cfg)
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.cls, std=0.02)
        init_weights(self.trunk, cfg.n_layer)
        # classifier head (ViT) or projection (CLIP); ``out_dim`` wins if given
        dim = out_dim if out_dim is not None else n_classes
        self.head = nn.Linear(cfg.d_model, dim, bias=out_dim is None) if dim else nn.Identity()

    def features(self, img):
        x = self.patch(img).flatten(2).transpose(1, 2)         # (B, N, C)
        x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], dim=1) + self.pos
        x = self.trunk(self.pre_norm(x))
        return x[:, 0]                                         # CLS token

    def forward(self, img):
        return self.head(self.features(img))


# -------------------------------------------------------------------- CLIP
class TextEncoder(nn.Module):
    """Causal text tower; the sequence is pooled at the end-of-text token."""

    def __init__(self, cfg: Config, out_dim: int):
        super().__init__()
        assert cfg.causal
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Parameter(torch.zeros(1, cfg.max_seq_len, cfg.d_model))
        self.trunk = Trunk(cfg)
        self.proj = nn.Linear(cfg.d_model, out_dim, bias=False)
        nn.init.normal_(self.tok.weight, std=0.02)
        nn.init.normal_(self.pos, std=0.01)
        init_weights(self.trunk, cfg.n_layer)

    def forward(self, ids, eot_index):
        T = ids.shape[1]
        x = self.tok(ids)
        if self.cfg.positions != "rope":
            x = x + self.pos[:, :T]
        x = self.trunk(x)
        return self.proj(x[torch.arange(ids.shape[0]), eot_index])


class CLIP(nn.Module):
    def __init__(self, vision_cfg: Config, text_cfg: Config, embed_dim: int = 512,
                 image_size: int = 224, patch: int = 32, vision_pre_norm: bool = True):
        super().__init__()
        self.visual = ViT(vision_cfg, image_size, patch, out_dim=embed_dim,
                          pre_norm=vision_pre_norm, patch_bias=False)
        self.text = TextEncoder(text_cfg, embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))   # learnable temperature

    def encode_image(self, img):
        return F.normalize(self.visual(img), dim=-1)

    def encode_text(self, ids, eot_index):
        return F.normalize(self.text(ids, eot_index), dim=-1)

    def forward(self, img, ids, eot_index):
        """Returns (logits_per_image, symmetric InfoNCE loss)."""
        im, tx = self.encode_image(img), self.encode_text(ids, eot_index)
        logits = self.logit_scale.exp().clamp(max=100) * im @ tx.t()
        labels = torch.arange(len(img), device=img.device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
        return logits, loss
