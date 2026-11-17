"""Copy published weights (HF GPT-2, timm ViT, HF CLIP) into tinyformer modules.

Each loader builds the matching ``Config``, renames / splits / transposes
tensors, and loads with ``strict=True`` so a missing or extra tensor is an
error rather than a silent random init.
"""
from __future__ import annotations

import torch

from .model import CLIP, GPT, Config, ViT


def _split_qkv(prefix_out: str, w: torch.Tensor, b: torch.Tensor | None, sd: dict):
    for name, wi in zip("qkv", w.chunk(3, dim=0)):
        sd[f"{prefix_out}.{name}.weight"] = wi
    if b is not None:
        for name, bi in zip("qkv", b.chunk(3, dim=0)):
            sd[f"{prefix_out}.{name}.bias"] = bi


# ------------------------------------------------------------------ GPT-2
def gpt2_config(hf_cfg) -> Config:
    return Config(vocab_size=hf_cfg.vocab_size, max_seq_len=hf_cfg.n_positions,
                  d_model=hf_cfg.n_embd, n_layer=hf_cfg.n_layer, n_head=hf_cfg.n_head,
                  positions="learned", norm="layer", norm_eps=hf_cfg.layer_norm_epsilon,
                  mlp="gelu_tanh", bias=True, causal=True, tie_weights=True)


def gpt2_state_dict(hf_sd: dict, n_layer: int) -> dict:
    """HF GPT-2 uses Conv1D (weight shape (in, out)) and a fused c_attn."""
    sd = {"tok.weight": hf_sd["transformer.wte.weight"],
          "pos.weight": hf_sd["transformer.wpe.weight"],
          "trunk.norm.weight": hf_sd["transformer.ln_f.weight"],
          "trunk.norm.bias": hf_sd["transformer.ln_f.bias"]}
    sd["head.weight"] = sd["tok.weight"]
    for i in range(n_layer):
        h, o = f"transformer.h.{i}", f"trunk.blocks.{i}"
        sd[f"{o}.norm1.weight"], sd[f"{o}.norm1.bias"] = hf_sd[f"{h}.ln_1.weight"], hf_sd[f"{h}.ln_1.bias"]
        sd[f"{o}.norm2.weight"], sd[f"{o}.norm2.bias"] = hf_sd[f"{h}.ln_2.weight"], hf_sd[f"{h}.ln_2.bias"]
        _split_qkv(f"{o}.attn", hf_sd[f"{h}.attn.c_attn.weight"].t(), hf_sd[f"{h}.attn.c_attn.bias"], sd)
        sd[f"{o}.attn.o.weight"] = hf_sd[f"{h}.attn.c_proj.weight"].t()
        sd[f"{o}.attn.o.bias"] = hf_sd[f"{h}.attn.c_proj.bias"]
        sd[f"{o}.mlp.up.weight"] = hf_sd[f"{h}.mlp.c_fc.weight"].t()
        sd[f"{o}.mlp.up.bias"] = hf_sd[f"{h}.mlp.c_fc.bias"]
        sd[f"{o}.mlp.down.weight"] = hf_sd[f"{h}.mlp.c_proj.weight"].t()
        sd[f"{o}.mlp.down.bias"] = hf_sd[f"{h}.mlp.c_proj.bias"]
    return sd


def gpt_from_hf(hf_model) -> GPT:
    cfg = gpt2_config(hf_model.config)
    m = GPT(cfg)
    hf_sd = {k: v for k, v in hf_model.state_dict().items() if not k.endswith(".attn.bias")
             and not k.endswith(".attn.masked_bias")}
    if "transformer.wte.weight" not in hf_sd:     # GPT2Model rather than GPT2LMHeadModel
        hf_sd = {"transformer." + k: v for k, v in hf_sd.items()}
    m.load_state_dict(gpt2_state_dict(hf_sd, cfg.n_layer), strict=True)
    return m.eval()


# -------------------------------------------------------------------- ViT
def vit_from_timm(tm) -> ViT:
    pe = tm.patch_embed.proj
    d = pe.out_channels
    n_layer = len(tm.blocks)
    cfg = Config(d_model=d, n_layer=n_layer, n_head=tm.blocks[0].attn.num_heads,
                 d_ff=tm.blocks[0].mlp.fc1.out_features, positions="none", norm="layer",
                 norm_eps=tm.norm.eps, mlp="gelu", bias=True, causal=False,
                 max_seq_len=tm.pos_embed.shape[1])
    img = tm.patch_embed.img_size[0]
    m = ViT(cfg, image_size=img, patch=pe.kernel_size[0], n_classes=tm.head.out_features)
    t = tm.state_dict()
    sd = {"patch.weight": t["patch_embed.proj.weight"], "patch.bias": t["patch_embed.proj.bias"],
          "cls": t["cls_token"], "pos": t["pos_embed"],
          "trunk.norm.weight": t["norm.weight"], "trunk.norm.bias": t["norm.bias"],
          "head.weight": t["head.weight"], "head.bias": t["head.bias"]}
    for i in range(n_layer):
        h, o = f"blocks.{i}", f"trunk.blocks.{i}"
        for a, b in (("norm1", "norm1"), ("norm2", "norm2"), ("attn.proj", "attn.o"),
                     ("mlp.fc1", "mlp.up"), ("mlp.fc2", "mlp.down")):
            sd[f"{o}.{b}.weight"], sd[f"{o}.{b}.bias"] = t[f"{h}.{a}.weight"], t[f"{h}.{a}.bias"]
        _split_qkv(f"{o}.attn", t[f"{h}.attn.qkv.weight"], t[f"{h}.attn.qkv.bias"], sd)
    m.load_state_dict(sd, strict=True)
    return m.eval()


# ------------------------------------------------------------------- CLIP
def _clip_layers(src: dict, sp: str, dp: str, n_layer: int, sd: dict):
    for i in range(n_layer):
        h, o = f"{sp}.encoder.layers.{i}", f"{dp}.trunk.blocks.{i}"
        for a, b in (("layer_norm1", "norm1"), ("layer_norm2", "norm2"),
                     ("self_attn.q_proj", "attn.q"), ("self_attn.k_proj", "attn.k"),
                     ("self_attn.v_proj", "attn.v"), ("self_attn.out_proj", "attn.o"),
                     ("mlp.fc1", "mlp.up"), ("mlp.fc2", "mlp.down")):
            sd[f"{o}.{b}.weight"], sd[f"{o}.{b}.bias"] = src[f"{h}.{a}.weight"], src[f"{h}.{a}.bias"]


def clip_from_hf(hf) -> CLIP:
    vc, tc = hf.config.vision_config, hf.config.text_config
    act = {"quick_gelu": "quick_gelu", "gelu": "gelu"}[vc.hidden_act]
    vcfg = Config(d_model=vc.hidden_size, n_layer=vc.num_hidden_layers, n_head=vc.num_attention_heads,
                  d_ff=vc.intermediate_size, positions="none", norm="layer", norm_eps=vc.layer_norm_eps,
                  mlp=act, causal=False, max_seq_len=(vc.image_size // vc.patch_size) ** 2 + 1)
    tcfg = Config(vocab_size=tc.vocab_size, max_seq_len=tc.max_position_embeddings,
                  d_model=tc.hidden_size, n_layer=tc.num_hidden_layers, n_head=tc.num_attention_heads,
                  d_ff=tc.intermediate_size, positions="learned", norm="layer",
                  norm_eps=tc.layer_norm_eps, mlp=act, causal=True, tie_weights=False)
    m = CLIP(vcfg, tcfg, embed_dim=hf.config.projection_dim, image_size=vc.image_size,
             patch=vc.patch_size, vision_pre_norm=True)
    s = hf.state_dict()
    v, t = "vision_model", "text_model"
    sd = {"visual.patch.weight": s[f"{v}.embeddings.patch_embedding.weight"],
          "visual.cls": s[f"{v}.embeddings.class_embedding"].view(1, 1, -1),
          "visual.pos": s[f"{v}.embeddings.position_embedding.weight"].unsqueeze(0),
          "visual.pre_norm.weight": s[f"{v}.pre_layrnorm.weight"],
          "visual.pre_norm.bias": s[f"{v}.pre_layrnorm.bias"],
          "visual.trunk.norm.weight": s[f"{v}.post_layernorm.weight"],
          "visual.trunk.norm.bias": s[f"{v}.post_layernorm.bias"],
          "visual.head.weight": s["visual_projection.weight"],
          "text.tok.weight": s[f"{t}.embeddings.token_embedding.weight"],
          "text.pos": s[f"{t}.embeddings.position_embedding.weight"].unsqueeze(0),
          "text.trunk.norm.weight": s[f"{t}.final_layer_norm.weight"],
          "text.trunk.norm.bias": s[f"{t}.final_layer_norm.bias"],
          "text.proj.weight": s["text_projection.weight"],
          "logit_scale": s["logit_scale"]}
    _clip_layers(s, v, "visual", vc.num_hidden_layers, sd)
    _clip_layers(s, t, "text", tc.num_hidden_layers, sd)
    m.load_state_dict(sd, strict=True)
    return m.eval()
