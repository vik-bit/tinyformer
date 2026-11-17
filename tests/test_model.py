import math

import pytest
import torch
import torch.nn.functional as F

from tinyformer.model import CLIP, GPT, Attention, Config, ViT

torch.manual_seed(0)


def tiny(**kw):
    base = dict(vocab_size=97, max_seq_len=32, d_model=32, n_layer=2, n_head=4)
    base.update(kw)
    return Config(**base)


VARIANTS = [
    dict(),
    dict(positions="rope"),
    dict(norm="rms", mlp="swiglu", bias=False),
    dict(n_kv_head=2, positions="rope"),
    dict(n_kv_head=1, mlp="gelu"),
    dict(tie_weights=False),
]


@pytest.mark.parametrize("kw", VARIANTS)
def test_shapes_and_loss(kw):
    m = GPT(tiny(**kw))
    x = torch.randint(0, 97, (3, 17))
    logits, loss = m(x, targets=torch.randint(0, 97, (3, 17)))
    assert logits.shape == (3, 17, 97)
    assert abs(loss.item() - math.log(97)) < 0.5     # ~uniform at init


@pytest.mark.parametrize("kw", VARIANTS)
def test_causality(kw):
    """Changing token t must not change logits at positions < t."""
    m = GPT(tiny(**kw)).eval()
    x = torch.randint(0, 97, (1, 20))
    y = x.clone()
    y[0, 12:] = (y[0, 12:] + 1) % 97
    a, b = m(x), m(y)
    assert torch.allclose(a[:, :12], b[:, :12], atol=1e-6)
    assert not torch.allclose(a[:, 12:], b[:, 12:], atol=1e-4)


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("n_kv", [4, 2, 1])
def test_attention_matches_sdpa(causal, n_kv):
    cfg = tiny(causal=causal, n_kv_head=n_kv)
    att = Attention(cfg).eval()
    x = torch.randn(2, 11, 32)
    q = att.q(x).view(2, 11, 4, 8).transpose(1, 2)
    k = att.k(x).view(2, 11, n_kv, 8).transpose(1, 2).repeat_interleave(4 // n_kv, 1)
    v = att.v(x).view(2, 11, n_kv, 8).transpose(1, 2).repeat_interleave(4 // n_kv, 1)
    ref = att.o(F.scaled_dot_product_attention(q, k, v, is_causal=causal).transpose(1, 2).reshape(2, 11, 32))
    assert torch.allclose(att(x), ref, atol=1e-5)


@pytest.mark.parametrize("kw", VARIANTS)
def test_kv_cache_matches_full_forward(kw):
    m = GPT(tiny(**kw)).eval()
    x = torch.randint(0, 97, (2, 9))
    full = m(x)
    caches = [dict() for _ in m.trunk.blocks]
    step = [m(x[:, :5], caches=caches, pos=0)]
    for t in range(5, 9):
        step.append(m(x[:, t:t + 1], caches=caches, pos=t))
    assert torch.allclose(full, torch.cat(step, 1), atol=1e-5)


@pytest.mark.parametrize("kw", VARIANTS)
def test_generate_cache_equals_no_cache(kw):
    m = GPT(tiny(**kw)).eval()
    x = torch.randint(0, 97, (2, 4))
    assert torch.equal(m.generate(x, 12, use_cache=True), m.generate(x, 12, use_cache=False))


def test_rope_is_relative():
    """With RoPE, q.k depends only on the offset between positions."""
    from tinyformer.model import apply_rope, rope_cache
    cos, sin = rope_cache(8, 64, 10000.0)
    q, k = torch.randn(8), torch.randn(8)

    def dot(i, j):
        qi = apply_rope(q.view(1, 1, 1, 8), cos[i:i + 1], sin[i:i + 1]).flatten()
        kj = apply_rope(k.view(1, 1, 1, 8), cos[j:j + 1], sin[j:j + 1]).flatten()
        return (qi @ kj).item()

    assert abs(dot(3, 1) - dot(40, 38)) < 1e-4
    assert abs(dot(3, 1) - dot(3, 2)) > 1e-4


def test_gqa_has_fewer_params():
    assert GPT(tiny(n_kv_head=1)).num_params() < GPT(tiny()).num_params()


@pytest.mark.parametrize("kw", [dict(), dict(positions="rope", n_kv_head=2, norm="rms", mlp="swiglu")])
def test_overfit_one_batch(kw):
    torch.manual_seed(0)
    m = GPT(tiny(**kw))
    x = torch.randint(0, 97, (4, 16))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(150):
        _, loss = m(x[:, :-1], targets=x[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1


def test_vit_shapes_and_bidirectional():
    cfg = Config(d_model=32, n_layer=2, n_head=4, positions="none", causal=False, mlp="gelu")
    m = ViT(cfg, image_size=32, patch=8, n_classes=10).eval()
    img = torch.randn(2, 3, 32, 32)
    out = m(img)
    assert out.shape == (2, 10)
    img2 = img.clone()
    img2[:, :, -8:, -8:] += 1.0      # last patch changes -> CLS changes (no causal mask)
    assert not torch.allclose(m(img2), out)


def test_clip_loss_and_overfit():
    torch.manual_seed(0)
    v = Config(d_model=32, n_layer=2, n_head=4, positions="none", causal=False, mlp="quick_gelu")
    t = Config(vocab_size=50, max_seq_len=12, d_model=32, n_layer=2, n_head=4, tie_weights=False)
    m = CLIP(v, t, embed_dim=16, image_size=32, patch=8)
    img = torch.randn(6, 3, 32, 32)
    ids = torch.randint(0, 50, (6, 12))
    eot = torch.full((6,), 11)
    _, loss0 = m(img, ids, eot)
    assert torch.isfinite(loss0)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(100):
        logits, loss = m(img, ids, eot)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert (logits.argmax(1) == torch.arange(6)).all()
