"""Reference matching on tiny *randomly initialised* configs (no downloads, runs in CI).

The full-size pretrained checks (GPT-2 small, ViT-B/16, CLIP ViT-B/32) are in
scripts/verify_*.py; these use the same conversion code on small models.
"""
import pytest
import torch

transformers = pytest.importorskip("transformers")
from tinyformer.convert import clip_from_hf, gpt_from_hf, vit_from_timm  # noqa: E402

torch.manual_seed(0)


def test_gpt2_tiny_random_matches_hf():
    cfg = transformers.GPT2Config(vocab_size=101, n_positions=40, n_embd=48, n_layer=3, n_head=4,
                                  resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)
    hf = transformers.GPT2LMHeadModel(cfg).eval()
    mine = gpt_from_hf(hf)
    x = torch.randint(0, 101, (2, 33))
    with torch.no_grad():
        ref = hf(x).logits
        out = mine(x)
    assert (out - ref).abs().max().item() < 1e-4
    g_ref = hf.generate(x[:, :5], max_new_tokens=20, do_sample=False, pad_token_id=0)
    assert torch.equal(mine.generate(x[:, :5], 20), g_ref)


def test_vit_tiny_random_matches_timm():
    timm = pytest.importorskip("timm")
    tm = timm.models.VisionTransformer(img_size=32, patch_size=8, embed_dim=48, depth=3, num_heads=4,
                                       num_classes=7).eval()
    for p in tm.parameters():          # timm zero-inits some tensors; randomise all
        torch.nn.init.normal_(p, std=0.05)
    mine = vit_from_timm(tm)
    img = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        assert (mine(img) - tm(img)).abs().max().item() < 1e-4


def test_clip_tiny_random_matches_hf():
    tc = dict(vocab_size=99, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
              num_attention_heads=4, max_position_embeddings=16, eos_token_id=98)
    vc = dict(hidden_size=48, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4,
              image_size=32, patch_size=8)
    hf = transformers.CLIPModel(transformers.CLIPConfig(text_config=tc, vision_config=vc,
                                                        projection_dim=24)).eval()
    for p in hf.parameters():
        torch.nn.init.normal_(p, std=0.05)
    mine = clip_from_hf(hf)
    img = torch.randn(3, 3, 32, 32)
    ids = torch.randint(0, 97, (3, 16))
    eot = torch.tensor([15, 9, 4])
    ids[torch.arange(3), eot] = 98
    mask = torch.arange(16)[None] <= eot[:, None]
    with torch.no_grad():
        ref_im = hf.get_image_features(pixel_values=img)
        ref_tx = hf.get_text_features(input_ids=ids, attention_mask=mask.long())
        my_im = mine.visual(img)
        my_tx = mine.text(ids, eot)
    if not torch.is_tensor(ref_im):    # newer transformers return a ModelOutput
        ref_im, ref_tx = ref_im.pooler_output, ref_tx.pooler_output
    assert (my_im - ref_im).abs().max().item() < 1e-4
    # causal mask means tokens after EOT cannot change the EOT state, so no padding mask needed
    assert (my_tx - ref_tx).abs().max().item() < 1e-4
