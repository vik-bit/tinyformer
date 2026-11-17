"""Load OpenAI CLIP ViT-B/32 (HF openai/clip-vit-base-patch32) into tinyformer.CLIP.

Compares image embeddings, text embeddings and image-text logits with HF on
16 Flickr8k test images and their 80 captions (HF's own processor/tokenizer
prepares the inputs for both). Writes results/verify_clip.json.
"""
import json
import sys
import time
from pathlib import Path

import torch
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tinyformer.convert import clip_from_hf  # noqa: E402
from tinyformer.flickr import load_split, open_image  # noqa: E402

NAME = "openai/clip-vit-base-patch32"


def feats(x):
    return x if torch.is_tensor(x) else x.pooler_output


def main():
    torch.set_grad_enabled(False)
    hf = CLIPModel.from_pretrained(NAME).eval()
    proc = CLIPProcessor.from_pretrained(NAME)
    mine = clip_from_hf(hf)
    imgs, caps = load_split("test")
    images = [open_image(b) for b in imgs[:16]]
    texts = [c for cs in caps[:16] for c in cs]
    inp = proc(text=texts, images=images, return_tensors="pt", padding=True)
    ids, mask = inp["input_ids"], inp["attention_mask"]
    eot = mask.sum(1) - 1                        # last real token is <|endoftext|>
    assert (ids[torch.arange(len(ids)), eot] == proc.tokenizer.eos_token_id).all()
    ref = hf(**inp)
    my_im, my_tx = mine.visual(inp["pixel_values"]), mine.text(ids, eot)
    ref_im = feats(hf.get_image_features(pixel_values=inp["pixel_values"]))
    ref_tx = feats(hf.get_text_features(input_ids=ids, attention_mask=mask))
    my_logits = mine.logit_scale.exp() * mine.encode_image(inp["pixel_values"]) @ mine.encode_text(ids, eot).t()
    out = {
        "model": NAME, "inputs": "16 Flickr8k test images x 80 captions (HF CLIPProcessor)",
        "max_abs_diff_image_embed": (my_im - ref_im).abs().max().item(),
        "max_abs_diff_text_embed": (my_tx - ref_tx).abs().max().item(),
        "max_abs_diff_logits_per_image": (my_logits - ref.logits_per_image).abs().max().item(),
        "argmax_caption_agreement": f"{(my_logits.argmax(1) == ref.logits_per_image.argmax(1)).sum().item()}/16",
        "logit_scale": mine.logit_scale.exp().item(),
        "torch": torch.__version__, "measured": time.strftime("%Y-%m-%d"),
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/verify_clip.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    # keep a few photos (not committed) to run the ViT check on real images
    d = Path("data/sample_images")
    d.mkdir(parents=True, exist_ok=True)
    for i, im in enumerate(images[:8]):
        im.save(d / f"flickr8k_test_{i}.jpg")


if __name__ == "__main__":
    main()
