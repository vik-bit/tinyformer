"""Embed the Flickr8k test split (1000 images, 5000 captions) with a tinyformer CLIP.

  python scripts/embed_flickr.py pretrained            # OpenAI ViT-B/32 weights in tinyformer.CLIP
  python scripts/embed_flickr.py scratch runs/clip/model.pt

Writes data/emb/{name}_img.npy, {name}_txt.npy (L2-normalised float32) and
{name}_gt.npy (image index for every caption).
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tinyformer.flickr import load_split, open_image  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "data" / "emb"


def batched(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def main():
    torch.set_grad_enabled(False)
    torch.set_num_threads(int(__import__("os").environ.get("THREADS", "4")))
    name = sys.argv[1]
    imgs, caps = load_split("test")
    texts = [c for cs in caps for c in cs]
    gt = np.repeat(np.arange(len(imgs)), 5)
    if name == "pretrained":
        from transformers import CLIPModel, CLIPProcessor
        from tinyformer.convert import clip_from_hf
        proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = clip_from_hf(CLIPModel.from_pretrained("openai/clip-vit-base-patch32"))

        def enc_img(batch):
            px = proc(images=[open_image(b) for b in batch], return_tensors="pt")["pixel_values"]
            return model.encode_image(px)

        def enc_txt(batch):
            t = proc.tokenizer(batch, padding=True, truncation=True, max_length=77, return_tensors="pt")
            return model.encode_text(t["input_ids"], t["attention_mask"].sum(1) - 1)
    else:
        from tinyformer.clip_train import load_scratch
        model, img_tf, tokenize = load_scratch(sys.argv[2])

        def enc_img(batch):
            return model.encode_image(torch.stack([img_tf(open_image(b)) for b in batch]))

        def enc_txt(batch):
            ids, eot = tokenize(batch)
            return model.encode_text(ids, eot)
    model.eval()
    t0 = time.perf_counter()
    im = torch.cat([enc_img(b) for b in batched(imgs, 32)]).numpy().astype(np.float32)
    t_img = time.perf_counter() - t0
    t0 = time.perf_counter()
    tx = torch.cat([enc_txt(b) for b in batched(texts, 128)]).numpy().astype(np.float32)
    t_txt = time.perf_counter() - t0
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / f"{name}_img.npy", im)
    np.save(OUT / f"{name}_txt.npy", tx)
    np.save(OUT / f"{name}_gt.npy", gt)
    print(f"{name}: {im.shape} images in {t_img:.0f}s, {tx.shape} captions in {t_txt:.0f}s")


if __name__ == "__main__":
    main()
