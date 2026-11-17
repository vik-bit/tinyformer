"""Load timm's ViT-B/16 (ImageNet-1k) into tinyformer.ViT and compare logits.

Inputs: 8 real photos if a folder is given (resized/normalised with timm's own
transform), else seeded random tensors. Writes results/verify_vit.json.
  python scripts/verify_vit.py [image_dir]
"""
import json
import sys
import time
from pathlib import Path

import timm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tinyformer.convert import vit_from_timm  # noqa: E402

NAME = "vit_base_patch16_224.augreg2_in21k_ft_in1k"


def main():
    torch.set_grad_enabled(False)
    tm = timm.create_model(NAME, pretrained=True).eval()
    mine = vit_from_timm(tm)
    if len(sys.argv) > 1:
        from PIL import Image
        cfg = timm.data.resolve_data_config({}, model=tm)
        tf = timm.data.create_transform(**cfg)
        files = sorted(Path(sys.argv[1]).glob("*"))[:8]
        x = torch.stack([tf(Image.open(f).convert("RGB")) for f in files])
        src = f"{len(files)} images from {sys.argv[1]}"
    else:
        torch.manual_seed(0)
        x = torch.randn(8, 3, 224, 224)
        src = "8 seeded N(0,1) tensors"
    a, b = tm(x), mine(x)
    t0 = time.perf_counter(); tm(x); t_ref = time.perf_counter() - t0
    t0 = time.perf_counter(); mine(x); t_mine = time.perf_counter() - t0
    out = {"model": NAME, "params": sum(p.numel() for p in mine.parameters()), "inputs": src,
           "max_abs_logit_diff": (a - b).abs().max().item(),
           "max_abs_prob_diff": (a.softmax(-1) - b.softmax(-1)).abs().max().item(),
           "top1_agreement": f"{(a.argmax(-1) == b.argmax(-1)).sum().item()}/{len(x)}",
           "top1_ref": a.argmax(-1).tolist(),
           "cpu_seconds_batch8": {"timm": t_ref, "tinyformer": t_mine},
           "timm": timm.__version__, "torch": torch.__version__, "measured": time.strftime("%Y-%m-%d")}
    Path("results").mkdir(exist_ok=True)
    Path("results/verify_vit.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
