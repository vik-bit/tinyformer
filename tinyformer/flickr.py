"""Flickr8k (HF mirror jxie/flickr8k, 6000/1000/1000 split, 5 captions per image)."""
from __future__ import annotations

import io
from pathlib import Path

DIR = Path(__file__).resolve().parents[1] / "data" / "flickr8k"


def load_split(split: str):
    """Returns (list of PIL images, list of 5-caption lists). Images decoded lazily by caller."""
    import pyarrow.parquet as pq
    from PIL import Image

    files = sorted(DIR.glob(f"{split}-*.parquet"))
    assert files, f"no parquet files for {split} in {DIR}"
    images, caps = [], []
    for f in files:
        t = pq.read_table(f).to_pydict()
        for i, img in enumerate(t["image"]):
            images.append(img["bytes"])
            caps.append([t[f"caption_{k}"][i] for k in range(5)])
    return images, caps


def open_image(b: bytes):
    from PIL import Image
    return Image.open(io.BytesIO(b)).convert("RGB")
