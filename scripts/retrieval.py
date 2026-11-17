"""Text -> image retrieval on Flickr8k test embeddings with vecdex (HNSW), checked against exact search.

Needs numpy and vecdex (github.com/vik-bit/vecdex; on Windows it builds under WSL):
  python scripts/retrieval.py pretrained scratch
For each embedding set: recall@1/5/10 from exact (brute-force) inner-product
search and from vecdex, 95% bootstrap CIs over the 5000 caption queries, and
how often vecdex's top-10 equals the exact top-10. Writes results/retrieval.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EMB = ROOT / "data" / "emb"
KS = (1, 5, 10)


def recall_hits(topk, gt):
    """(nq, k) labels -> dict K -> bool array of hits."""
    return {k: (topk[:, :k] == gt[:, None]).any(1) for k in KS}


def boot_ci(hits, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(hits), (n, len(hits)))
    means = hits[idx].mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def evaluate(name, ef=64):
    im, tx, gt = (np.load(EMB / f"{name}_{s}.npy") for s in ("img", "txt", "gt"))
    t0 = time.perf_counter()
    exact = np.argsort(-(tx @ im.T), axis=1, kind="stable")[:, :10]
    t_exact = time.perf_counter() - t0
    out = {"images": len(im), "queries": len(tx), "dim": im.shape[1]}
    ex_hits = recall_hits(exact, gt)
    out["exact"] = {f"R@{k}": float(h.mean()) for k, h in ex_hits.items()}
    out["exact"].update({f"R@{k}_ci95": boot_ci(h) for k, h in ex_hits.items()})
    out["exact"]["seconds_all_queries"] = t_exact
    # image -> text (5 correct captions per image), exact only
    i2t = np.argsort(-(im @ tx.T), axis=1)[:, :10]
    out["exact_image_to_text"] = {f"R@{k}": float((gt[i2t[:, :k]] == np.arange(len(im))[:, None]).any(1).mean())
                                  for k in KS}
    try:
        import vecdex
    except ImportError:
        out["vecdex"] = "not installed"
        return out
    idx = vecdex.Index(dim=im.shape[1], max_elements=len(im), metric="ip", M=16, ef_construction=200)
    t0 = time.perf_counter()
    idx.add(im)
    t_build = time.perf_counter() - t0
    t0 = time.perf_counter()
    labels, _ = idx.search(tx, k=10, ef=ef)
    t_search = time.perf_counter() - t0
    labels = np.asarray(labels)
    vx_hits = recall_hits(labels, gt)
    out["vecdex"] = {f"R@{k}": float(h.mean()) for k, h in vx_hits.items()}
    out["vecdex"].update({f"R@{k}_ci95": boot_ci(h) for k, h in vx_hits.items()})
    out["vecdex"].update({
        "params": {"M": 16, "ef_construction": 200, "ef": ef, "metric": "ip"},
        "build_seconds": t_build, "search_seconds_all_queries": t_search,
        "top10_identical_to_exact": float((labels == exact).all(1).mean()),
        "top10_set_overlap_with_exact": float(np.mean([len(set(a) & set(b)) / 10 for a, b in zip(labels, exact)])),
        "version": getattr(vecdex, "__version__", "?"), "simd": str(vecdex.simd),
    })
    return out


def main():
    res_path = ROOT / "results" / "retrieval.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    for name in sys.argv[1:]:
        res[name] = evaluate(name)
        print(name, json.dumps({k: v for k, v in res[name].items()}, indent=1))
    res_path.parent.mkdir(exist_ok=True)
    res_path.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
