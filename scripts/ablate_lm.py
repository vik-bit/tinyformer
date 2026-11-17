"""CPU ablations: positions (learned vs RoPE), attention (MHA vs GQA), vocab size.

Each config x seed is a separate train_lm.py run; small model so it fits a laptop CPU.
  python scripts/ablate_lm.py run      # trains (skips runs that already finished)
  python scripts/ablate_lm.py report   # results/ablations.json + results/ablation_curves.png
"""
import json
import statistics as st
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "ablate"
COMMON = ["--d_model", "128", "--n_layer", "4", "--n_head", "4", "--seq", "128", "--batch", "32",
          "--steps", "800", "--warmup", "50", "--lr", "2e-3", "--eval_every", "100",
          "--eval_batches", "20", "--threads", "8"]
CONFIGS = {
    "learned_mha_v4096": ["--vocab", "4096", "--positions", "learned"],
    "rope_mha_v4096": ["--vocab", "4096", "--positions", "rope"],
    "rope_gqa2_v4096": ["--vocab", "4096", "--positions", "rope", "--n_kv_head", "2"],
    "learned_mha_v1024": ["--vocab", "1024", "--positions", "learned"],
    "learned_mha_v8192": ["--vocab", "8192", "--positions", "learned"],
}
SEEDS = [0, 1]


def run():
    for name, extra in CONFIGS.items():
        for s in SEEDS:
            out = RUNS / f"{name}_s{s}"
            if (out / "model.pt").exists():
                continue
            cmd = [sys.executable, str(ROOT / "scripts" / "train_lm.py"), *COMMON, *extra,
                   "--seed", str(s), "--out", str(out)]
            print(" ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)


def load(out):
    evals = [json.loads(l) for l in open(out / "log.jsonl") if "val_loss" in l]
    meta = json.loads((out / "meta.json").read_text())
    return evals, meta


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    panels = [("positions", ["learned_mha_v4096", "rope_mha_v4096"], "val_loss"),
              ("attention", ["rope_mha_v4096", "rope_gqa2_v4096"], "val_loss"),
              ("vocab size", ["learned_mha_v1024", "learned_mha_v4096", "learned_mha_v8192"], "val_bpb")]
    for name in CONFIGS:
        finals, finals_bpb, curves = [], [], []
        for s in SEEDS:
            out = RUNS / f"{name}_s{s}"
            if not (out / "model.pt").exists():
                continue
            ev, meta = load(out)
            finals.append(ev[-1]["val_loss"])
            finals_bpb.append(ev[-1]["val_bpb"])
            curves.append(ev)
        if not finals:
            continue
        rows[name] = {"params": meta["params"], "seeds": len(finals),
                      "val_loss": finals, "val_loss_mean": st.mean(finals),
                      "val_bpb": finals_bpb, "val_bpb_mean": st.mean(finals_bpb),
                      "train_seconds_mean": st.mean(c[-1]["elapsed_s"] for c in curves),
                      "tokens_seen": curves[0][-1]["tokens_seen"], "curves": curves}
    for ax, (title, names, key) in zip(axes, panels):
        for n in names:
            if n not in rows:
                continue
            for i, c in enumerate(rows[n]["curves"]):
                ax.plot([e["step"] for e in c][1:], [e[key] for e in c][1:], color=f"C{names.index(n)}",
                        alpha=0.9 if i == 0 else 0.5, label=n if i == 0 else None)
        ax.set_title(title)
        ax.set_xlabel("step (32 x 128 tokens)")
        ax.set_ylabel(key.replace("_", " ") + (" (bits/byte)" if key == "val_bpb" else " (nats/token)"))
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    (ROOT / "results").mkdir(exist_ok=True)
    fig.savefig(ROOT / "results" / "ablation_curves.png", dpi=110)
    slim = {k: {kk: vv for kk, vv in v.items() if kk != "curves"} for k, v in rows.items()}
    (ROOT / "results" / "ablations.json").write_text(json.dumps({"common_args": COMMON, "runs": slim}, indent=2))
    for k, v in slim.items():
        print(f"{k:22s} params={v['params']/1e6:.2f}M loss={v['val_loss']} bpb={v['val_bpb']}")


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]]()
