"""
plot_training_curves.py — Visualize the full training run from the log.

Parses the training log produced by train.py and plots four panels:
  1. Total loss per epoch (with id/triplet/center components)
  2. Train accuracy per epoch
  3. Eval Rank-1 per evaluation
  4. Eval mAP per evaluation

The first three panels are stacked or paired; eval Rank-1 and mAP are
overlaid on the same axes (right plot) so the train/eval gap is visible
at a glance — that gap is the diagnostic signal for overfitting.

Usage:
    # Default: reads ./train.log, writes training_curves.png
    python plot_training_curves.py

    # Or specify paths:
    python plot_training_curves.py --log path/to/train.log --out curves.png
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

# Per-epoch summary line, e.g.:
# "06:45:48  INFO  Epoch 100 done [57s] | avg loss=1.2066 | train acc=99.9%"
EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)\s+done\s+\[\d+s\]\s+\|\s+"
    r"avg loss=([\d.]+)\s+\|\s+"
    r"train acc=([\d.]+)%"
)

# Mid-epoch step lines carry the loss component breakdown:
# "Epoch  10 | step   50/93 | loss=3.9742 (id=3.0249, tri=0.7779, cen=342.6548) | acc=55.4%"
STEP_RE = re.compile(
    r"Epoch\s+(\d+)\s+\|\s+step\s+\d+/\d+\s+\|\s+"
    r"loss=[\d.]+\s+"
    r"\(id=([\d.]+),\s+tri=([\d.]+),\s+cen=([\d.]+)\)"
)

# Eval lines, e.g.:
# "[Eval] Epoch  10 — Rank-1: 62.05%  mAP: 36.64%"
EVAL_RE = re.compile(
    r"\[Eval\]\s+Epoch\s+(\d+)\s+.*?Rank-1:\s+([\d.]+)%\s+mAP:\s+([\d.]+)%"
)


def parse_log(text: str):
    """Walks the log once, harvests all four series."""
    epochs, train_loss, train_acc        = [], [], []
    id_loss, tri_loss, cen_loss          = {}, {}, {}
    eval_epochs, eval_rank1, eval_map    = [], [], []

    for line in text.splitlines():
        m = EPOCH_RE.search(line)
        if m:
            epochs.append(int(m.group(1)))
            train_loss.append(float(m.group(2)))
            train_acc.append(float(m.group(3)))
            continue

        m = STEP_RE.search(line)
        if m:
            ep = int(m.group(1))
            # Last step line per epoch wins (it has the running avg)
            id_loss[ep]  = float(m.group(2))
            tri_loss[ep] = float(m.group(3))
            cen_loss[ep] = float(m.group(4))
            continue

        m = EVAL_RE.search(line)
        if m:
            eval_epochs.append(int(m.group(1)))
            eval_rank1.append(float(m.group(2)))
            eval_map.append(float(m.group(3)))

    # Align id/tri/cen series with epochs list
    id_series  = [id_loss.get(e)  for e in epochs]
    tri_series = [tri_loss.get(e) for e in epochs]
    cen_series = [cen_loss.get(e) for e in epochs]

    return {
        "epochs":      epochs,
        "train_loss":  train_loss,
        "train_acc":   train_acc,
        "id_loss":     id_series,
        "tri_loss":    tri_series,
        "cen_loss":    cen_series,
        "eval_epochs": eval_epochs,
        "eval_rank1":  eval_rank1,
        "eval_map":    eval_map,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot(d: dict, out_path: str):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # --- (0, 0): total loss + components ---
    ax = axes[0, 0]
    ax.plot(d["epochs"], d["train_loss"], color="#21295C",
            linewidth=2.0, label="Total loss (avg)")
    ax.plot(d["epochs"], d["id_loss"],   color="#1C7293",
            linewidth=1.3, alpha=0.85, label="ID loss")
    ax.plot(d["epochs"], d["tri_loss"],  color="#5BA969",
            linewidth=1.3, alpha=0.85, label="Triplet loss")
    ax.set_title("Training loss", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- (0, 1): center loss on its own axis (very different scale) ---
    ax = axes[0, 1]
    ax.plot(d["epochs"], d["cen_loss"], color="#F2A65A", linewidth=2.0)
    ax.set_title("Center loss (separate scale)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Center loss")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    # Annotate the early collapse — center loss drops three orders of magnitude
    ax.text(0.97, 0.95,
            "Note: log scale.\nDrops sharply early as\ncentroids stabilise.",
            transform=ax.transAxes, fontsize=9,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#888", alpha=0.9))

    # --- (1, 0): train accuracy ---
    ax = axes[1, 0]
    ax.plot(d["epochs"], d["train_acc"], color="#065A82", linewidth=2.0)
    ax.set_title("Train accuracy (closed-set, 751 IDs)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 102)
    ax.grid(True, alpha=0.3)
    ax.axhline(99, color="#888", linestyle=":", linewidth=1, alpha=0.6)
    ax.text(d["epochs"][-1], 99, "  99%", color="#666", fontsize=9,
            verticalalignment="center")

    # --- (1, 1): eval Rank-1 + mAP overlaid (the diagnostic plot) ---
    ax = axes[1, 1]
    ax.plot(d["eval_epochs"], d["eval_rank1"],
            "-o", color="#065A82", linewidth=2.0, markersize=5,
            label="Rank-1")
    ax.plot(d["eval_epochs"], d["eval_map"],
            "-s", color="#C45A5A", linewidth=2.0, markersize=5,
            label="mAP")
    ax.set_title("Eval Rank-1 & mAP (open-set, 750 IDs)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Annotate best Rank-1
    best_idx = int(max(range(len(d["eval_rank1"])),
                       key=lambda i: d["eval_rank1"][i]))
    bx = d["eval_epochs"][best_idx]
    by = d["eval_rank1"][best_idx]
    ax.annotate(f"Best: {by:.2f}% @ epoch {bx}",
                xy=(bx, by),
                xytext=(bx - 30, by - 18),
                fontsize=9, color="#065A82",
                arrowprops=dict(arrowstyle="->", color="#065A82", alpha=0.7))

    fig.suptitle("ReID training run — 120 epochs, RGB primary input",
                 fontsize=15, fontweight="bold", y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="train.log",
                   help="Path to training log file (default: train.log)")
    p.add_argument("--out", default="training_curves.png",
                   help="Output image path (default: training_curves.png)")
    args = p.parse_args()

    text = Path(args.log).read_text()
    d = parse_log(text)

    print(f"Parsed {len(d['epochs'])} epochs and "
          f"{len(d['eval_epochs'])} evaluations.")
    print(f"Final train acc: {d['train_acc'][-1]:.2f}%  |  "
          f"Final Rank-1: {d['eval_rank1'][-1]:.2f}%  |  "
          f"Final mAP: {d['eval_map'][-1]:.2f}%")

    plot(d, args.out)


if __name__ == "__main__":
    main()