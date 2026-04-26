"""
Boxplot of mean message length per seed across architectures.

Loads messages_epoch100.npy from each results/{arch}/seed_{seed}/ folder.
Message length = index of first 0 (EOS/pad) token; full max_len if no 0 present.

Usage:
    python scripts/plot_message_length.py
    python scripts/plot_message_length.py --results_dir results --epoch 100
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ARCHS = ["lstm", "gru", "transformer", "mlp"]
COLORS = {"lstm": "#2196F3", "gru": "#4CAF50", "transformer": "#FF5722", "mlp": "#9C27B0"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results")
    p.add_argument("--epoch", type=int, default=100, help="Which epoch's messages to load")
    p.add_argument("--output", default=None)
    return p.parse_args()


def message_lengths(messages: np.ndarray) -> np.ndarray:
    """
    messages: (N, max_len) int array
    Returns (N,) array of lengths — position of first 0, or max_len if no 0.
    """
    n, max_len = messages.shape
    lengths = np.full(n, max_len, dtype=float)
    for i, msg in enumerate(messages):
        zeros = np.where(msg == 0)[0]
        if len(zeros):
            lengths[i] = zeros[0]
    return lengths


def load_arch_data(results_dir: Path, arch: str, epoch: int) -> list[float]:
    """Returns list of per-seed mean message lengths."""
    arch_dir = results_dir / arch
    if not arch_dir.exists():
        return []

    means = []
    for seed_dir in sorted(arch_dir.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
            continue
        npy_path = seed_dir / f"messages_epoch{epoch}.npy"
        if not npy_path.exists():
            print(f"  missing {npy_path.relative_to(results_dir)}")
            continue
        messages = np.load(npy_path)
        if messages.ndim != 2:
            print(f"  unexpected shape {messages.shape} in {npy_path}, skipping")
            continue
        lengths = message_lengths(messages)
        means.append(float(lengths.mean()))
        print(f"  {seed_dir.name} | n={len(messages)} | mean_len={lengths.mean():.2f} "
              f"| min={lengths.min():.0f} max={lengths.max():.0f}")
    return means


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_path = Path(args.output) if args.output else results_dir / "message_length.png"

    arch_means: dict[str, list[float]] = {}
    for arch in ARCHS:
        print(f"\n{arch}:")
        means = load_arch_data(results_dir, arch, args.epoch)
        if means:
            arch_means[arch] = means

    present = [a for a in ARCHS if a in arch_means]
    if not present:
        print(f"\nNo messages_epoch{args.epoch}.npy files found. "
              f"Run training first, or try --epoch <N> matching an eval checkpoint.")
        return

    # ------------------------------------------------------------------ plot
    fig, ax = plt.subplots(figsize=(7, 5))

    data = [arch_means[a] for a in present]
    positions = list(range(1, len(present) + 1))

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.45,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=5, linestyle="none", alpha=0.6),
    )

    for patch, arch in zip(bp["boxes"], present):
        patch.set_facecolor(COLORS[arch])
        patch.set_alpha(0.7)

    # overlay individual seed points
    for i, (arch, pos) in enumerate(zip(present, positions)):
        vals = arch_means[arch]
        jitter = np.random.default_rng(i).uniform(-0.08, 0.08, len(vals))
        ax.scatter(np.full(len(vals), pos) + jitter, vals,
                   color=COLORS[arch], zorder=3, s=40, edgecolors="white", linewidths=0.5)

    ax.set_xticks(positions)
    ax.set_xticklabels(present, fontsize=11)
    ax.set_ylabel("Mean message length (tokens)", fontsize=11)
    ax.set_title(f"Message Length Distribution at Epoch {args.epoch}\n"
                 f"(0 = EOS/pad, each point = one seed)", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
