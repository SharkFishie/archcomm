"""
Plot val_acc and topo_rho learning curves across seeds per architecture.

Usage:
    python scripts/plot_learning_curves.py
    python scripts/plot_learning_curves.py --results_dir results --output results/learning_curves.png
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ARCHS = ["lstm", "gru", "transformer", "mlp"]
COLORS = {"lstm": "#2196F3", "gru": "#4CAF50", "transformer": "#FF5722", "mlp": "#9C27B0"}
METRICS = ["val_acc", "topo_rho"]
LABELS = {"val_acc": "Validation Accuracy", "topo_rho": "Topographic Similarity (ρ)"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results")
    p.add_argument("--output", default=None)
    return p.parse_args()


def load_runs(results_dir: Path) -> dict:
    """Returns {arch: [list of epoch dicts per seed]}."""
    runs = defaultdict(list)
    for arch_dir in sorted(results_dir.iterdir()):
        if not arch_dir.is_dir():
            continue
        arch = arch_dir.name
        for seed_dir in sorted(arch_dir.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue
            metrics_path = seed_dir / "metrics.json"
            if not metrics_path.exists():
                continue
            with open(metrics_path) as f:
                log = json.load(f)
            if log:
                runs[arch].append(log)
    return runs


def align_by_epoch(seed_logs: list[list[dict]], metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Align multiple per-seed logs by epoch, return (epochs, mean, std).
    Only epochs present in ALL seeds are included.
    """
    epoch_sets = [set(row["epoch"] for row in log) for log in seed_logs]
    common_epochs = sorted(set.intersection(*epoch_sets))

    values = []
    for log in seed_logs:
        by_epoch = {row["epoch"]: row.get(metric) for row in log}
        row_vals = [by_epoch[e] for e in common_epochs]
        # replace None (NaN topo_rho) with nan so it doesn't drag the mean
        row_vals = [v if v is not None else float("nan") for v in row_vals]
        values.append(row_vals)

    arr = np.array(values, dtype=float)          # (n_seeds, n_epochs)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    return np.array(common_epochs), mean, std


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_path = Path(args.output) if args.output else results_dir / "learning_curves.png"

    runs = load_runs(results_dir)
    if not runs:
        print("No metrics.json files found. Run training first.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Emergent Language — Learning Curves by Architecture", fontsize=13, y=1.01)

    for ax, metric in zip(axes, METRICS):
        plotted = False
        for arch in ARCHS:
            seed_logs = runs.get(arch)
            if not seed_logs:
                continue
            epochs, mean, std = align_by_epoch(seed_logs, metric)
            color = COLORS[arch]
            n = len(seed_logs)
            label = f"{arch} (n={n})"
            ax.plot(epochs, mean, color=color, linewidth=2, label=label, marker="o", markersize=4)
            ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.15)
            plotted = True

        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(LABELS[metric], fontsize=11)
        ax.set_title(LABELS[metric], fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)

        if metric == "val_acc":
            ax.axhline(0.2, color="gray", linestyle=":", linewidth=1, label="chance (0.2)")
            ax.set_ylim(bottom=0)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
