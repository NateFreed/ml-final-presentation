"""Figures for the presentation.

Main deliverable: accuracy (and AUC) versus Conway step count, for each
negative class, with the density baseline overlaid. Also: example boards
so the audience can see what each generator looks like at early and late steps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from game_of_life import LifeGrid

from . import config
from .dataset import CLASS_TO_ID, load_raw


def _load_metrics(model_name: str) -> dict:
    path = config.OUTPUT_DIR / f"{model_name}_metrics.json"
    with open(path) as f:
        return json.load(f)


def plot_step_curves(model_name: str, metric: str = "accuracy") -> Path:
    """One line per negative class for CNN and baseline."""
    metrics = _load_metrics(model_name)
    held = set(config.HELD_OUT_STEPS)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, kind, title in [
        (axes[0], "cnn", f"{model_name}: CNN"),
        (axes[1], "baseline", f"{model_name}: density baseline"),
    ]:
        key = "accuracy" if metric == "accuracy" else "auc"
        for neg in config.NEGATIVE_CLASSES:
            xs = []
            ys = []
            ylo = []
            yhi = []
            for step in config.STEPS:
                row = metrics["by_negative"][neg][str(step)][kind]
                xs.append(step)
                ys.append(row[key])
                if metric == "accuracy":
                    ylo.append(row["acc_low"])
                    yhi.append(row["acc_high"])
            (line,) = ax.plot(xs, ys, label=neg, linewidth=2)
            if metric == "accuracy":
                ax.fill_between(xs, ylo, yhi, color=line.get_color(), alpha=0.15)
            # Mark held-out training steps
            hx = [s for s in xs if s in held]
            hy = [ys[xs.index(s)] for s in hx]
            ax.scatter(hx, hy, color=line.get_color(), s=28, zorder=3)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance")
        ax.set_xlabel("Conway steps from random noise (k)")
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.set_xlim(1, config.MAX_STEPS)
        ax.set_ylim(0.45, 1.02)
        ax.grid(True, alpha=0.3)
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle(
        f"Distinguishing Conway boards from lookalikes vs step count\n"
        f"(dots = steps held out of training)",
        fontsize=12,
    )
    fig.tight_layout()
    out = config.OUTPUT_DIR / f"{model_name}_{metric}_by_step.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")
    return out


def plot_headline_overlay(model_name: str = "model_a") -> Path:
    """CNN vs baseline on the headline comparison: Conway vs fresh random."""
    metrics = _load_metrics(model_name)
    neg = "random"
    steps = config.STEPS
    cnn = [metrics["by_negative"][neg][str(s)]["cnn"]["accuracy"] for s in steps]
    base = [metrics["by_negative"][neg][str(s)]["baseline"]["accuracy"] for s in steps]
    cnn_lo = [metrics["by_negative"][neg][str(s)]["cnn"]["acc_low"] for s in steps]
    cnn_hi = [metrics["by_negative"][neg][str(s)]["cnn"]["acc_high"] for s in steps]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, cnn, label="CNN", linewidth=2)
    ax.fill_between(steps, cnn_lo, cnn_hi, alpha=0.2)
    ax.plot(steps, base, label="Density baseline", linewidth=2)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance")
    ax.set_xlabel("Steps from random noise (k)")
    ax.set_ylabel("Accuracy")
    ax.set_title(
        f"{model_name}: Conway after k steps vs fresh random noise\n"
        "(If CNN ≈ baseline, density alone may explain the result)"
    )
    ax.set_ylim(0.45, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = config.OUTPUT_DIR / f"{model_name}_headline_random.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")
    return out


def plot_example_boards() -> Path:
    """Show what each generator looks like at a few step counts."""
    raw = load_raw()
    show_steps = [1, 5, 15, 30]
    classes = config.CLASS_NAMES
    # Use the first test soup for a reproducible panel.
    test_soup = raw["soup_ids"][raw["split_ids"] == 2][0]
    fig, axes = plt.subplots(
        len(classes),
        len(show_steps),
        figsize=(2.2 * len(show_steps), 2.0 * len(classes)),
    )
    for r, class_name in enumerate(classes):
        cid = CLASS_TO_ID[class_name]
        for c, step in enumerate(show_steps):
            mask = (
                (raw["split_ids"] == 2)
                & (raw["soup_ids"] == test_soup)
                & (raw["class_ids"] == cid)
                & (raw["steps"] == step)
            )
            board = raw["boards"][np.flatnonzero(mask)[0]]
            ax = axes[r, c]
            LifeGrid.view(board, ax=ax)
            if r == 0:
                ax.set_title(f"k={step}")
            if c == 0:
                ax.set_ylabel(class_name, rotation=90, size=9)
    fig.suptitle(f"Example boards from one test soup (grid {config.GRID_SIZE}×{config.GRID_SIZE})")
    fig.tight_layout()
    out = config.OUTPUT_DIR / "example_boards.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Make presentation figures")
    parser.add_argument("--model", choices=["A", "B", "both"], default="both")
    args = parser.parse_args()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_example_boards()
    names = []
    if args.model in ("A", "both"):
        names.append("model_a")
    if args.model in ("B", "both"):
        names.append("model_b")
    for name in names:
        plot_step_curves(name, metric="accuracy")
        plot_step_curves(name, metric="auc")
        plot_headline_overlay(name)


if __name__ == "__main__":
    main()
