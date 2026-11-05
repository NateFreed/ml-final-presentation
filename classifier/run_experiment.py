"""Run the full experiment end to end.

Order:
1. Generate the fixed dataset (if missing, or with --force-data)
2. Train Model A and Model B
3. Evaluate by step and negative class
4. Write presentation figures
5. Render false positives / false negatives with LifeGrid.view
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from . import config
from .dataset import generate_dataset
from .evaluate import evaluate_model
from .inspect_mistakes import render_mistakes
from .dataset import load_raw
from .plots import plot_example_boards, plot_headline_overlay, plot_step_curves
from .train import train_one_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Conway CNN experiment")
    parser.add_argument("--force-data", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    data_path = config.DATA_DIR / "dataset.npz"
    if args.force_data or not data_path.exists():
        generate_dataset(out_path=data_path)
    else:
        print(f"Using existing dataset at {data_path}")

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")
    raw = load_raw()

    if not args.skip_train:
        train_one_model(
            "model_a",
            config.MODEL_A_NEGATIVES,
            config.TRAIN_SEED_A,
            device,
            raw,
        )
        train_one_model(
            "model_b",
            config.MODEL_B_NEGATIVES,
            config.TRAIN_SEED_B,
            device,
            raw,
        )

    for name in ("model_a", "model_b"):
        evaluate_model(name, device, raw)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_example_boards()
    for name in ("model_a", "model_b"):
        plot_step_curves(name, metric="accuracy")
        plot_step_curves(name, metric="auc")
        plot_headline_overlay(name)

    for name in ("model_a", "model_b"):
        for neg in ("random", "highlife", "density_matched", "perlin", "live_free_or_die"):
            for step in (1, 5, 15):
                render_mistakes(name, neg, step, device, raw, top_k=8)

    print("\nDone. Figures and mistake panels are under:", config.OUTPUT_DIR)


if __name__ == "__main__":
    main()
