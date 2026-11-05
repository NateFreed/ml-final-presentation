"""Browse false positives and false negatives with LifeGrid.view.

Uses the teammates' board viewer so the presentation can show the same
visual language for demos and for model mistakes.

For each model and each negative class, at a chosen step, we draw the
highest-confidence false positives (negative boards scored as Conway) and
false negatives (Conway boards scored as negative).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from game_of_life import LifeGrid

from . import config
from .dataset import CLASS_TO_ID, load_raw
from .evaluate import cnn_probs, load_cnn


def _load_threshold(model_name: str) -> float:
    metrics_path = config.OUTPUT_DIR / f"{model_name}_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return float(json.load(f).get("cnn_threshold", 0.5))
    return 0.5


def find_mistakes(
    model_name: str,
    neg_name: str,
    step: int,
    device: torch.device,
    raw: dict,
    top_k: int = 8,
) -> dict:
    model = load_cnn(config.OUTPUT_DIR / f"{model_name}.pt", device)
    threshold = _load_threshold(model_name)
    conway_id = CLASS_TO_ID["conway"]
    neg_id = CLASS_TO_ID[neg_name]
    mask = (
        (raw["split_ids"] == 2)
        & (raw["steps"] == step)
        & np.isin(raw["class_ids"], [conway_id, neg_id])
    )
    idxs = np.flatnonzero(mask)
    boards = raw["boards"][idxs]
    labels = raw["labels"][idxs]
    class_ids = raw["class_ids"][idxs]
    probs = cnn_probs(model, boards, device)
    preds = probs >= threshold

    fp = np.flatnonzero((labels == 0) & preds)
    fn = np.flatnonzero((labels == 1) & ~preds)

    fp = fp[np.argsort(probs[fp])[::-1][:top_k]]
    fn = fn[np.argsort(probs[fn])[:top_k]]  # lowest P(conway) among missed Conway

    return {
        "boards": boards,
        "probs": probs,
        "class_ids": class_ids,
        "labels": labels,
        "fp_local": fp,
        "fn_local": fn,
        "dataset_indices": idxs,
        "threshold": threshold,
    }


def _panel(boards, probs, class_ids, local_idxs, title: str, out_path: Path) -> None:
    if len(local_idxs) == 0:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.axis("off")
        ax.set_title(f"{title}\n(none)")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Wrote {out_path} (empty)")
        return

    n = len(local_idxs)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.8 * rows))
    axes = np.array(axes).reshape(-1)
    id_to_name = {i: n for i, n in enumerate(config.CLASS_NAMES)}
    for ax in axes[n:]:
        ax.axis("off")
    for ax, loc in zip(axes, local_idxs):
        board = boards[loc]
        LifeGrid.view(board, ax=ax)
        cname = id_to_name[int(class_ids[loc])]
        ax.set_title(f"{cname}\nP(conway)={probs[loc]:.2f}", fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def render_mistakes(
    model_name: str,
    neg_name: str,
    step: int,
    device: torch.device,
    raw: dict,
    top_k: int = 8,
) -> None:
    found = find_mistakes(model_name, neg_name, step, device, raw, top_k=top_k)
    out_dir = config.OUTPUT_DIR / "mistakes"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{model_name}_vs_{neg_name}_step{step}"
    _panel(
        found["boards"],
        found["probs"],
        found["class_ids"],
        found["fp_local"],
        title=f"{model_name} false positives vs {neg_name} at k={step}\n"
        f"(negative boards the model called Conway)",
        out_path=out_dir / f"{stem}_fp.png",
    )
    _panel(
        found["boards"],
        found["probs"],
        found["class_ids"],
        found["fn_local"],
        title=f"{model_name} false negatives vs {neg_name} at k={step}\n"
        f"(Conway boards the model missed)",
        out_path=out_dir / f"{stem}_fn.png",
    )

    # Also dump a small JSON index so the notebook can list them.
    summary = {
        "model": model_name,
        "negative": neg_name,
        "step": step,
        "false_positives": [
            {
                "dataset_index": int(found["dataset_indices"][i]),
                "prob": float(found["probs"][i]),
                "class": config.CLASS_NAMES[int(found["class_ids"][i])],
            }
            for i in found["fp_local"]
        ],
        "false_negatives": [
            {
                "dataset_index": int(found["dataset_indices"][i]),
                "prob": float(found["probs"][i]),
                "class": config.CLASS_NAMES[int(found["class_ids"][i])],
            }
            for i in found["fn_local"]
        ],
    }
    with open(out_dir / f"{stem}.json", "w") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render FP/FN boards with LifeGrid.view"
    )
    parser.add_argument("--model", choices=["A", "B", "both"], default="both")
    parser.add_argument(
        "--negatives",
        nargs="+",
        default=["random", "highlife", "density_matched"],
        help="Negative classes to inspect",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=[1, 5, 15],
        help="Step counts to inspect",
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    raw = load_raw()
    names = []
    if args.model in ("A", "both"):
        names.append("model_a")
    if args.model in ("B", "both"):
        names.append("model_b")

    for name in names:
        for neg in args.negatives:
            for step in args.steps:
                render_mistakes(
                    name, neg, step, device, raw, top_k=args.top_k
                )


if __name__ == "__main__":
    main()
