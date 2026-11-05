"""Train Model A (vs random) and Model B (vs all negatives).

Both models share the ConwayCNN architecture. They differ only in which
negative classes appear during training. Early stopping watches validation
ROC-AUC so we do not overfit the small boards.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler

from . import config
from .baseline import fit_density_baseline
from .dataset import BoardDataset, get_numpy_split, load_raw
from .model import ConwayCNN


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(
    dataset: BoardDataset,
    batch_size: int,
    shuffle: bool,
    balance: bool,
) -> DataLoader:
    sampler = None
    if balance:
        labels = dataset.labels
        n_pos = max(float((labels == 1).sum()), 1.0)
        n_neg = max(float((labels == 0).sum()), 1.0)
        weights = np.where(labels == 1, 1.0 / n_pos, 1.0 / n_neg).astype(np.float64)
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(weights),
            num_samples=len(weights),
            replacement=True,
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config.NUM_WORKERS,
    )


@torch.no_grad()
def collect_logits(model: ConwayCNN, loader: DataLoader, device: torch.device):
    model.eval()
    logits_all = []
    labels_all = []
    for boards, labels, _, _ in loader:
        boards = boards.to(device)
        logits = model(boards)
        logits_all.append(logits.cpu().numpy())
        labels_all.append(labels.numpy())
    return np.concatenate(logits_all), np.concatenate(labels_all)


def roc_auc_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    probs = 1.0 / (1.0 + np.exp(-logits))
    return float(roc_auc_score(labels, probs))


def train_one_model(
    name: str,
    negative_classes: list[str],
    seed: int,
    device: torch.device,
    raw: dict,
) -> dict:
    print(f"\n=== Training {name}: positives=conway, negatives={negative_classes} ===")
    set_seed(seed)
    t_all0 = time.perf_counter()
    train_ds = BoardDataset(raw, "train", negative_classes, include_held_out_steps=False)
    val_ds = BoardDataset(raw, "val", negative_classes, include_held_out_steps=False)
    train_loader = make_loader(train_ds, config.BATCH_SIZE, shuffle=True, balance=True)
    val_loader = make_loader(val_ds, config.BATCH_SIZE, shuffle=False, balance=False)
    print(f"  train size={len(train_ds)}  val size={len(val_ds)}  device={device}")

    model = ConwayCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()

    best_auc = -1.0
    best_state = None
    patience_left = config.PATIENCE
    history = []

    t_train0 = time.perf_counter()
    for epoch in range(1, config.MAX_EPOCHS + 1):
        epoch0 = time.perf_counter()
        model.train()
        running = 0.0
        n_batches = 0
        for boards, labels, _, _ in train_loader:
            boards = boards.to(device)
            labels = labels.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(boards)
            loss = loss_fn(logits, labels)
            loss.backward()
            opt.step()
            running += float(loss.item())
            n_batches += 1

        val_logits, val_labels = collect_logits(model, val_loader, device)
        val_auc = roc_auc_from_logits(val_logits, val_labels)
        train_loss = running / max(n_batches, 1)
        epoch_sec = time.perf_counter() - epoch0
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_auc": val_auc,
                "epoch_seconds": epoch_sec,
            }
        )
        print(
            f"  epoch {epoch:02d}  train_loss={train_loss:.4f}  "
            f"val_auc={val_auc:.4f}  ({epoch_sec:.1f}s)"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = config.PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  early stop at epoch {epoch} (best val_auc={best_auc:.4f})")
                break

    train_seconds = time.perf_counter() - t_train0
    assert best_state is not None
    model.load_state_dict(best_state)

    out_dir = config.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{name}.pt"
    timing = {
        "train_seconds": train_seconds,
        "train_minutes": train_seconds / 60.0,
        "n_epochs": len(history),
        "device": str(device),
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    torch.save(
        {
            "model_state": best_state,
            "name": name,
            "negative_classes": negative_classes,
            "best_val_auc": best_auc,
            "history": history,
            "timing": timing,
            "config": {
                "grid_size": config.GRID_SIZE,
                "held_out_steps": config.HELD_OUT_STEPS,
            },
        },
        ckpt_path,
    )
    print(
        f"  saved {ckpt_path} (best val_auc={best_auc:.4f}, "
        f"train {train_seconds:.1f}s / {train_seconds/60:.2f} min)"
    )

    # Density baseline on the same training slice
    t_base0 = time.perf_counter()
    train_np = get_numpy_split(raw, "train", negative_classes, include_held_out_steps=False)
    baseline = fit_density_baseline(
        train_np["boards"], train_np["labels"], train_np["steps"]
    )
    baseline_seconds = time.perf_counter() - t_base0
    import pickle

    with open(out_dir / f"{name}_baseline.pkl", "wb") as f:
        pickle.dump(baseline, f)

    total_seconds = time.perf_counter() - t_all0
    timing["baseline_seconds"] = baseline_seconds
    timing["total_seconds"] = total_seconds
    timing["total_minutes"] = total_seconds / 60.0

    meta = {
        "name": name,
        "negative_classes": negative_classes,
        "best_val_auc": best_auc,
        "history": history,
        "timing": timing,
        "baseline_path": str(out_dir / f"{name}_baseline.pkl"),
        "checkpoint": str(ckpt_path),
    }
    with open(out_dir / f"{name}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    times_path = out_dir / "training_times.json"
    times = {}
    if times_path.exists():
        with open(times_path) as f:
            times = json.load(f)
    times[name] = timing
    with open(times_path, "w") as f:
        json.dump(times, f, indent=2)
    print(f"  wrote timing to {times_path}")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Conway CNN models A and/or B")
    parser.add_argument(
        "--model",
        choices=["A", "B", "both"],
        default="both",
        help="Which model setup to train",
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")
    raw = load_raw()

    if args.model in ("A", "both"):
        train_one_model(
            name="model_a",
            negative_classes=config.MODEL_A_NEGATIVES,
            seed=config.TRAIN_SEED_A,
            device=device,
            raw=raw,
        )
    if args.model in ("B", "both"):
        train_one_model(
            name="model_b",
            negative_classes=config.MODEL_B_NEGATIVES,
            seed=config.TRAIN_SEED_B,
            device=device,
            raw=raw,
        )


if __name__ == "__main__":
    main()
