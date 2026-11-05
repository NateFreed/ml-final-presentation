"""Fine-tune only the CNN head with HighLife heavily upweighted.

Freezes the convolutional feature extractor from a pretrained Model B
checkpoint, then retrains `model.head` so negatives are dominated by
HighLife while a little mass remains on the other lookalikes (to limit
forgetting). Writes a new checkpoint; does not overwrite the backup.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from . import config
from .dataset import CLASS_TO_ID, BoardDataset, load_raw
from .evaluate import evaluate_model
from .model import ConwayCNN
from .train import collect_logits, roc_auc_from_logits, set_seed


def backup_pretrained(backup_dir: Path | None = None) -> Path:
    """Copy current Model B artifacts into outputs/backups/pretrained_model_b/."""
    backup_dir = Path(backup_dir or (config.OUTPUT_DIR / "backups" / "pretrained_model_b"))
    backup_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "model_b.pt",
        "model_b_meta.json",
        "model_b_baseline.pkl",
        "model_b_metrics.json",
        "training_times.json",
    ):
        src = config.OUTPUT_DIR / name
        if src.exists():
            shutil.copy2(src, backup_dir / name)
    print(f"Backed up Model B artifacts to {backup_dir}")
    return backup_dir


def make_highlife_heavy_loader(
    dataset: BoardDataset,
    batch_size: int,
    highlife_neg_fraction: float = 0.8,
) -> DataLoader:
    """Sampler: 50% Conway, and among negatives `highlife_neg_fraction` HighLife."""
    labels = dataset.labels
    class_ids = dataset.class_ids
    highlife_id = CLASS_TO_ID["highlife"]

    pos = labels == 1
    neg_hl = class_ids == highlife_id
    neg_other = (labels == 0) & ~neg_hl

    weights = np.zeros(len(labels), dtype=np.float64)
    weights[pos] = 0.5 / max(float(pos.sum()), 1.0)
    weights[neg_hl] = (0.5 * highlife_neg_fraction) / max(float(neg_hl.sum()), 1.0)
    weights[neg_other] = (0.5 * (1.0 - highlife_neg_fraction)) / max(
        float(neg_other.sum()), 1.0
    )

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(weights),
        num_samples=len(weights),
        replacement=True,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=config.NUM_WORKERS,
    )


def finetune_head(
    source_ckpt: Path,
    out_name: str = "model_b_highlife_ft",
    highlife_neg_fraction: float = 0.8,
    epochs: int = 15,
    lr: float = 1e-3,
    patience: int = 5,
    seed: int = 42,
    device: torch.device | None = None,
) -> dict:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    raw = load_raw()

    train_ds = BoardDataset(
        raw, "train", config.MODEL_B_NEGATIVES, include_held_out_steps=False
    )
    val_ds = BoardDataset(
        raw, "val", config.MODEL_B_NEGATIVES, include_held_out_steps=False
    )
    # Val for early stop: Conway vs HighLife only, so we optimize the hard case.
    val_hl_ds = BoardDataset(
        raw, "val", ["highlife"], include_held_out_steps=False
    )
    train_loader = make_highlife_heavy_loader(
        train_ds, config.BATCH_SIZE, highlife_neg_fraction=highlife_neg_fraction
    )
    val_loader = DataLoader(
        val_hl_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    payload = torch.load(source_ckpt, map_location=device, weights_only=False)
    model = ConwayCNN().to(device)
    model.load_state_dict(payload["model_state"])

    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True

    opt = torch.optim.Adam(model.head.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    print(
        f"\n=== Fine-tuning head from {source_ckpt.name} -> {out_name} ===\n"
        f"  freeze=features  highlife_neg_fraction={highlife_neg_fraction}  "
        f"device={device}\n"
        f"  train={len(train_ds)}  val_highlife={len(val_hl_ds)}"
    )

    best_auc = -1.0
    best_state = None
    patience_left = patience
    history = []
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch0 = time.perf_counter()
        model.train()
        model.features.eval()  # frozen backbone stays in eval (no BN anyway)
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
                "val_highlife_auc": val_auc,
                "epoch_seconds": epoch_sec,
            }
        )
        print(
            f"  epoch {epoch:02d}  train_loss={train_loss:.4f}  "
            f"val_highlife_auc={val_auc:.4f}  ({epoch_sec:.1f}s)"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  early stop at epoch {epoch} (best val_hl_auc={best_auc:.4f})")
                break

    train_seconds = time.perf_counter() - t0
    assert best_state is not None
    model.load_state_dict(best_state)

    out_dir = config.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{out_name}.pt"
    timing = {
        "train_seconds": train_seconds,
        "train_minutes": train_seconds / 60.0,
        "n_epochs": len(history),
        "device": str(device),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "highlife_neg_fraction": highlife_neg_fraction,
        "lr": lr,
        "source_checkpoint": str(source_ckpt),
    }
    torch.save(
        {
            "model_state": best_state,
            "name": out_name,
            "negative_classes": config.MODEL_B_NEGATIVES,
            "best_val_auc": best_auc,
            "history": history,
            "timing": timing,
            "finetune": {
                "frozen": "features",
                "trainable": "head",
                "highlife_neg_fraction": highlife_neg_fraction,
                "early_stop_metric": "val_conway_vs_highlife_auc",
            },
            "config": payload.get("config", {}),
        },
        ckpt_path,
    )

    # Reuse the pretrained density baseline (features unchanged for that model).
    src_baseline = config.OUTPUT_DIR / "model_b_baseline.pkl"
    if not src_baseline.exists():
        src_baseline = (
            config.OUTPUT_DIR / "backups" / "pretrained_model_b" / "model_b_baseline.pkl"
        )
    dst_baseline = out_dir / f"{out_name}_baseline.pkl"
    shutil.copy2(src_baseline, dst_baseline)

    meta = {
        "name": out_name,
        "source_checkpoint": str(source_ckpt),
        "best_val_highlife_auc": best_auc,
        "history": history,
        "timing": timing,
        "checkpoint": str(ckpt_path),
        "baseline_path": str(dst_baseline),
    }
    with open(out_dir / f"{out_name}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(
        f"  saved {ckpt_path} (best val HighLife AUC={best_auc:.4f}, "
        f"{train_seconds:.1f}s)"
    )
    return meta


def compare_models(
    before_name: str,
    after_name: str,
    device: torch.device,
    raw: dict,
) -> dict:
    """Evaluate both checkpoints and write a side-by-side comparison."""
    before = evaluate_model(before_name, device, raw)
    after = evaluate_model(after_name, device, raw)

    steps = [1, 5, 15, 30]
    comparison = {
        "before": before_name,
        "after": after_name,
        "before_threshold": before.get("cnn_threshold"),
        "after_threshold": after.get("cnn_threshold"),
        "by_negative": {},
    }
    print("\n=== Before vs after (CNN accuracy / AUC) ===")
    for neg in config.NEGATIVE_CLASSES:
        comparison["by_negative"][neg] = {}
        print(f"\nvs {neg}")
        print(f"{'k':>4}  {'before_acc':>10} {'after_acc':>10} {'d_acc':>8}  "
              f"{'before_auc':>10} {'after_auc':>10} {'d_auc':>8}")
        for step in steps:
            b = before["by_negative"][neg][str(step)]["cnn"]
            a = after["by_negative"][neg][str(step)]["cnn"]
            row = {
                "before_acc": b["accuracy"],
                "after_acc": a["accuracy"],
                "delta_acc": a["accuracy"] - b["accuracy"],
                "before_auc": b["auc"],
                "after_auc": a["auc"],
                "delta_auc": a["auc"] - b["auc"],
            }
            comparison["by_negative"][neg][str(step)] = row
            print(
                f"{step:4d}  {row['before_acc']:10.3f} {row['after_acc']:10.3f} "
                f"{row['delta_acc']:+8.3f}  {row['before_auc']:10.3f} "
                f"{row['after_auc']:10.3f} {row['delta_auc']:+8.3f}"
            )

    out_path = config.OUTPUT_DIR / f"{after_name}_vs_{before_name}_comparison.json"
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nWrote {out_path}")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backup Model B and fine-tune last layer for HighLife"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Checkpoint to fine-tune (default: backups copy, else outputs/model_b.pt)",
    )
    parser.add_argument("--out-name", default="model_b_highlife_ft")
    parser.add_argument("--highlife-neg-fraction", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-backup", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if not args.skip_backup:
        backup_pretrained()

    source = args.source
    if source is None:
        backed = config.OUTPUT_DIR / "backups" / "pretrained_model_b" / "model_b.pt"
        source = backed if backed.exists() else (config.OUTPUT_DIR / "model_b.pt")

    finetune_head(
        source_ckpt=source,
        out_name=args.out_name,
        highlife_neg_fraction=args.highlife_neg_fraction,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        seed=args.seed,
        device=device,
    )

    if not args.skip_eval:
        # Restore pretrained checkpoint path name for evaluate_model("model_b")
        # by ensuring outputs/model_b.pt matches the backup (fine-tune does not overwrite it).
        raw = load_raw()
        compare_models("model_b", args.out_name, device, raw)


if __name__ == "__main__":
    main()
