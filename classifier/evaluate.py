"""Evaluate Model A / Model B and the density baseline by step and negative class.

For each (model, negative_class, step) we score Conway boards against that
negative on the test split, reporting accuracy and ROC-AUC. This is the
table that plots.py turns into the presentation curves.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

from . import config
from .dataset import CLASS_TO_ID, load_raw
from .model import ConwayCNN


def load_cnn(ckpt_path: Path, device: torch.device) -> ConwayCNN:
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ConwayCNN().to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


def load_baseline(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


@torch.no_grad()
def cnn_probs(model: ConwayCNN, boards: np.ndarray, device: torch.device) -> np.ndarray:
    """Return P(conway) for a batch of boards."""
    model.eval()
    out = []
    batch = 256
    for i in range(0, len(boards), batch):
        chunk = boards[i : i + batch]
        x = torch.from_numpy(chunk).float().unsqueeze(1).to(device)
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()
        out.append(probs)
    return np.concatenate(out) if out else np.array([], dtype=np.float64)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (rough uncertainty band)."""
    if n == 0:
        return 0.5, 0.5
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return float(center - half), float(center + half)


def score_binary(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    preds = (probs >= threshold).astype(np.uint8)
    acc = float(accuracy_score(labels, preds))
    if len(np.unique(labels)) < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(labels, probs))
    n = len(labels)
    k = int((preds == labels).sum())
    lo, hi = wilson_interval(k, n)
    return {
        "accuracy": acc,
        "auc": auc,
        "n": int(n),
        "acc_low": lo,
        "acc_high": hi,
        "threshold": float(threshold),
    }


def best_threshold(labels: np.ndarray, probs: np.ndarray) -> float:
    """Pick a decision threshold on validation via Youden's J (TPR - FPR).

    Model B trains with five negatives, so raw sigmoid scores are not
    guaranteed to be centered on 0.5. Tuning the cut on val (never test)
    makes the accuracy curves reflect ranking quality instead of an
    arbitrary cut.
    """
    from sklearn.metrics import roc_curve

    if len(np.unique(labels)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(labels, probs)
    j = tpr - fpr
    return float(thresholds[int(np.argmax(j))])


def evaluate_model(
    model_name: str,
    device: torch.device,
    raw: dict,
) -> dict:
    out_dir = config.OUTPUT_DIR
    ckpt_path = out_dir / f"{model_name}.pt"
    baseline_path = out_dir / f"{model_name}_baseline.pkl"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    train_negatives = list(ckpt.get("negative_classes", config.MODEL_B_NEGATIVES))
    model = load_cnn(ckpt_path, device)
    baseline = load_baseline(baseline_path)

    conway_id = CLASS_TO_ID["conway"]
    train_neg_ids = [CLASS_TO_ID[n] for n in train_negatives]

    # Tune one decision threshold on the validation slice the model trained for.
    val_mask = (
        (raw["split_ids"] == 1)
        & ~np.isin(raw["steps"], config.HELD_OUT_STEPS)
        & np.isin(raw["class_ids"], [conway_id] + train_neg_ids)
    )
    val_boards = raw["boards"][val_mask]
    val_labels = raw["labels"][val_mask]
    val_steps = raw["steps"][val_mask]
    val_cnn_p = cnn_probs(model, val_boards, device)
    val_base_p = baseline.predict_proba(val_boards, val_steps)
    cnn_threshold = best_threshold(val_labels, val_cnn_p)
    baseline_threshold = best_threshold(val_labels, val_base_p)
    print(
        f"{model_name}: val-tuned thresholds  "
        f"cnn={cnn_threshold:.4f}  baseline={baseline_threshold:.4f}"
    )

    results = {
        "model": model_name,
        "train_negatives": train_negatives,
        "cnn_threshold": cnn_threshold,
        "baseline_threshold": baseline_threshold,
        "by_negative": {},
        "step0_sanity": {},
    }

    # Sanity: pure noise with fake 50/50 labels must stay near chance.
    rng = np.random.default_rng(0)
    step0 = (rng.random((400, config.GRID_SIZE, config.GRID_SIZE)) < 0.5).astype(np.uint8)
    labels0 = np.array([1] * 200 + [0] * 200, dtype=np.uint8)
    probs0 = cnn_probs(model, step0, device)
    results["step0_sanity"] = score_binary(labels0, probs0, threshold=cnn_threshold)

    test_mask = raw["split_ids"] == 2  # test
    for neg_name in config.NEGATIVE_CLASSES:
        neg_id = CLASS_TO_ID[neg_name]
        per_step = {}
        for step in config.STEPS:
            mask = (
                test_mask
                & (raw["steps"] == step)
                & np.isin(raw["class_ids"], [conway_id, neg_id])
            )
            boards = raw["boards"][mask]
            labels = raw["labels"][mask]
            steps = raw["steps"][mask]
            class_ids = raw["class_ids"][mask]
            indices = np.flatnonzero(mask)

            cnn_p = cnn_probs(model, boards, device)
            base_p = baseline.predict_proba(boards, steps)

            per_step[str(step)] = {
                "cnn": score_binary(labels, cnn_p, threshold=cnn_threshold),
                "baseline": score_binary(labels, base_p, threshold=baseline_threshold),
                "mistake_preview": _mistake_preview(
                    labels,
                    cnn_p,
                    class_ids,
                    indices,
                    steps,
                    neg_id,
                    threshold=cnn_threshold,
                ),
            }
        results["by_negative"][neg_name] = per_step
        mid = per_step[str(15)]["cnn"]
        print(
            f"{model_name} vs {neg_name} @ step 15: "
            f"acc={mid['accuracy']:.3f} auc={mid['auc']:.3f}"
        )

    out_path = out_dir / f"{model_name}_metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}")
    print(
        f"  step-0 sanity (pure noise, fake labels): "
        f"acc={results['step0_sanity']['accuracy']:.3f} "
        f"auc={results['step0_sanity']['auc']:.3f}"
    )
    return results


def _mistake_preview(
    labels: np.ndarray,
    probs: np.ndarray,
    class_ids: np.ndarray,
    indices: np.ndarray,
    steps: np.ndarray,
    neg_id: int,
    top_k: int = 5,
    threshold: float = 0.5,
) -> dict:
    """High-confidence false positives / false negatives for later viewing."""
    preds = probs >= threshold
    fp_mask = (labels == 0) & preds
    fn_mask = (labels == 1) & ~preds

    def top_mistakes(mask, prefer_high_prob: bool):
        idxs = np.flatnonzero(mask)
        if len(idxs) == 0:
            return []
        order = np.argsort(probs[idxs])
        if prefer_high_prob:
            order = order[::-1]
        chosen = idxs[order[:top_k]]
        return [
            {
                "dataset_index": int(indices[i]),
                "prob": float(probs[i]),
                "class_id": int(class_ids[i]),
                "step": int(steps[i]),
            }
            for i in chosen
        ]

    return {
        "false_positives": top_mistakes(fp_mask, prefer_high_prob=True),
        "false_negatives": top_mistakes(fn_mask, prefer_high_prob=False),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Conway CNN models")
    parser.add_argument("--model", choices=["A", "B", "both"], default="both")
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
        evaluate_model(name, device, raw)


if __name__ == "__main__":
    main()
