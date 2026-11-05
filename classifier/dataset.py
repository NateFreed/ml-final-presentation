"""Generate and load the fixed experiment dataset.

Splits are by *initial soup*, not by individual boards. Every step of a
trajectory stays inside one split so the model cannot memorize a soup that
also appears in the test set.

Held-out steps (see config.HELD_OUT_STEPS) are removed from train and val
only. The test set keeps every step so we can plot a continuous curve and
check that performance still makes sense on steps the CNN never trained on.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from . import config
from .generators import generate_soup_bundle


CLASS_TO_ID = {name: i for i, name in enumerate(config.CLASS_NAMES)}
SPLIT_TO_ID = {"train": 0, "val": 1, "test": 2}
ID_TO_SPLIT = {v: k for k, v in SPLIT_TO_ID.items()}


def _allocate_soup_ids() -> dict[str, np.ndarray]:
    """Contiguous soup id ranges for train / val / test."""
    n_train = config.N_TRAIN_SOUPS
    n_val = config.N_VAL_SOUPS
    n_test = config.N_TEST_SOUPS
    train_ids = np.arange(0, n_train)
    val_ids = np.arange(n_train, n_train + n_val)
    test_ids = np.arange(n_train + n_val, n_train + n_val + n_test)
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def generate_dataset(
    out_path: Path | None = None,
    seed: int = config.DATASET_SEED,
) -> Path:
    """Materialize every (split, soup, class, step) board into one .npz file."""
    out_path = Path(out_path or (config.DATA_DIR / "dataset.npz"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    split_ids = _allocate_soup_ids()
    boards_list = []
    labels_list = []
    class_ids_list = []
    steps_list = []
    split_ids_list = []
    soup_ids_list = []

    # Seeded master RNG only chooses per-soup seeds so generation is reproducible
    # even if we later parallelize.
    master = np.random.default_rng(seed)

    for split_name, soup_ids in split_ids.items():
        split_id = SPLIT_TO_ID[split_name]
        print(f"Generating {split_name}: {len(soup_ids)} soups...")
        for soup_id in soup_ids:
            soup_seed = int(master.integers(0, 2**31 - 1))
            bundle = generate_soup_bundle(
                soup_id=int(soup_id),
                size=config.GRID_SIZE,
                density=config.INITIAL_DENSITY,
                max_steps=config.MAX_STEPS,
                seed=soup_seed,
            )
            for class_name, traj in bundle.items():
                class_id = CLASS_TO_ID[class_name]
                label = 1 if class_name == config.POSITIVE_CLASS else 0
                for step_idx, board in enumerate(traj):
                    step = step_idx + 1  # boards are after 1..MAX_STEPS steps
                    boards_list.append(board)
                    labels_list.append(label)
                    class_ids_list.append(class_id)
                    steps_list.append(step)
                    split_ids_list.append(split_id)
                    soup_ids_list.append(int(soup_id))

    boards = np.stack(boards_list).astype(np.uint8)
    payload = {
        "boards": boards,
        "labels": np.asarray(labels_list, dtype=np.uint8),
        "class_ids": np.asarray(class_ids_list, dtype=np.uint8),
        "steps": np.asarray(steps_list, dtype=np.uint8),
        "split_ids": np.asarray(split_ids_list, dtype=np.uint8),
        "soup_ids": np.asarray(soup_ids_list, dtype=np.int32),
        "class_names": np.asarray(config.CLASS_NAMES),
        "held_out_steps": np.asarray(config.HELD_OUT_STEPS, dtype=np.uint8),
        "grid_size": np.asarray(config.GRID_SIZE),
        "initial_density": np.asarray(config.INITIAL_DENSITY),
        "dataset_seed": np.asarray(seed),
    }
    np.savez_compressed(out_path, **payload)
    print(f"Wrote {boards.shape[0]} boards to {out_path}")
    return out_path


def load_raw(path: Path | None = None) -> dict:
    """Load the dataset dict from disk."""
    path = Path(path or (config.DATA_DIR / "dataset.npz"))
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run: python -m classifier.dataset"
        )
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def _mask_for(
    raw: dict,
    split: str,
    negative_classes: list[str] | None,
    include_held_out_steps: bool,
) -> np.ndarray:
    """Boolean mask selecting boards for a model / split."""
    split_id = SPLIT_TO_ID[split]
    mask = raw["split_ids"] == split_id

    if negative_classes is None:
        allowed = set(range(len(config.CLASS_NAMES)))
    else:
        allowed = {CLASS_TO_ID[config.POSITIVE_CLASS]} | {
            CLASS_TO_ID[name] for name in negative_classes
        }
    mask &= np.isin(raw["class_ids"], list(allowed))

    if not include_held_out_steps and split in ("train", "val"):
        mask &= ~np.isin(raw["steps"], config.HELD_OUT_STEPS)
    return mask


class BoardDataset(Dataset):
    """PyTorch dataset over a filtered slice of the saved boards."""

    def __init__(
        self,
        raw: dict,
        split: str,
        negative_classes: list[str] | None,
        include_held_out_steps: bool = False,
    ) -> None:
        self.mask = _mask_for(raw, split, negative_classes, include_held_out_steps)
        self.boards = raw["boards"][self.mask]
        self.labels = raw["labels"][self.mask].astype(np.float32)
        self.class_ids = raw["class_ids"][self.mask]
        self.steps = raw["steps"][self.mask]
        self.soup_ids = raw["soup_ids"][self.mask]
        self.indices = np.flatnonzero(self.mask)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        board = torch.from_numpy(self.boards[idx]).float().unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        step = torch.tensor(int(self.steps[idx]), dtype=torch.int64)
        class_id = torch.tensor(int(self.class_ids[idx]), dtype=torch.int64)
        return board, label, step, class_id


def get_numpy_split(
    raw: dict,
    split: str,
    negative_classes: list[str] | None,
    include_held_out_steps: bool = False,
) -> dict[str, np.ndarray]:
    """Same filtering as BoardDataset, returned as numpy arrays."""
    mask = _mask_for(raw, split, negative_classes, include_held_out_steps)
    return {
        "boards": raw["boards"][mask],
        "labels": raw["labels"][mask],
        "class_ids": raw["class_ids"][mask],
        "steps": raw["steps"][mask],
        "soup_ids": raw["soup_ids"][mask],
        "indices": np.flatnonzero(mask),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the Conway classifier dataset")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=config.DATASET_SEED)
    args = parser.parse_args()
    generate_dataset(out_path=args.out, seed=args.seed)


if __name__ == "__main__":
    main()
