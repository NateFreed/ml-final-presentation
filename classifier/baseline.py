"""Density / supergrid logistic baseline.

Part 2 of the outline measured how density evolves under Conway. That curve
alone does not classify a board. Here we turn those ideas into features of
*one* board and fit a logistic regression per step:

- global live-cell fraction
- mean density inside non-overlapping 4x4 and 8x8 blocks
- fraction of those blocks that are completely empty

If the CNN and this baseline look the same on Conway-vs-random, the CNN is
probably just rediscovering density. If the CNN pulls ahead on HighLife or
density-matched noise, local structure is doing real work.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _block_stats(board: np.ndarray, block: int) -> tuple[float, float]:
    """Mean block density and fraction of empty blocks for a tiled supergrid."""
    h, w = board.shape
    assert h % block == 0 and w % block == 0
    reshaped = board.reshape(h // block, block, w // block, block)
    block_means = reshaped.mean(axis=(1, 3))
    return float(block_means.mean()), float((block_means == 0).mean())


def extract_features(boards: np.ndarray) -> np.ndarray:
    """Feature matrix of shape (N, 5) for a batch of (N, H, W) boards."""
    boards = np.asarray(boards, dtype=np.float64)
    if boards.ndim == 2:
        boards = boards[None, ...]
    feats = []
    for board in boards:
        global_density = float(board.mean())
        mean4, empty4 = _block_stats(board, 4)
        mean8, empty8 = _block_stats(board, 8)
        feats.append([global_density, mean4, empty4, mean8, empty8])
    return np.asarray(feats, dtype=np.float64)


FEATURE_NAMES = [
    "global_density",
    "mean_density_4x4",
    "frac_empty_4x4",
    "mean_density_8x8",
    "frac_empty_8x8",
]


@dataclass
class DensityBaseline:
    """One logistic model per step index present in training."""

    models: dict[int, Pipeline]

    def predict_proba(self, boards: np.ndarray, steps: np.ndarray) -> np.ndarray:
        """P(conway) for each board, using the model fit for that step."""
        boards = np.asarray(boards)
        steps = np.asarray(steps)
        probs = np.full(len(steps), 0.5, dtype=np.float64)
        for step in np.unique(steps):
            mask = steps == step
            model = self.models.get(int(step))
            if model is None:
                # Unseen step: fall back to nearest trained step.
                nearest = min(self.models, key=lambda s: abs(s - int(step)))
                model = self.models[nearest]
            probs[mask] = model.predict_proba(extract_features(boards[mask]))[:, 1]
        return probs


def fit_density_baseline(
    boards: np.ndarray,
    labels: np.ndarray,
    steps: np.ndarray,
) -> DensityBaseline:
    """Fit a separate logistic regression on each training step."""
    models: dict[int, Pipeline] = {}
    for step in sorted(set(int(s) for s in steps)):
        mask = steps == step
        x = extract_features(boards[mask])
        y = labels[mask]
        if len(np.unique(y)) < 2:
            continue
        pipe = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
        pipe.fit(x, y)
        models[step] = pipe
    if not models:
        raise RuntimeError("Density baseline could not fit any step")
    return DensityBaseline(models=models)
