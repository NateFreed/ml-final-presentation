"""Small CNN for single binary Life boards.

Design choices that matter for the talk:
- One input channel: a single snapshot. Pairing frames would make the task
  nearly trivial because the next state is a local function of the 3x3.
- Circular padding: the board is a torus, same as LifeGrid.
- 3x3 kernels: match the Moore neighborhood that generates the patterns.
- Global average pooling: a blinker in the corner counts like one in the center.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConwayCNN(nn.Module):
    """Three circular-padded conv layers -> GAP -> linear logit."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, padding_mode="circular"),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, padding_mode="circular"),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, padding_mode="circular"),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, H, W) float in {0, 1}
        h = self.features(x)
        h = h.mean(dim=(2, 3))
        return self.head(h).squeeze(-1)
