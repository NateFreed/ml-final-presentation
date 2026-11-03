"""Binary 2D grid for Conway's Game of Life and other Life-like rules.

A frame is a square numpy array of 0s and 1s (dtype uint8). Every step
method takes a frame and returns a new one without modifying its input.
The grid wraps around top/bottom and left/right, so it is a torus.

Rules use the standard B/S ("birth/survival") notation:
    "B3/S23"  ->  a dead cell with 3 live neighbors is born,
                  a live cell with 2 or 3 live neighbors survives,
                  every other cell is dead in the next frame.
"""

import numpy as np
import matplotlib.pyplot as plt

# Some well-known Life-like rules, by name.
RULES = {
    "conway": "B3/S23",
    "live_free_or_die": "B2/S0",
    "highlife": "B36/S23",
    "seeds": "B2/S",
    "day_and_night": "B3678/S34678",
}

# The 8 (row, col) shifts that reach a cell's Moore neighbors.
NEIGHBOR_OFFSETS = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                    if (dr, dc) != (0, 0)]


def parseRule(rule):
    """Turn a rule into a (birth, survive) pair of neighbor-count sets.

    `rule` can be a name from RULES ("conway"), a B/S string ("B3/S23"),
    or a (birth, survive) pair of iterables like ((3,), (2, 3)).
    """
    if isinstance(rule, str):
        rule = RULES.get(rule.lower(), rule)
        birth, survive = set(), set()
        for part in rule.upper().split("/"):
            counts = {int(ch) for ch in part[1:]}
            if part.startswith("B"):
                birth = counts
            elif part.startswith("S"):
                survive = counts
            else:
                raise ValueError(f"Can't parse rule {rule!r}; expected something like 'B3/S23'")
        return birth, survive
    birth, survive = rule
    return set(birth), set(survive)


class LifeGrid:
    """A square binary grid whose cells start randomly alive (1) or dead (0)."""

    def __init__(self, sideLength, density=0.5, seed=None):
        """
        sideLength: number of rows (and columns) in the grid.
        density:    probability that each cell starts alive, between 0 and 1.
        seed:       optional random seed, so a starting grid can be reproduced.
        """
        if not 0 <= density <= 1:
            raise ValueError("density must be between 0 and 1")
        self.sideLength = sideLength
        self.density = density
        rng = np.random.default_rng(seed)
        self.frame = (rng.random((sideLength, sideLength)) < density).astype(np.uint8)

    @staticmethod
    def countNeighbors(frame):
        """Number of live neighbors of every cell, with wraparound edges."""
        frame = np.asarray(frame, dtype=np.uint8)
        # np.roll shifts the whole grid and wraps the edge that falls off
        # back onto the other side, which gives the torus for free.
        return sum(np.roll(frame, shift, axis=(0, 1)) for shift in NEIGHBOR_OFFSETS)

    @staticmethod
    def stepForward(frame, rule="B3/S23"):
        """Return the frame after one step of any Life-like rule (see parseRule)."""
        frame = np.asarray(frame, dtype=np.uint8)
        birth, survive = parseRule(rule)
        neighbors = LifeGrid.countNeighbors(frame)
        born = (frame == 0) & np.isin(neighbors, list(birth))
        survives = (frame == 1) & np.isin(neighbors, list(survive))
        return (born | survives).astype(np.uint8)

    @staticmethod
    def conwayStep(frame):
        """Return the frame after one step of Conway's Game of Life (B3/S23)."""
        return LifeGrid.stepForward(frame, "B3/S23")

    def advance(self, steps=1, rule="B3/S23"):
        """Replace this grid's frame with the frame `steps` steps later."""
        for _ in range(steps):
            self.frame = self.stepForward(self.frame, rule)
        return self.frame

    @staticmethod
    def view(frame, title=None, ax=None):
        """Draw a frame on screen: 1 = white, 0 = black."""
        show = ax is None
        if ax is None:
            _, ax = plt.subplots()
        ax.imshow(frame, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        if title:
            ax.set_title(title)
        if show:
            plt.show()
        return ax


if __name__ == "__main__":
    grid = LifeGrid(sideLength=64, density=0.3, seed=0)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    LifeGrid.view(grid.frame, "Start", ax=axes[0])
    LifeGrid.view(grid.advance(1), "After 1 step", ax=axes[1])
    LifeGrid.view(grid.advance(49), "After 50 steps", ax=axes[2])
    plt.tight_layout()
    plt.show()
