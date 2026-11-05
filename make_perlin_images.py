"""Render a second settled Conway grid next to a Perlin noise look-alike.

conway_stable_2.png    a fully still Conway grid, different from conway_stable.png
perlin_lookalike.png   thresholded Perlin noise with the same number of live cells

The look-alike is pure noise with a cutoff: no cell is edited by hand. Cells are
live when their noise value falls in a high band, so a tall peak in the noise
becomes a hollow ring and a lower peak becomes a solid blob. Many noise fields
are tried, and the one whose clusters of cells most resemble the Conway grid is
kept.
"""

from pathlib import Path

import matplotlib.image as mpimg
import numpy as np

from game_of_life import LifeGrid, NEIGHBOR_OFFSETS
from make_stable_images import COLS, ROWS, render, settle

OUT_DIR = Path(__file__).parent / "slide_images" / "perlin_vs_stable"

CONWAY_SEED = 270       # fully still after 392 steps; conway_stable.png uses seed 269

LATTICE_COLS = range(7, 19)     # how many noise features fit across the grid
LOW_FREQUENCY_WEIGHTS = (0.0, 0.5, 1.0)     # a coarse layer that makes the spacing uneven
HOLE_CELLS = (0, 4, 8, 12, 16, 24, 32)      # how many of the highest noise values stay dead
SEEDS_PER_SETTING = 60
LARGEST_CLUSTER_BIN = 10


def perlinNoise(latticeRows, latticeCols, rng):
    """2D Perlin noise on the grid, wrapping at the edges like the grid does."""
    angles = rng.random((latticeRows, latticeCols)) * 2 * np.pi
    gradRow, gradCol = np.sin(angles), np.cos(angles)

    rowPos, colPos = np.meshgrid(np.arange(ROWS) * latticeRows / ROWS,
                                 np.arange(COLS) * latticeCols / COLS, indexing="ij")
    row0, col0 = np.floor(rowPos).astype(int), np.floor(colPos).astype(int)
    fracRow, fracCol = rowPos - row0, colPos - col0
    row1, col1 = (row0 + 1) % latticeRows, (col0 + 1) % latticeCols

    def corner(r, c, dRow, dCol):
        return gradRow[r, c] * dRow + gradCol[r, c] * dCol

    fade = lambda t: t * t * t * (t * (t * 6 - 15) + 10)
    blendCol, blendRow = fade(fracCol), fade(fracRow)
    top = corner(row0, col0, fracRow, fracCol) * (1 - blendCol) + corner(row0, col1, fracRow, fracCol - 1) * blendCol
    bottom = corner(row1, col0, fracRow - 1, fracCol) * (1 - blendCol) + corner(row1, col1, fracRow - 1, fracCol - 1) * blendCol
    return top * (1 - blendRow) + bottom * blendRow


def noiseFrame(latticeCols, lowWeight, holeCells, seed, alive):
    """Keep a band of Perlin noise values so that exactly `alive` cells are live."""
    rng = np.random.default_rng(seed)
    latticeRows = max(2, round(latticeCols * ROWS / COLS))
    noise = perlinNoise(latticeRows, latticeCols, rng)
    if lowWeight:
        noise = noise + lowWeight * perlinNoise(max(2, latticeRows // 3), max(2, latticeCols // 3), rng)
    order = np.argsort(noise, axis=None)
    start = order.size - alive - holeCells
    frame = np.zeros(ROWS * COLS, dtype=np.uint8)
    frame[order[start:start + alive]] = 1
    return frame.reshape(ROWS, COLS)


def clusterSizes(frame):
    """Sizes of the groups of touching live cells, with wraparound edges."""
    unvisited = {tuple(cell) for cell in np.argwhere(frame == 1)}
    sizes = []
    while unvisited:
        stack = [unvisited.pop()]
        size = 0
        while stack:
            r, c = stack.pop()
            size += 1
            for dr, dc in NEIGHBOR_OFFSETS:
                neighbor = ((r + dr) % ROWS, (c + dc) % COLS)
                if neighbor in unvisited:
                    unvisited.remove(neighbor)
                    stack.append(neighbor)
        sizes.append(size)
    return sizes


def clusterHistogram(frame):
    """Share of live cells in clusters of each size, with big clusters pooled together."""
    counts = np.zeros(LARGEST_CLUSTER_BIN + 1)
    for size in clusterSizes(frame):
        counts[min(size, LARGEST_CLUSTER_BIN)] += size
    return counts / counts.sum()


def enclosedHoles(frame):
    """Number of dead regions cut off from the background, like the middle of a ring."""
    unvisited = {tuple(cell) for cell in np.argwhere(frame == 0)}
    regions = 0
    while unvisited:
        stack = [unvisited.pop()]
        regions += 1
        while stack:
            r, c = stack.pop()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                neighbor = ((r + dr) % ROWS, (c + dc) % COLS)
                if neighbor in unvisited:
                    unvisited.remove(neighbor)
                    stack.append(neighbor)
    return regions - 1


def difference(frame, targetHistogram, targetHoles, targetClusters):
    """How unlike the Conway grid a frame looks: lower is closer."""
    changed = (LifeGrid.conwayStep(frame) != frame).sum() / frame.sum()
    holes = abs(enclosedHoles(frame) - targetHoles) / targetClusters
    return np.abs(clusterHistogram(frame) - targetHistogram).sum() + changed + holes


if __name__ == "__main__":
    stable, stepsToSettle, period = settle(CONWAY_SEED)
    assert period == 1
    assert (LifeGrid.conwayStep(stable) == stable).all()
    alive = int(stable.sum())
    targetHistogram = clusterHistogram(stable)
    targetHoles = enclosedHoles(stable)
    targetClusters = len(clusterSizes(stable))

    best = None
    tried = 0
    for latticeCols in LATTICE_COLS:
        for lowWeight in LOW_FREQUENCY_WEIGHTS:
            for holeCells in HOLE_CELLS:
                for seed in range(SEEDS_PER_SETTING):
                    tried += 1
                    frame = noiseFrame(latticeCols, lowWeight, holeCells, seed, alive)
                    score = difference(frame, targetHistogram, targetHoles, targetClusters)
                    if best is None or score < best[0]:
                        best = (score, latticeCols, lowWeight, holeCells, seed, frame)
    score, latticeCols, lowWeight, holeCells, seed, perlin = best
    assert perlin.sum() == alive

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mpimg.imsave(OUT_DIR / "conway_stable_2.png", render(stable))
    mpimg.imsave(OUT_DIR / "perlin_lookalike.png", render(perlin))

    print(f"Conway: seed {CONWAY_SEED}, settled after {stepsToSettle} steps, alive {alive} of {ROWS * COLS}")
    print(f"Conway cluster sizes: {sorted(clusterSizes(stable))}")
    print(f"Perlin: tried {tried}, best lattice {latticeCols} across, low weight {lowWeight}, "
          f"hole cells {holeCells}, seed {seed}, score {score:.3f}")
    print(f"Perlin cluster sizes: {sorted(clusterSizes(perlin))}")
    print(f"enclosed holes: Conway {targetHoles}, Perlin {enclosedHoles(perlin)}")
    print("\n".join("".join("#" if v else "." for v in row) for row in stable))
    print()
    print("\n".join("".join("#" if v else "." for v in row) for row in perlin))
