"""Render a mid-game Conway frame next to a Perlin noise look-alike.

conway_midgame.png     a random soup partway through Conway's rules
perlin_lookalike.png   Perlin noise with the same number of live cells

Both follow the classifier's setup: the soup starts at config.INITIAL_DENSITY and
stops within config.MAX_STEPS, and the noise comes from classifier/generators.py
with its default settings. Only the board shape differs, to fit the slide boxes.
The look-alike is noise with a cutoff and no hand edits. Many noise fields are
tried, and the one whose clusters of cells most resemble the Conway frame is kept.
"""

from pathlib import Path

import matplotlib.image as mpimg
import numpy as np

from classifier import config
from classifier.generators import perlin_field
from game_of_life import LifeGrid, NEIGHBOR_OFFSETS
from make_stable_images import COLS, ROWS, render

OUT_DIR = Path(__file__).parent / "slide_images" / "perlin_vs_conway"

SOUP_SEED = 1
MID_STEP = 20
NOISE_SEEDS_TO_TRY = 3000
CLUSTER_BINS = [1, 4, 7, 11, 21, 41]    # cluster sizes 1-3, 4-6, 7-10, 11-20, 21-40, 41+


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
    """Share of live cells in clusters of each size range."""
    counts = np.zeros(len(CLUSTER_BINS))
    for size in clusterSizes(frame):
        counts[np.searchsorted(CLUSTER_BINS, size, side="right") - 1] += size
    return counts / counts.sum()


def changedShare(frame):
    """Cells that would change on the next step, per live cell."""
    return (LifeGrid.conwayStep(frame) != frame).sum() / frame.sum()


def perlinFrame(seed, alive):
    """The classifier's Perlin field, cut to the slide's shape and thresholded."""
    field = perlin_field(COLS, np.random.default_rng(seed))[:ROWS]
    frame = np.zeros(ROWS * COLS, dtype=np.uint8)
    frame[np.argsort(field, axis=None)[-alive:]] = 1
    return frame.reshape(ROWS, COLS)


def describe(name, frame):
    following = LifeGrid.conwayStep(frame)
    neighbors = LifeGrid.countNeighbors(frame)
    live = frame == 1
    sizes = clusterSizes(frame)
    print(f"{name}: alive {int(frame.sum())} ({frame.mean():.1%}), clusters {len(sizes)}, largest {max(sizes)}")
    print(f"  next step: {int((following != frame).sum())} cells change, alive becomes {int(following.sum())}")
    print(f"  live cells with 0-1 neighbors: {(neighbors[live] <= 1).mean():.0%}, "
          f"2-3: {np.isin(neighbors[live], (2, 3)).mean():.0%}, 4+: {(neighbors[live] >= 4).mean():.0%}")


if __name__ == "__main__":
    rng = np.random.default_rng(SOUP_SEED)
    conway = (rng.random((ROWS, COLS)) < config.INITIAL_DENSITY).astype(np.uint8)
    for _ in range(MID_STEP):
        conway = LifeGrid.conwayStep(conway)
    assert MID_STEP <= config.MAX_STEPS
    assert (LifeGrid.conwayStep(conway) != conway).any()
    alive = int(conway.sum())
    target = clusterHistogram(conway)
    targetChanged = changedShare(conway)

    def difference(frame):
        return np.abs(clusterHistogram(frame) - target).sum() + abs(changedShare(frame) - targetChanged)

    seed = min(range(NOISE_SEEDS_TO_TRY), key=lambda s: difference(perlinFrame(s, alive)))
    perlin = perlinFrame(seed, alive)
    assert perlin.sum() == alive

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mpimg.imsave(OUT_DIR / "conway_midgame.png", render(conway))
    mpimg.imsave(OUT_DIR / "perlin_lookalike.png", render(perlin))

    print(f"soup seed {SOUP_SEED}, step {MID_STEP}, noise seed {seed} of {NOISE_SEEDS_TO_TRY}")
    describe("Conway", conway)
    describe("Perlin", perlin)
