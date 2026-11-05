"""Render a random grid and a settled Conway grid as matching PNGs for a slide.

conway_stable.png         a grid that Conway's rules no longer change
random_same_density.png   random cells, with the same number alive as the stable grid
random_static_50pct.png   random cells at 50% density, like TV static
"""

from pathlib import Path

import matplotlib.image as mpimg
import numpy as np

from game_of_life import LifeGrid

OUT_DIR = Path(__file__).parent / "slide_images" / "random_vs_stable"
ROWS, COLS = 50, 55     # same shape as the slide's image boxes (about 1.1 : 1)
CELL_PX = 38
LINE_PX = 2

START_DENSITY = 0.35
SEEDS_TO_TRY = 300
MAX_STEPS = 5000

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRID_LINE = (128, 128, 128)


def settle(seed):
    """Run Conway's rules from a random start until a frame repeats.

    Returns (frame, stepsToSettle, period), or None if nothing repeats in MAX_STEPS.
    A period of 1 means the frame no longer changes at all.
    """
    rng = np.random.default_rng(seed)
    frame = (rng.random((ROWS, COLS)) < START_DENSITY).astype(np.uint8)
    seen = {}
    for step in range(MAX_STEPS):
        key = frame.tobytes()
        if key in seen:
            return frame, seen[key], step - seen[key]
        seen[key] = step
        frame = LifeGrid.conwayStep(frame)
    return None


def render(frame):
    """Draw a frame with grid lines: 1 = white, 0 = black."""
    height = ROWS * CELL_PX + (ROWS + 1) * LINE_PX
    width = COLS * CELL_PX + (COLS + 1) * LINE_PX
    image = np.full((height, width, 3), GRID_LINE, dtype=np.uint8)
    for r in range(ROWS):
        for c in range(COLS):
            top = LINE_PX + r * (CELL_PX + LINE_PX)
            left = LINE_PX + c * (CELL_PX + LINE_PX)
            image[top:top + CELL_PX, left:left + CELL_PX] = WHITE if frame[r, c] else BLACK
    return image


if __name__ == "__main__":
    results = {seed: settle(seed) for seed in range(SEEDS_TO_TRY)}
    finished = {seed: result for seed, result in results.items() if result is not None}
    still = {seed: result for seed, result in finished.items() if result[2] == 1}
    periods = [result[2] for result in finished.values()]
    print("seeds tried:", SEEDS_TO_TRY, " settled:", len(finished), " fully still:", len(still))
    print("most common periods:", sorted(set(periods), key=periods.count, reverse=True)[:5])

    # Of the grids that stopped changing completely, use the one with the most live cells.
    seed = max(still, key=lambda s: int(still[s][0].sum()))
    stable, stepsToSettle, _ = still[seed]
    alive = int(stable.sum())

    rng = np.random.default_rng(seed + 1_000_000)
    randomStatic = np.zeros(ROWS * COLS, dtype=np.uint8)
    randomStatic[rng.choice(ROWS * COLS, alive, replace=False)] = 1
    randomStatic = randomStatic.reshape(ROWS, COLS)
    randomStatic50 = (rng.random((ROWS, COLS)) < 0.5).astype(np.uint8)

    assert (LifeGrid.conwayStep(stable) == stable).all()
    assert randomStatic.sum() == alive

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mpimg.imsave(OUT_DIR / "conway_stable.png", render(stable))
    mpimg.imsave(OUT_DIR / "random_same_density.png", render(randomStatic))
    mpimg.imsave(OUT_DIR / "random_static_50pct.png", render(randomStatic50))

    print(f"seed: {seed}  steps to settle: {stepsToSettle}  alive: {alive} of {ROWS * COLS}")
    print(f"50% version alive: {int(randomStatic50.sum())}")
    print("\n".join("".join("#" if v else "." for v in row) for row in stable))
