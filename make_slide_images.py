"""Render one Game of Life step as three matching PNGs for a slide.

1_current.png  the starting frame (1 = white, 0 = black)
2_changes.png  the same frame with every cell that is about to flip highlighted
3_next.png     the frame after one Conway step
"""

from pathlib import Path

import matplotlib.image as mpimg
import numpy as np

from game_of_life import LifeGrid

OUT_DIR = Path(__file__).parent / "slide_images"
SIDE = 12
CELL_PX = 96
LINE_PX = 4

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRID_LINE = (128, 128, 128)
DYING = (213, 94, 0)    # alive now, dead next step
BORN = (86, 180, 233)   # dead now, alive next step

# Each pattern sits away from the others and the edges, so none of them interact.
LIVE_CELLS = [
    (1, 2), (2, 3), (3, 1), (3, 2), (3, 3),              # glider
    (2, 7), (2, 8), (2, 9),                              # blinker
    (8, 1), (8, 2), (9, 1), (9, 2),                      # block
    (7, 6), (7, 7), (7, 8), (8, 5), (8, 6), (8, 7),      # toad
]


def render(colors):
    """Turn a SIDE x SIDE grid of RGB colors into an image with grid lines."""
    size = SIDE * CELL_PX + (SIDE + 1) * LINE_PX
    image = np.full((size, size, 3), GRID_LINE, dtype=np.uint8)
    for r in range(SIDE):
        for c in range(SIDE):
            top = LINE_PX + r * (CELL_PX + LINE_PX)
            left = LINE_PX + c * (CELL_PX + LINE_PX)
            image[top:top + CELL_PX, left:left + CELL_PX] = colors[r, c]
    return image


def frameColors(frame):
    colors = np.zeros((SIDE, SIDE, 3), dtype=np.uint8)
    colors[frame == 1] = WHITE
    colors[frame == 0] = BLACK
    return colors


if __name__ == "__main__":
    current = np.zeros((SIDE, SIDE), dtype=np.uint8)
    for r, c in LIVE_CELLS:
        current[r, c] = 1
    following = LifeGrid.conwayStep(current)

    dying = (current == 1) & (following == 0)
    born = (current == 0) & (following == 1)
    highlighted = frameColors(current)
    highlighted[dying] = DYING
    highlighted[born] = BORN

    OUT_DIR.mkdir(exist_ok=True)
    mpimg.imsave(OUT_DIR / "1_current.png", render(frameColors(current)))
    mpimg.imsave(OUT_DIR / "2_changes.png", render(highlighted))
    mpimg.imsave(OUT_DIR / "3_next.png", render(frameColors(following)))

    # Applying exactly the highlighted flips to the first frame must give the third.
    flipped = current.copy()
    flipped[dying] = 0
    flipped[born] = 1
    assert (flipped == following).all()

    print("alive now:", int(current.sum()), " alive next:", int(following.sum()))
    print("dying:", int(dying.sum()), " born:", int(born.sum()))
    print("current:")
    print("\n".join("".join("#" if v else "." for v in row) for row in current))
    print("next:")
    print("\n".join("".join("#" if v else "." for v in row) for row in following))
