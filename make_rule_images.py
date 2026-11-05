"""Render explanation images for the slide on how the rules work.

neighbors.png  a cell (pink) and its eight neighbors (yellow)
rules.png      the four outcomes for a cell, each checked against LifeGrid.conwayStep
"""

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from game_of_life import LifeGrid, NEIGHBOR_OFFSETS

OUT_DIR = Path(__file__).parent / "slide_images" / "helpful"

BLACK = (0, 0, 0)
GRID_LINE = (128, 128, 128)
CELL_COLOR = (204, 121, 167)        # the cell being updated
NEIGHBOR_COLOR = (240, 228, 66)     # its eight neighbors

# (title, subtitle, is the center cell alive, live neighbors as (row, col) in a 3 x 3 block,
#  should the center be alive next step)
CASES = [
    ("Dies: too few", "0 or 1 live neighbors", True, [(0, 0)], False),
    ("Survives", "2 or 3 live neighbors", True, [(0, 1), (2, 2)], True),
    ("Dies: too many", "4 or more live neighbors", True, [(0, 0), (0, 2), (1, 0), (2, 1)], False),
    ("Born", "dead cell, exactly 3 live neighbors", False, [(0, 1), (1, 0), (2, 2)], True),
]


def renderNeighbors(side=5, cellPx=232, linePx=8):
    size = side * cellPx + (side + 1) * linePx
    image = np.full((size, size, 3), GRID_LINE, dtype=np.uint8)
    center = side // 2
    neighbors = {(center + dr, center + dc) for dr, dc in NEIGHBOR_OFFSETS}
    for r in range(side):
        for c in range(side):
            top = linePx + r * (cellPx + linePx)
            left = linePx + c * (cellPx + linePx)
            if (r, c) == (center, center):
                color = CELL_COLOR
            elif (r, c) in neighbors:
                color = NEIGHBOR_COLOR
            else:
                color = BLACK
            image[top:top + cellPx, left:left + cellPx] = color
    return image


def centerAfterStep(alive, liveNeighbors):
    """Run the real step on a 3 x 3 block in an empty grid, and read back the center."""
    frame = np.zeros((9, 9), dtype=np.uint8)
    frame[4, 4] = int(alive)
    for r, c in liveNeighbors:
        frame[3 + r, 3 + c] = 1
    assert LifeGrid.countNeighbors(frame)[4, 4] == len(liveNeighbors)
    return bool(LifeGrid.conwayStep(frame)[4, 4])


def drawTile(ax, x, y, size, alive, ring=False):
    edge = tuple(v / 255 for v in (CELL_COLOR if ring else GRID_LINE))
    ax.add_patch(Rectangle((x, y), size, size, facecolor="white" if alive else "black",
                           edgecolor=edge, linewidth=7 if ring else 1.5, zorder=3 if ring else 2))


def renderRules(path):
    plt.rcParams["font.sans-serif"] = ["Roboto", "Helvetica", "Arial", "DejaVu Sans"]
    fig = plt.figure(figsize=(8, 8), dpi=200, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.set_aspect("equal")
    ax.axis("off")

    tile = 1.75
    for i, (title, subtitle, alive, liveNeighbors, expected) in enumerate(CASES):
        assert centerAfterStep(alive, liveNeighbors) == expected, title
        panelX = 0.5 + (i % 2) * 9.75
        panelY = 10 if i < 2 else 0
        ax.text(panelX, panelY + 9, title, fontsize=21, fontweight="bold", va="center")
        ax.text(panelX, panelY + 8, subtitle, fontsize=13, va="center", color="#333333")

        gridY = panelY + 1.5
        for r in range(3):
            for c in range(3):
                isCenter = (r, c) == (1, 1)
                cellAlive = alive if isCenter else (r, c) in liveNeighbors
                drawTile(ax, panelX + c * tile, gridY + (2 - r) * tile, tile, cellAlive, ring=isCenter)

        arrowY = gridY + 1.5 * tile
        ax.annotate("", xy=(panelX + 3 * tile + 1.75, arrowY), xytext=(panelX + 3 * tile + 0.3, arrowY),
                    arrowprops=dict(arrowstyle="-|>", color="#333333", linewidth=2.5, mutation_scale=22))
        drawTile(ax, panelX + 3 * tile + 2.0, gridY + tile, tile, expected, ring=True)

    fig.savefig(path, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mpimg.imsave(OUT_DIR / "neighbors.png", renderNeighbors())
    renderRules(OUT_DIR / "rules.png")
    print("rule cases checked against LifeGrid.conwayStep:", len(CASES))
