# ml-final-presentation

Conway's Game of Life, and other Life-like rules, on a square binary grid whose edges wrap around.

Requires `numpy` and `matplotlib`:

```bash
pip install numpy matplotlib
```

## Quick start

```python
from game_of_life import LifeGrid

grid = LifeGrid(sideLength=64, density=0.3, seed=0)   # random 0/1 grid
nxt = LifeGrid.conwayStep(grid.frame)                 # one Conway step, returns a new frame
lfod = LifeGrid.stepForward(grid.frame, "B2/S0")      # any rule, here "Live Free or Die"
LifeGrid.view(nxt)                                    # 1 = white, 0 = black
```

Run `python game_of_life.py` to see a demo: a random 64×64 grid at the start, after 1 step, and after 50 steps.

## What's in `game_of_life.py`

### `LifeGrid(sideLength, density=0.5, seed=None)`

Makes a `sideLength × sideLength` grid. Each cell independently starts alive (1) with probability `density`, otherwise dead (0). `density` must be between 0 and 1. Pass the same `seed` to get the same starting grid again, which is useful for reproducible experiments. The grid is stored in `grid.frame`.

### `LifeGrid.conwayStep(frame)`

Takes a frame and returns a **new** frame after one step of Conway's Game of Life. The frame you pass in is not modified. The edges wrap around top/bottom and left/right, so the grid is a torus: a cell on the right edge has neighbors on the left edge, and a corner cell has neighbors in all four corners.

### `LifeGrid.stepForward(frame, rule="B3/S23")`

The general version: one step of any Life-like rule. `conwayStep` just calls this with `"B3/S23"`. Rules use the standard birth/survival notation:

- `B` lists the live-neighbor counts that bring a dead cell to life.
- `S` lists the counts that keep a live cell alive.
- Every other cell is dead in the next frame.

So `"B3/S23"` means a dead cell with exactly 3 live neighbors is born, and a live cell with 2 or 3 live neighbors survives.

`rule` can be given in three ways:

| Form | Example |
| --- | --- |
| A name from `RULES` | `"conway"`, `"live_free_or_die"`, `"highlife"`, `"seeds"`, `"day_and_night"` |
| A B/S string | `"B36/S23"`, `"B2/S0"`, `"B2/S"` (empty S: nothing survives) |
| A `(birth, survive)` pair | `((3,), (2, 3))` |

To add a named rule, add an entry to the `RULES` dict at the top of the file.

### `grid.advance(steps=1, rule="B3/S23")`

Convenience method: moves the grid's own `frame` forward `steps` steps (replacing it) and returns the new frame. Unlike `conwayStep` and `stepForward`, this changes the grid object.

### `LifeGrid.view(frame, title=None, ax=None)`

Draws a frame with matplotlib: 1 = white, 0 = black. With no `ax` it opens a window. Pass an `ax` to draw into an existing figure, for example to put several frames side by side as the demo does.

### Helpers

- `LifeGrid.countNeighbors(frame)` returns the number of live neighbors of every cell, with wraparound.
- `parseRule(rule)` turns any of the rule forms above into a `(birth, survive)` pair of sets.

## Design notes

**Frames are numpy arrays, not nested tuples.** A frame is a 2D numpy array of 0s and 1s (`dtype uint8`). Neighbor counting shifts the whole grid in each of the 8 directions with `np.roll`. `np.roll` wraps whatever falls off one edge back onto the other, so wraparound comes free and a step is a few array operations instead of a Python loop over every cell. That is much faster for large grids or many steps. The step methods still accept a tuple of tuples (or a list of lists) as input and convert it.

**Hashing frames.** numpy arrays can't go in a set or be used as dict keys. If you need that, for example to detect when the grid starts repeating, use `frame.tobytes()`, which gives a hashable key for the frame.

## Testing

Run:

```bash
python test_game_of_life.py
```

It checks:

- A blinker flips back and forth every step, and the input frame isn't modified.
- A 2×2 block split across the four corners stays still, which only happens if both edges wrap.
- A glider travels all the way around an 8×8 grid and returns to its start after 32 steps.
- On random grids, `stepForward` matches a slow cell-by-cell reference version for five different rules.
- Rule parsing, tuple input, density/seed behavior, and that `view` draws 1 as white and 0 as black.
