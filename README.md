# ml-final-presentation

Conway's Game of Life (and other Life-like rules) on a square binary grid whose edges wrap around.

Requires `numpy` and `matplotlib`.

```python
from game_of_life import LifeGrid

grid = LifeGrid(sideLength=64, density=0.3, seed=0)   # random 0/1 grid
nxt = LifeGrid.conwayStep(grid.frame)                 # one Conway step, returns a new frame
lfod = LifeGrid.stepForward(grid.frame, "B2/S0")      # any rule in B/S notation ("Live Free or Die")
LifeGrid.view(nxt)                                    # 1 = white, 0 = black
```

`stepForward` accepts a rule name from `RULES` (`"conway"`, `"live_free_or_die"`, `"highlife"`, ...), a B/S string like `"B36/S23"`, or a `(birth, survive)` pair like `((3,), (2, 3))`.

Run `python game_of_life.py` for a demo.
