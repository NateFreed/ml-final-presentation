# Conway board classifier

Final project for Machine Learning: can a CNN tell **Conway’s Game of Life** boards apart from lookalike boards, and how does that ability change with the number of steps from random noise?

This repo has two layers:

1. **`game_of_life.py`** — your teammates’ toroidal Life engine and `LifeGrid.view` board viewer (any B/S rule).
2. **`classifier/`** — dataset generation, CNN + density baseline, step-count evaluation, figures, and FP/FN inspection that reuses `LifeGrid.view`.

## The question, carefully

We classify a **single** 32×32 binary board:

> Was this board produced by running Conway (B3/S23) for \(k \ge 1\) steps, starting from fair random noise?

We do **not** feed frame pairs. Given two consecutive frames, checking the rule is nearly trivial because the next state is a fixed function of each cell’s 3×3 neighborhood. The interesting task is a snapshot from the *distribution* of random soups after \(k\) steps (ash, blinkers, blocks, empty regions). The model is not “verifying the ruleset” in a logical sense.

### Why a density baseline exists

After one Conway step from density 0.5, expected live density falls to about **0.27**, then keeps falling. A threshold on live-cell count already separates Conway-at-step-\(k\) from fresh 50% noise. So a CNN that only wins on “Conway vs random” may be counting cells.

Controls:

| Negative class | What it tests |
| --- | --- |
| `random` | Headline curve (Conway after \(k\) vs fresh noise). Density helps a lot. |
| `density_matched` | iid noise with the same live fraction as the Conway board. Density alone cannot work. |
| `perlin` | Smooth spatial noise, thresholded to that density. Catches “clump detectors.” |
| `live_free_or_die` | B2/S0 from the project outline. Diverges quickly from Conway. |
| `highlife` | B36/S23. Much closer to Conway early on — the honest “similar ruleset” test. |
| `day_and_night` | B3678/S34678. **Test-only** generalization probe — never seen in Model A/B training. |

### Two models, same architecture

- **Model A** — trained only on Conway vs `random`. Matches the requested step-count graph. Then evaluated on harder negatives without retraining.
- **Model B** — trained with negatives drawn evenly from all five negative types. This is the model that can claim to distinguish Conway from similar boards.

One network sees all training steps; we **slice metrics by \(k\)** at test time. Steps `{5,10,15,20,25,30}` are held out of train/val (test keeps them) so the curve can be checked off the training support.

## Setup

```bash
cd ml-final-presentation
pip install -r requirements.txt
```

Optional: run the Life engine tests first:

```bash
python test_game_of_life.py
```

## Run the full experiment

From `ml-final-presentation/`:

```bash
python -m classifier.run_experiment
```

**Expected runtime (CPU):** about **45–60 minutes** the first time (dataset + both models + eval + figures + mistake panels). Re-runs with an existing `data/dataset.npz` and `--skip-train` are about **2 minutes**. Do not launch two `run_experiment` processes at once — they thrash the CPU and can look hung.

That will:

1. Generate `data/dataset.npz` (skipped if it already exists; pass `--force-data` to rebuild)
2. Train `outputs/model_a.pt` and `outputs/model_b.pt` (+ density baselines)
3. Write `outputs/model_*_metrics.json`
4. Save step-count figures and example boards under `outputs/`
5. Render false positive / false negative panels with `LifeGrid.view` under `outputs/mistakes/`

Pieces can also be run alone:

```bash
python -m classifier.dataset
python -m classifier.train --model both
python -m classifier.evaluate --model both
python -m classifier.plots --model both
python -m classifier.inspect_mistakes --model both --steps 1 5 15
```

## Looking at mistakes

```bash
python -m classifier.inspect_mistakes --model both \
  --negatives random highlife density_matched \
  --steps 1 5 15
```

Each panel is drawn with `LifeGrid.view` (1 = white, 0 = black), same as the Life demo. Files land in `outputs/mistakes/`:

- `model_a_vs_highlife_step5_fp.png` — negatives the model called Conway
- `model_a_vs_highlife_step5_fn.png` — Conway boards the model missed

## Presentation notebook

```bash
jupyter notebook notebooks/presentation.ipynb
```

The notebook loads saved metrics and figures; it does not retrain.

## Project layout

```
game_of_life.py          # LifeGrid stepper + view
test_game_of_life.py
classifier/
  config.py              # grid size, steps, soup counts, held-out steps
  generators.py          # rules via LifeGrid + random / density / Perlin
  dataset.py             # soup-level splits → data/dataset.npz
  model.py               # circular-pad CNN + global average pool
  baseline.py            # per-step logistic density/supergrid baseline
  train.py               # Model A and Model B
  evaluate.py            # accuracy / AUC by step × negative class
  plots.py               # step curves + example boards
  inspect_mistakes.py    # FP/FN via LifeGrid.view
  run_experiment.py      # one-shot pipeline
notebooks/presentation.ipynb
data/                    # generated (gitignored)
outputs/                 # checkpoints, metrics, figures (gitignored)
```

## Slide images

The images for the slides are in `slide_images/`, and each folder comes from one script at the repo root. `make_code_image.py` also needs `pillow`, `pygments` and macOS's Menlo font.

| Script | Output | What it shows |
| --- | --- | --- |
| `make_slide_images.py` | `slide_images/*.png` | One Conway step: current frame, the cells about to flip, next frame |
| `make_stable_images.py` | `slide_images/random_vs_stable/` | A fully still Conway grid, a random grid with the same number of live cells, and 50% random static |
| `make_rule_images.py` | `slide_images/helpful/` | A cell and its eight neighbors, and the four rule outcomes |
| `make_perlin_images.py` | `slide_images/perlin_vs_conway/` | A Conway frame 20 steps from random noise and a Perlin noise look-alike with the same number of live cells |
| `make_code_image.py` | `slide_images/code/` | `game_of_life.py` lines 68 to 89, drawn like a code editor window |

`slide_images/index.html` and `slide_images/slide_set.html` are the pages that show the images together.

These images are illustrations and are separate from the classifier's dataset:

- The still Conway grid was run until it stopped changing, and was picked from the 7 of 300 random starts that ended fully still.
- The mid-game frame and the Perlin grid follow the classifier's setup (a 50% soup, at most 30 steps, and `perlin_field` from `classifier/generators.py`), but on a 55×50 board instead of 32×32 so they fit the slide boxes. The Perlin grid was picked from 3,000 noise fields as the closest match to that one frame, so it is a flattering example rather than a typical one.

## What not to claim in the talk

- The CNN has “learned Conway’s rules” as a predicate over all boards. It has learned the *distribution* of random-soup snapshots after \(k\) steps.
- Live Free or Die is a strong lookalike. Prefer HighLife (and density-matched / Perlin) when arguing the model sees structure.
- Life was published in **1970** (not 1977).

## Life engine quick start

```python
from game_of_life import LifeGrid

grid = LifeGrid(sideLength=64, density=0.3, seed=0)
nxt = LifeGrid.conwayStep(grid.frame)
LifeGrid.view(nxt)
```

See the original notes below for rule parsing, wraparound, and tests.

<details>
<summary>LifeGrid API (from the original README)</summary>

### `LifeGrid(sideLength, density=0.5, seed=None)`

Square grid; each cell starts alive with probability `density`. Stored in `grid.frame`.

### `LifeGrid.conwayStep(frame)` / `LifeGrid.stepForward(frame, rule=...)`

One step; input unchanged. Edges wrap (torus). Rules: `"conway"`, `"live_free_or_die"`, `"highlife"`, B/S strings like `"B3/S23"`, or `(birth, survive)` pairs.

### `LifeGrid.view(frame, title=None, ax=None)`

Draw 1 = white, 0 = black.

</details>
