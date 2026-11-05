"""Shared experiment settings.

Keep these knobs in one place so the talk, the plots, and the training
scripts all describe the same experiment.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

# Board geometry and soup settings
GRID_SIZE = 32
INITIAL_DENSITY = 0.5
MAX_STEPS = 30
STEPS = list(range(1, MAX_STEPS + 1))

# Steps withheld from train/val so the curve can be checked off the
# training support. Test still includes every step.
HELD_OUT_STEPS = [5, 10, 15, 20, 25, 30]

# Soups per split. Each soup yields one board per (class, step).
N_TRAIN_SOUPS = 400
N_VAL_SOUPS = 100
N_TEST_SOUPS = 200

# Class names. Index 0 is the positive class for both models.
CLASS_NAMES = [
    "conway",
    "random",
    "density_matched",
    "perlin",
    "live_free_or_die",
    "highlife",
]
POSITIVE_CLASS = "conway"
NEGATIVE_CLASSES = [c for c in CLASS_NAMES if c != POSITIVE_CLASS]

# Model A only sees fresh random noise as the negative during training.
MODEL_A_NEGATIVES = ["random"]
# Model B sees every negative type, evenly.
MODEL_B_NEGATIVES = list(NEGATIVE_CLASSES)

# Reproducibility
DATASET_SEED = 0
TRAIN_SEED_A = 1
TRAIN_SEED_B = 2

# Training
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
MAX_EPOCHS = 25
PATIENCE = 5
NUM_WORKERS = 0

# Perlin: feature size of a few cells, similar scale to still lifes / blinkers.
PERLIN_BASE_PERIOD = 4.0
PERLIN_OCTAVES = 3
