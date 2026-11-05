"""Quick checks for board generators. Run: python -m classifier.test_generators"""

from __future__ import annotations

import numpy as np

from .generators import (
    density_matched_noise,
    generate_soup_bundle,
    perlin_board,
    random_board,
    run_rule,
)


def test_density_matched_close():
    rng = np.random.default_rng(0)
    board = density_matched_noise(64, 0.25, rng)
    assert abs(board.mean() - 0.25) < 0.05


def test_perlin_matches_density():
    rng = np.random.default_rng(1)
    for p in (0.1, 0.3, 0.5):
        board = perlin_board(32, p, rng)
        assert abs(board.mean() - p) < 0.02


def test_bundle_shapes():
    bundle = generate_soup_bundle(soup_id=0, size=16, max_steps=5, seed=0)
    assert set(bundle) == {
        "conway",
        "random",
        "density_matched",
        "perlin",
        "live_free_or_die",
        "highlife",
        "day_and_night",
    }
    for arr in bundle.values():
        assert arr.shape == (5, 16, 16)
        assert arr.dtype == np.uint8


def test_run_rule_uses_lifegrid():
    rng = np.random.default_rng(2)
    init = random_board(12, 0.4, rng)
    frames = run_rule(init, "conway", 3)
    assert len(frames) == 3
    assert frames[0].shape == (12, 12)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed", name)
    print("all generator tests passed")
