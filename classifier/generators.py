"""Board generators used as positive and negative examples.

Conway / Live Free or Die / HighLife boards are stepped with the teammates'
`LifeGrid` so the classifier and the visualizer share one rule engine.

The noise generators exist to pressure-test what the CNN is actually using:
- random: fresh Bernoulli noise at density 0.5 (easy; density alone helps)
- density_matched: iid noise with the same live fraction as a Conway board
- perlin: smooth spatial noise thresholded to that same density
"""

from __future__ import annotations

import numpy as np

from game_of_life import LifeGrid

from . import config


def random_board(size: int, density: float, rng: np.random.Generator) -> np.ndarray:
    """Independent Bernoulli cells. Density 0.5 is the default soup."""
    return (rng.random((size, size)) < density).astype(np.uint8)


def density_matched_noise(size: int, target_density: float, rng: np.random.Generator) -> np.ndarray:
    """iid noise whose expected density matches a reference board.

    A classifier that only counts live cells cannot beat chance here.
    """
    target_density = float(np.clip(target_density, 0.0, 1.0))
    return random_board(size, target_density, rng)


def _fade(t: np.ndarray) -> np.ndarray:
    """Perlin's smoothstep: 6t^5 - 15t^4 + 10t^3."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _perlin_octave(size: int, period: float, rng: np.random.Generator) -> np.ndarray:
    """One octave of classic 2D Perlin noise on a torus-friendly lattice."""
    # Lattice spacing: roughly `period` cells between gradient samples.
    grid_n = max(2, int(np.ceil(size / period)) + 1)
    angles = rng.random((grid_n, grid_n)) * 2.0 * np.pi
    grads = np.stack([np.cos(angles), np.sin(angles)], axis=-1)

    ys = np.arange(size, dtype=np.float64)
    xs = np.arange(size, dtype=np.float64)
    # Map pixel coords into lattice space.
    gy = ys * ((grid_n - 1) / size)
    gx = xs * ((grid_n - 1) / size)
    yy, xx = np.meshgrid(gy, gx, indexing="ij")

    x0 = np.floor(xx).astype(int)
    y0 = np.floor(yy).astype(int)
    x1 = np.clip(x0 + 1, 0, grid_n - 1)
    y1 = np.clip(y0 + 1, 0, grid_n - 1)
    fx = xx - x0
    fy = yy - y0
    u = _fade(fx)
    v = _fade(fy)

    def dot(ix, iy, dx, dy):
        g = grads[iy, ix]
        return g[..., 0] * dx + g[..., 1] * dy

    n00 = dot(x0, y0, fx, fy)
    n10 = dot(x1, y0, fx - 1, fy)
    n01 = dot(x0, y1, fx, fy - 1)
    n11 = dot(x1, y1, fx - 1, fy - 1)
    nx0 = n00 * (1 - u) + n10 * u
    nx1 = n01 * (1 - u) + n11 * u
    return nx0 * (1 - v) + nx1 * v


def perlin_field(
    size: int,
    rng: np.random.Generator,
    base_period: float = config.PERLIN_BASE_PERIOD,
    octaves: int = config.PERLIN_OCTAVES,
) -> np.ndarray:
    """Multi-octave Perlin field, roughly zero-mean, unit-ish scale."""
    field = np.zeros((size, size), dtype=np.float64)
    amplitude = 1.0
    period = base_period
    for _ in range(octaves):
        field += amplitude * _perlin_octave(size, period, rng)
        period *= 0.5
        amplitude *= 0.5
    return field


def perlin_board(
    size: int,
    target_density: float,
    rng: np.random.Generator,
    base_period: float = config.PERLIN_BASE_PERIOD,
    octaves: int = config.PERLIN_OCTAVES,
) -> np.ndarray:
    """Threshold Perlin noise so the live fraction matches `target_density`.

    Smooth blobs look "structured" to a human, but they are not Conway ash.
    Matching density removes the global-count shortcut.
    """
    target_density = float(np.clip(target_density, 0.0, 1.0))
    field = perlin_field(size, rng, base_period=base_period, octaves=octaves)
    if target_density <= 0.0:
        return np.zeros((size, size), dtype=np.uint8)
    if target_density >= 1.0:
        return np.ones((size, size), dtype=np.uint8)
    threshold = np.quantile(field, 1.0 - target_density)
    return (field >= threshold).astype(np.uint8)


def run_rule(initial: np.ndarray, rule: str, steps: int) -> list[np.ndarray]:
    """Return frames after 1..steps applications of `rule` (input unchanged)."""
    frame = np.asarray(initial, dtype=np.uint8)
    out = []
    for _ in range(steps):
        frame = LifeGrid.stepForward(frame, rule)
        out.append(frame)
    return out


def generate_soup_bundle(
    soup_id: int,
    size: int = config.GRID_SIZE,
    density: float = config.INITIAL_DENSITY,
    max_steps: int = config.MAX_STEPS,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Build all class boards for one initial soup across steps 1..max_steps.

    Returns a dict class_name -> array of shape (max_steps, size, size),
    indexed so boards[k - 1] is the board after k steps (or the noise
    control paired with Conway at step k).
    """
    rng = np.random.default_rng(seed if seed is not None else soup_id)
    initial = random_board(size, density, rng)

    conway = np.stack(run_rule(initial, "conway", max_steps))
    lfod = np.stack(run_rule(initial, "live_free_or_die", max_steps))
    highlife = np.stack(run_rule(initial, "highlife", max_steps))

    randoms = np.empty_like(conway)
    density_matched = np.empty_like(conway)
    perlins = np.empty_like(conway)
    for k in range(max_steps):
        p = float(conway[k].mean())
        randoms[k] = random_board(size, density, rng)
        density_matched[k] = density_matched_noise(size, p, rng)
        perlins[k] = perlin_board(size, p, rng)

    return {
        "conway": conway,
        "random": randoms,
        "density_matched": density_matched,
        "perlin": perlins,
        "live_free_or_die": lfod,
        "highlife": highlife,
    }
