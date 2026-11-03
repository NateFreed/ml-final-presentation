"""Checks for game_of_life.py. Run with: python test_game_of_life.py"""

import matplotlib
matplotlib.use("Agg")  # draw off-screen so the tests don't open windows
import matplotlib.pyplot as plt
import numpy as np

from game_of_life import LifeGrid, parseRule


def naiveStep(frame, birth, survive):
    """Slow cell-by-cell reference version of stepForward, for comparison."""
    n = len(frame)
    out = np.zeros_like(frame)
    for r in range(n):
        for c in range(n):
            k = sum(frame[(r + dr) % n][(c + dc) % n]
                    for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0))
            alive = frame[r][c] == 1
            out[r][c] = (k in survive) if alive else (k in birth)
    return out


def testBlinkerOscillates():
    f = np.zeros((5, 5), np.uint8)
    f[2, 1:4] = 1
    s1 = LifeGrid.conwayStep(f)
    assert s1[1:4, 2].sum() == 3 and s1.sum() == 3
    assert (LifeGrid.conwayStep(s1) == f).all()
    assert f[2, 1:4].sum() == 3 and f.sum() == 3  # input frame was not modified


def testBlockAcrossCornerIsStill():
    # The four cells only form a 2x2 block if both edges wrap around.
    b = np.zeros((6, 6), np.uint8)
    b[[0, 0, 5, 5], [0, 5, 0, 5]] = 1
    assert (LifeGrid.conwayStep(b) == b).all()


def testGliderWrapsAround():
    # A glider moves one cell diagonally every 4 steps, so on an n x n torus
    # it is back where it started after 4n steps.
    n = 8
    g = np.zeros((n, n), np.uint8)
    g[0, 1] = g[1, 2] = g[2, 0] = g[2, 1] = g[2, 2] = 1
    h = g.copy()
    for _ in range(4 * n):
        h = LifeGrid.conwayStep(h)
    assert (h == g).all()


def testRulesMatchNaiveVersion():
    for rule in ["conway", "B2/S0", "B36/S23", "seeds", ((3, 6), (2, 3))]:
        birth, survive = parseRule(rule)
        for seed in range(5):
            frame = LifeGrid(17, 0.4, seed).frame
            assert (LifeGrid.stepForward(frame, rule) == naiveStep(frame, birth, survive)).all(), rule


def testParseRule():
    assert parseRule("B3/S23") == ({3}, {2, 3})
    assert parseRule("live_free_or_die") == ({2}, {0})
    assert parseRule("seeds") == ({2}, set())


def testTupleInput():
    f = tuple(tuple(1 if (r == 2 and 1 <= c <= 3) else 0 for c in range(5)) for r in range(5))
    assert LifeGrid.conwayStep(f).sum() == 3


def testDensityAndSeed():
    assert LifeGrid(20, 0).frame.sum() == 0
    assert LifeGrid(20, 1).frame.sum() == 400
    assert abs(LifeGrid(500, 0.3, seed=1).frame.mean() - 0.3) < 0.01
    assert (LifeGrid(10, 0.5, seed=3).frame == LifeGrid(10, 0.5, seed=3).frame).all()


def testViewColors():
    _, ax = plt.subplots()
    LifeGrid.view(np.array([[0, 1]]), ax=ax)
    rgba = ax.get_images()[0].to_rgba(np.array([[0, 1]]))
    assert tuple(rgba[0, 0][:3]) == (0, 0, 0)  # 0 is black
    assert tuple(rgba[0, 1][:3]) == (1, 1, 1)  # 1 is white
    plt.close("all")


if __name__ == "__main__":
    tests = [fn for name, fn in list(globals().items()) if name.startswith("test")]
    for test in tests:
        test()
        print("passed", test.__name__)
    print(f"all {len(tests)} tests passed")
