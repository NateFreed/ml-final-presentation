"""CNN classifier that distinguishes Conway boards from lookalike boards.

Pipeline order (same as the presentation narrative):
1. Generate boards with Life-like rules and noise controls.
2. Split by initial soup so no trajectory leaks across train/val/test.
3. Train Model A (vs fresh noise) and Model B (vs all negatives).
4. Score accuracy and ROC-AUC sliced by step count.
5. Compare to a density/supergrid baseline, then inspect mistakes with LifeGrid.view.
"""
