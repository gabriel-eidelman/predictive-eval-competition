"""
Config for submission
"""

from __future__ import annotations

CONFIG = {
    "calibrate": {
        # Per-category affine maps, empirical-Bayes-shrunk to the global fit.
        # "none" -> single global map for every item.
        "pooling": "category",            # "none" | "category"

        # Fit the intercept b as well as the slope a. False -> pure scale
        # (temperature-like, b == 0).
        "fit_intercept": True,

        # Prior / init for the slope a (a0). The ridge pulls the fit toward
        # (a0, 0); 1.0 == "trust the raw logits' scale by default".
        "init_scale": 1.0,

        # A category needs at least this many anchors to get its own map;
        # below it, the category uses the global map.
        "min_category_anchors": 2,

        # Newton iterations for each affine fit (converges in a handful).
        "newton_iters": 25,

        # L2 toward (init_scale, 0). Stabilises tiny-K fits; lower for sharper
        # fits when anchor counts are healthy, raise if fits look unstable.
        "ridge": 1e-3,

        # Minimum shrinkage weight toward the per-category fit (0 == let the
        # data decide entirely).
        "eb_floor": 0.0,

        # Output clamp.
        "clip_range": (0.02, 0.98),
    },
}