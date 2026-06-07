"""Dispatcher for predict().

All state loads at module level (the sandbox imports this once per container).
predict() does lookups + arithmetic only — never I/O, never loading.

Shipped path:
    subject_ability   per-subject smoothed mean logit (subject_mean lookup)
    difficulty        constant DIFFICULTY_LOGIT offset (from training_constants)
    calibrate         joint affine (Platt) map, EB-pooled per category, then clip

The Calibrator is fit ONCE, lazily, on the first call that carries labeled
anchors, and reused thereafter. The platform passes the same labeled-list
object every call within a round, so identity of that object is our "already
fit?" signal — once fit against it, later calls skip straight to predict().
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from config import CONFIG                                  # noqa: E402
import subject_ability                                      # noqa: E402
from training_constants import DIFFICULTY_LOGIT            # noqa: E402
from calibration import (                                  # noqa: E402
    Calibrator,
    CalibratorConfig,
    combine_logit,
)


# ---------------------------------------------------------------------------
# Module-level init: load active artifacts. Runs ONCE.
# ---------------------------------------------------------------------------

_SUBJECT_STATE = subject_ability.init()
_DIFFICULTY_LOGIT = float(DIFFICULTY_LOGIT)

_CAL_CFG = CalibratorConfig(**CONFIG["calibrate"])
_CLIP_LO, _CLIP_HI = _CAL_CFG.clip_range

# Lazily-fit calibrator state. _CAL is the instance; _FIT_KEY is the id() of
# the labeled list it was fit against, so we re-fit only if a genuinely new
# anchor set arrives.
_CAL: Calibrator | None = None
_FIT_KEY: int | None = None


def _raw_logit(input_dict: dict) -> float:
    """Raw (pre-calibration) logit for a single input.

    Re-used during fitting to score the labeled anchors through the exact same
    pipeline as the test items.
    """
    s = subject_ability.fetch(_SUBJECT_STATE, input_dict.get("subject_content", ""))
    return combine_logit(s, _DIFFICULTY_LOGIT)


def _get_calibrator(labeled: list[dict] | None) -> Calibrator:
    """Return a fitted Calibrator, fitting once per distinct anchor set."""
    global _CAL, _FIT_KEY
    key = id(labeled) if labeled else None
    if _CAL is None or key != _FIT_KEY:
        _CAL = Calibrator(_CAL_CFG).fit(labeled, _raw_logit)
        _FIT_KEY = key
    return _CAL


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    """Return P(subject passes item) in [clip_lo, clip_hi]. Never raises."""
    try:
        cal = _get_calibrator(labeled)
        z = _raw_logit(input)
        return float(cal.predict(z, category=input.get("benchmark")))
    except Exception:
        # Absolute floor: never raise, never return a non-float.
        return float((_CLIP_LO + _CLIP_HI) / 2.0)