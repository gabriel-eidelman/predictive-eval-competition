"""Probability calibration for subject-logit scoring. 

Pipeline per item:
    raw = logit_subject + difficulty_offset            (combine)
    p   = logistic(a * raw + b)                         (affine Platt map)

"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from utils import clamp_prob, logistic


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@dataclass
class CalibratorConfig:
    """Typed calibration settings.

    pooling:
        "none"     -> single global affine map for every item.
        "category" -> per-category maps, empirical-Bayes-shrunk to the global.

    The global map is always fit (it is the shrinkage target and the fallback
    for unseen categories).
    """

    pooling: str = "category"            # "none" | "category"
    fit_intercept: bool = True           # False -> pure scale (temperature-like)
    init_scale: float = 1.0              # a0; <1 widens, >1 sharpens
    clip_range: tuple[float, float] = (0.02, 0.98)
    obs_clip: tuple[float, float] = (0.05, 0.95)   # reserved
    min_category_anchors: int = 2        # below this, category uses global map
    newton_iters: int = 25
    ridge: float = 1e-3                  # L2 toward (init_scale, 0); stabilizes tiny K
    eb_floor: float = 0.0                # min shrink weight toward category fit


@dataclass
class _AffineMap:
    a: float
    b: float

    def logit(self, raw: float) -> float:
        return self.a * raw + self.b


@dataclass
class _Fit:
    glob: _AffineMap
    by_cat: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _sig(z: float) -> float:
    """Numerically safe logistic, used internally for the fits."""
    if z >= 0.0:
        ez = math.exp(-z) if z < 30.0 else 0.0
        return 1.0 / (1.0 + ez)
    ez = math.exp(z) if z > -30.0 else 0.0
    return ez / (1.0 + ez)


def _variance(values: list, ddof: int = 0) -> float:
    n = len(values)
    if n - ddof <= 0:
        return 0.0
    mean = sum(values) / n
    return sum((v - mean) ** 2 for v in values) / (n - ddof)


# ---------------------------------------------------------------------------
# combine
# ---------------------------------------------------------------------------


def combine_logit(subject_out: dict, difficulty_logit: float,
                  w_factor: float = 1.0, bias: float = 0.0) -> float:
    """Raw pre-calibration logit: subject term plus additive difficulty offset.

    The subject term is the scalar per-subject logit, which already carries
    rough item difficulty; difficulty_logit nudges it. w_factor/bias default to
    the identity (the shipped behaviour) and exist only for ablations.
    """
    return w_factor * (subject_out["logit"] + float(difficulty_logit)) + bias


# ---------------------------------------------------------------------------
# fitting helpers
# ---------------------------------------------------------------------------


def _newton_affine(
    raw: list,
    y: list,
    fit_intercept: bool,
    a0: float,
    iters: int,
    ridge: float,
) -> _AffineMap:
    """Fit p = logistic(a*raw + b) by Newton's method on penalised log-loss.

    Minimises  sum BCE(logistic(a*raw+b), y) + 0.5*ridge*((a-a0)^2 + b^2).
    The Hessian of logistic loss is PSD, so Newton with a tiny ridge is stable
    in 1-2 dims and converges in a handful of steps. Falls back to the prior
    (a0, 0) on a degenerate Hessian.
    """
    a, b = float(a0), 0.0
    n = len(raw)
    if n == 0:
        return _AffineMap(a, b)

    for _ in range(iters):
        # Accumulate gradient and Hessian entries in one pass.
        g_a = ridge * (a - a0)
        g_b = ridge * b
        h_aa = ridge
        h_ab = 0.0
        h_bb = ridge
        for r, yi in zip(raw, y):
            p = _sig(a * r + b)
            w = p * (1.0 - p)
            if w < 1e-9:
                w = 1e-9
            resid = p - yi
            g_a += r * resid
            h_aa += w * r * r
            if fit_intercept:
                g_b += resid
                h_ab += w * r
                h_bb += w

        if fit_intercept:
            det = h_aa * h_bb - h_ab * h_ab
            if abs(det) < 1e-12:
                break
            da = (h_bb * g_a - h_ab * g_b) / det
            db = (h_aa * g_b - h_ab * g_a) / det
            a -= da
            b -= db
            step = abs(da) + abs(db)
        else:
            if h_aa < 1e-12:
                break
            da = g_a / h_aa
            a -= da
            step = abs(da)

        if step < 1e-8:
            break

    if not (math.isfinite(a) and math.isfinite(b)):
        return _AffineMap(float(a0), 0.0)
    return _AffineMap(a, b)


def _slope_sampling_var(raw: list, fmap: _AffineMap) -> float:
    """Approx sampling variance of the fitted slope a: inverse Fisher info.

    Var(a_hat) ~= 1 / sum_k raw_k^2 * p_k (1-p_k). Large when anchors are few
    or the model is already saturated (p near 0/1) -> the category fit is
    untrustworthy and should defer to the global map.
    """
    info = 0.0
    for r in raw:
        p = _sig(fmap.logit(r))
        info += r * r * p * (1.0 - p)
    if info <= 1e-12:
        return float("inf")
    return 1.0 / info


# ---------------------------------------------------------------------------
# calibrator
# ---------------------------------------------------------------------------


class Calibrator:
    """Fit-once probability calibrator.

    Usage:
        cal = Calibrator(cfg).fit(labeled, predict_raw_fn)
        p = cal.predict(raw_logit, category="mmlu")
    """

    def __init__(self, cfg: CalibratorConfig = None):
        self.cfg = cfg or CalibratorConfig()
        self._fit = None

    # -- fit ---------------------------------------------------------------

    def fit(self, labeled, predict_raw_fn) -> "Calibrator":
        """Score anchors through predict_raw_fn and fit the affine map(s).

        predict_raw_fn(example) must return the same raw logit the test items
        receive (i.e. combine_logit output) so the calibration is in-distribution.
        Robust to per-example scoring failures: bad rows are dropped, not zipped.
        """
        cfg = self.cfg
        raw, y, cats = [], [], []
        for ex in labeled or []:
            try:
                r = float(predict_raw_fn(ex))
                yv = float(int(ex.get("label", 0)))
            except Exception:
                continue
            if math.isfinite(r):
                raw.append(r)
                y.append(yv)
                cats.append(ex.get("benchmark", "") or "")

        if not raw:
            self._fit = _Fit(glob=_AffineMap(cfg.init_scale, 0.0))
            return self

        glob = _newton_affine(
            raw, y, cfg.fit_intercept, cfg.init_scale, cfg.newton_iters, cfg.ridge
        )
        fit = _Fit(glob=glob)

        if cfg.pooling == "category":
            fit.by_cat = self._fit_categories(raw, y, cats, glob)

        self._fit = fit
        return self

    def _fit_categories(self, raw, y, cats, glob) -> dict:
        cfg = self.cfg
        uniq = list(dict.fromkeys(cats))

        # First pass: raw per-category fits + their slope sampling variances.
        cat_maps = {}
        cat_var = {}
        cat_slopes = []
        for c in uniq:
            r_c = [r for r, cc in zip(raw, cats) if cc == c]
            y_c = [yi for yi, cc in zip(y, cats) if cc == c]
            if len(r_c) < cfg.min_category_anchors:
                continue
            fm = _newton_affine(
                r_c, y_c, cfg.fit_intercept, glob.a, cfg.newton_iters, cfg.ridge
            )
            # A non-positive slope means the category anchors are too noisy to
            # carry directional signal; an inverted map is worse than no map, so
            # drop it and let this category fall back to the global fit.
            if fm.a <= 0.0 or not math.isfinite(fm.a):
                continue
            cat_maps[c] = fm
            v = _slope_sampling_var(r_c, fm)
            cat_var[c] = v
            if math.isfinite(v):
                cat_slopes.append(fm.a)

        # Empirical-Bayes: between-category slope variance tau^2 estimated from
        # the spread of category slopes around the global slope. Shrinkage
        # weight w_c = tau^2 / (tau^2 + var_c) — Stein-style. No tuned lambda.
        if len(cat_slopes) >= 2:
            tau2 = _variance([s - glob.a for s in cat_slopes], ddof=1)
        else:
            tau2 = 0.0

        out = {}
        for c, fm in cat_maps.items():
            v = cat_var[c]
            if not math.isfinite(v) or (tau2 + v) <= 1e-12:
                w = 0.0
            else:
                w = tau2 / (tau2 + v)
            w = max(cfg.eb_floor, min(1.0, w))
            out[c] = _AffineMap(
                a=w * fm.a + (1.0 - w) * glob.a,
                b=w * fm.b + (1.0 - w) * glob.b,
            )
        return out

    # -- predict -----------------------------------------------------------

    def predict(self, raw_logit: float, category: str = None) -> float:
        """Map a raw logit to a calibrated, clipped probability."""
        if self._fit is None:
            raise RuntimeError("Calibrator.predict called before fit()")
        cfg = self.cfg
        fmap = self._fit.glob
        if category is not None and category in self._fit.by_cat:
            fmap = self._fit.by_cat[category]

        lo, hi = cfg.clip_range
        return clamp_prob(logistic(fmap.logit(float(raw_logit))), float(lo), float(hi))


# ---------------------------------------------------------------------------
# functional shim (keeps simple call sites working without the class)
# ---------------------------------------------------------------------------


def calibrated_prob(raw_logit, calibrator: Calibrator, category=None) -> float:
    """Thin wrapper for sites that already hold a fitted Calibrator."""
    return calibrator.predict(raw_logit, category=category)