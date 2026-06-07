"""Scoring metrics matching the competition.

Primary: negative log-loss, reported higher-is-better (the leaderboard shows
the mean log-likelihood of the ground-truth labels; larger = better).

    log_loss   = -(1/N) * sum[ y*log(p) + (1-y)*log(1-p) ]   (lower is better)
    score_nll  = -log_loss = mean log-likelihood                (higher is better)

Secondary: AUC-ROC (higher is better), computed without sklearn so the harness
runs even in a bare environment. If sklearn is present we cross-check.
"""

from __future__ import annotations

import math


def _clip(p: float, eps: float = 1e-15) -> float:
    return min(1.0 - eps, max(eps, p))


def log_loss(y_true: list[int], y_prob: list[float]) -> float:
    """Mean binary cross-entropy (lower is better)."""
    if not y_true:
        return float("nan")
    total = 0.0
    for y, p in zip(y_true, y_prob):
        p = _clip(float(p))
        total += y * math.log(p) + (1 - y) * math.log(1 - p)
    return -total / len(y_true)


def score_nll(y_true: list[int], y_prob: list[float]) -> float:
    """Leaderboard-style negative log-loss (higher is better)."""
    return -log_loss(y_true, y_prob)


def auroc(y_true: list[int], y_prob: list[float]) -> float:
    """AUC-ROC via the rank (Mann-Whitney U) statistic, with tie handling."""
    pos = [p for y, p in zip(y_true, y_prob) if y == 1]
    neg = [p for y, p in zip(y_true, y_prob) if y == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")  # undefined with a single class

    # Rank all scores (average ranks for ties).
    paired = sorted(
        [(p, 1) for p in pos] + [(p, 0) for p in neg], key=lambda t: t[0]
    )
    ranks = [0.0] * len(paired)
    i = 0
    while i < len(paired):
        j = i
        while j + 1 < len(paired) and paired[j + 1][0] == paired[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-indexed average rank
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1

    sum_ranks_pos = sum(r for r, (_, lbl) in zip(ranks, paired) if lbl == 1)
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return auc


def brier(y_true: list[int], y_prob: list[float]) -> float:
    """Mean squared error of probabilities (lower is better); handy extra."""
    if not y_true:
        return float("nan")
    return sum((p - y) ** 2 for y, p in zip(y_true, y_prob)) / len(y_true)


def compute_all(y_true: list[int], y_prob: list[float]) -> dict:
    out = {
        "n": len(y_true),
        "n_pos": sum(1 for y in y_true if y == 1),
        "n_neg": sum(1 for y in y_true if y == 0),
        "log_loss": log_loss(y_true, y_prob),
        "score_nll_higher_better": score_nll(y_true, y_prob),
        "auroc": auroc(y_true, y_prob),
        "brier": brier(y_true, y_prob),
        "mean_pred": (sum(y_prob) / len(y_prob)) if y_prob else float("nan"),
        "base_rate": (sum(y_true) / len(y_true)) if y_true else float("nan"),
    }
    # Optional cross-check against sklearn.
    try:
        from sklearn.metrics import log_loss as sk_ll, roc_auc_score

        out["_sklearn_log_loss"] = float(sk_ll(y_true, y_prob, labels=[0, 1]))
        if out["n_pos"] and out["n_neg"]:
            out["_sklearn_auroc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        pass
    return out
