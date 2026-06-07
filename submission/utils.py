from __future__ import annotations

import math


def logistic(x: float) -> float:
    if x >= 0:
        ez = math.exp(-x)
        return 1.0 / (1.0 + ez)
    ez = math.exp(x)
    return ez / (1.0 + ez)


def log_odds(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def extract_name(subject_content: str) -> str:
    """Extract the subject name from the leading 'Name: ...' line.

    Everything after the first newline is optional metadata; the name is the
    canonical identifier used for lookups.
    """
    if not subject_content:
        return ""
    first = subject_content.split("\n", 1)[0].strip()
    if first.lower().startswith("name:"):
        return first.split(":", 1)[1].strip()
    return first


def clamp_prob(p: float, lo: float = 0.02, hi: float = 0.98) -> float:
    if p < lo:
        return lo
    if p > hi:
        return hi
    return p
