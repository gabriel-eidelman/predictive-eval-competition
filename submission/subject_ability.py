"""Stage 1 — topic competence retrieval (topic_mean mode).

Per-topic Laplace-smoothed average correctness, falling back to the train-set
overall average as the prior for unseen topics. Yields a scalar logit.

Artifacts: topic_correctness.json (per-topic averages).
Initialization runs once through init(); fetch() performs only dict access plus arithmetic.
"""

from __future__ import annotations

import json
from pathlib import Path

from training_constants import GLOBAL_MEAN
from utils import log_odds, extract_name


_ARTIFACT_DIR = Path(__file__).parent
_CACHE: dict = {}


def init() -> dict:
    """Load the topic-average table. Safe to call repeatedly."""
    if _CACHE:
        return _CACHE

    by_topic = _load_json("topic_correctness.json", fallback={})
    config = {
        "logit_overall": log_odds(GLOBAL_MEAN),
        # label -> smoothed average correctness in [0, 1]
        "topic_correctness": {k: float(v) for k, v in by_topic.items()},
    }
    _CACHE.update(config)
    return config


def fetch(config: dict, topic_text: str) -> dict:
    """Return {'logit': ...} for a topic. Tolerates unseen topics without error."""
    label = extract_name(topic_text)
    average = config["topic_correctness"].get(label)
    if average is None:
        return {"logit": config["logit_overall"]}
    return {"logit": log_odds(average)}


load = init  # alias expected by the simulator harness


def _load_json(name: str, fallback):
    file_path = _ARTIFACT_DIR / name
    if not file_path.exists():
        return fallback
    with open(file_path) as f:
        return json.load(f)