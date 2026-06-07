"""Run curated examples through the official submission's predict().

The sandbox imports the submission's model.py once and calls predict() per
hidden (model_id, item_id) pair. We replicate that:

  - Add the submission directory's PARENT to sys.path and import the module as
    `model` (the dispatcher does `from submission.config import CONFIG`, so the
    package root — the folder containing `submission/` — must be importable).
  - Strip the ground-truth `label` from each example before handing the
    four-field dict to predict().
  - Build the `labeled` channel the way the platform does: reveal up to K real
    labels per benchmark (data-category proxy) and pass that SAME list to every
    predict() call (per the spec: "The same labeled list is passed on every
    call within a round").
  - Catch exceptions / invalid outputs per example so one bad call doesn't kill
    the run (and so we can report how the submission's own try/except behaves).
"""

from __future__ import annotations

import importlib
import logging
import math
import random
import sys
import time
from pathlib import Path

import config as C

log = logging.getLogger("simulate.run")

INPUT_KEYS = ("benchmark", "condition", "subject_content", "item_content")


def import_predict():
    """Import submission/model.py and return its predict callable.

    The dispatcher uses absolute imports rooted at the `submission` package
    (`from submission.config import CONFIG`), so we put the PARENT of the
    submission dir on sys.path and import `submission.model`.
    """
    sub_dir = Path(C.SUBMISSION_DIR).resolve()
    model_file = sub_dir / "model.py"
    if not model_file.exists():
        raise FileNotFoundError(
            f"Expected submission entry point at {model_file} (does not exist). "
            f"Set SUBMISSION_DIR in config.py."
        )

    pkg_root = sub_dir.parent
    for p in (str(pkg_root), str(sub_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)

    # Stash simulate_submission's cached 'config' so that submission/model.py's
    # bare `from config import CONFIG` finds submission/config.py, not ours.
    # The caller already holds a reference to our config as 'C', so removing
    # it from sys.modules is safe.
    _SUBMISSION_MODULE_NAMES = [
        "config", "utils", "calibration", "subject_ability", "training_constants"
    ]
    _stashed = {name: sys.modules.pop(name) for name in _SUBMISSION_MODULE_NAMES
                if name in sys.modules}

    # Try the package-qualified import first (matches the dispatcher's own
    # `import submission.stage1 as stage1` style), then a bare fallback.
    last_err: Exception | None = None
    for modname in (f"{sub_dir.name}.model", "model"):
        try:
            mod = importlib.import_module(modname)
            if hasattr(mod, "predict"):
                log.info("Imported predict from %s (%s)", modname, model_file)
                return mod.predict
        except Exception as e:  # noqa: BLE001
            last_err = e
            log.debug("import %s failed: %r", modname, e)

    raise ImportError(
        f"Could not import predict() from {model_file}. Last error: {last_err!r}"
    )


def _strip_to_input(example: dict) -> dict:
    return {k: ("" if example.get(k) is None else str(example[k])) for k in INPUT_KEYS}


def build_labeled_channel(examples: list[dict]) -> list[dict]:
    """Reveal up to K real labels per benchmark, like the platform does.

    Returns a list of dicts with the four input keys plus an int "label".
    Empty when LABELS_PER_BENCHMARK == 0.
    """
    k = int(C.LABELS_PER_BENCHMARK)
    if k <= 0:
        return []
    rng = random.Random(C.SEED + 1)
    by_bench: dict[str, list[dict]] = {}
    for ex in examples:
        by_bench.setdefault(ex["benchmark"], []).append(ex)

    labeled: list[dict] = []
    for bench, rows in by_bench.items():
        pick = rows if len(rows) <= k else rng.sample(rows, k)
        for ex in pick:
            d = _strip_to_input(ex)
            d["label"] = int(ex["label"])
            labeled.append(d)
    log.info(
        "Built labeled channel: %d anchors across %d benchmarks (K=%d).",
        len(labeled),
        len(by_bench),
        k,
    )
    return labeled


def _valid_prob(v) -> bool:
    return (
        isinstance(v, float)
        and not math.isnan(v)
        and not math.isinf(v)
        and 0.0 <= v <= 1.0
    )


def _load_known_subject_filter():
    """Return (known_names:set, parse_fn) from the submission's Stage 1 state,
    or (None, None) if unavailable. Uses the SAME parser the model uses, so the
    hit/miss decision here matches stage1.lookup exactly.
    """
    sub_dir = Path(C.SUBMISSION_DIR).resolve()
    for p in (str(sub_dir.parent), str(sub_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        import importlib

        subject_ability = importlib.import_module(f"{sub_dir.name}.subject_ability")
        state = subject_ability.load()
        # known = set(state["subject_accuracy"].keys())
        import json
        old_means_path = sub_dir.parent / "old_means" / "subject_mean.json"
        with open(old_means_path) as f:
            old_means = json.load(f)
        known = set(old_means.keys())
        from utils import extract_name  # same parser stage1 uses
        return known, extract_name
    except Exception as e:  # noqa: BLE001
        log.warning(
            "Could not load Stage 1 known-subject set for filtering (%r); "
            "RESTRICT_TO_KNOWN_SUBJECTS will be ignored.", e
        )
        return None, None


def _restrict_to_known(examples: list[dict]) -> list[dict]:
    """Drop examples whose parsed subject name is not in the Stage 1 table."""
    known, parse_fn = _load_known_subject_filter()
    if not known or parse_fn is None:
        return examples
    kept = [ex for ex in examples if parse_fn(ex["subject_content"]) in known]
    dropped = len(examples) - len(kept)
    log.info(
        "RESTRICT_TO_KNOWN_SUBJECTS: kept %d, dropped %d unknown-subject examples "
        "(%.1f%% of curated).",
        len(kept), dropped, 100.0 * dropped / max(1, len(examples)),
    )
    if not kept:
        log.warning("Filtering removed ALL examples; check subject rendering vs. "
                    "subject_mean.json keys.")
    return kept


def run(examples: list[dict]) -> dict:
    """Run every example through predict(). Returns a results dict."""
    if getattr(C, "RESTRICT_TO_KNOWN_SUBJECTS", False):
        examples = _restrict_to_known(examples)

    predict = import_predict()
    labeled = build_labeled_channel(examples)
    labeled_arg = labeled if labeled else None

    y_true: list[int] = []
    y_prob: list[float] = []
    rows: list[dict] = []  # per-example records for persistence
    n_exc = 0
    n_invalid = 0
    latencies: list[float] = []

    t0 = time.time()
    for idx, ex in enumerate(examples):
        inp = _strip_to_input(ex)
        rec = {
            "benchmark": ex["benchmark"],
            "condition": ex["condition"],
            "label": int(ex["label"]),
        }
        try:
            t = time.time()
            out = predict(inp, labeled_arg)
            latencies.append(time.time() - t)
        except Exception as e:  # noqa: BLE001 — should not happen; predict catches
            n_exc += 1
            rec.update({"pred": None, "status": f"exception:{type(e).__name__}"})
            rows.append(rec)
            log.error("predict() raised on example %d (%s): %r",
                      idx, ex["benchmark"], e)
            continue

        # Coerce numpy/torch scalars defensively (the contract demands a float).
        try:
            out_f = float(out)
        except (TypeError, ValueError):
            out_f = float("nan")

        if not _valid_prob(out_f):
            n_invalid += 1
            rec.update({"pred": None, "status": f"invalid_output:{out!r}"})
            rows.append(rec)
            log.warning("Invalid predict() output on example %d (%s): %r",
                        idx, ex["benchmark"], out)
            continue

        rec.update({"pred": out_f, "status": "ok"})
        rows.append(rec)
        y_true.append(int(ex["label"]))
        y_prob.append(out_f)

        if (idx + 1) % 500 == 0:
            log.info("  ... %d/%d predicted", idx + 1, len(examples))

    elapsed = time.time() - t0
    log.info(
        "Ran %d examples in %.1fs (%d ok, %d exceptions, %d invalid).",
        len(examples), elapsed, len(y_prob), n_exc, n_invalid,
    )

    return {
        "y_true": y_true,
        "y_prob": y_prob,
        "rows": rows,
        "labeled_count": len(labeled),
        "n_total": len(examples),
        "n_scored": len(y_prob),
        "n_exceptions": n_exc,
        "n_invalid": n_invalid,
        "elapsed_sec": elapsed,
        "mean_latency_ms": (1000 * sum(latencies) / len(latencies))
        if latencies
        else None,
    }