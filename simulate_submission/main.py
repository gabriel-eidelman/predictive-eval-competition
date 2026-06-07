"""Simulate an official Codabench submission end-to-end, offline.

Pipeline:
  1. Curate examples across benchmarks (curate.py).
  2. Convert to predict() input format + carry ground-truth labels.
  3. Run each through submission/model.py:predict, mirroring the platform's
     labeled channel (run_submission.py).
  4. Score with leaderboard metrics — val NLL (higher is better) + AUROC
     (metrics.py).
  5. Print a summary and persist results + metadata to disk.

Run:
    python main.py
"""

from __future__ import annotations

import csv
import json
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import config as C


def setup_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, C.LOG_LEVEL, logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


log = logging.getLogger("simulate.main")


def _per_benchmark_breakdown(rows: list[dict]) -> dict:
    import metrics as M

    by: dict[str, dict] = {}
    grouped: dict[str, list[dict]] = {}
    for r in rows:
        if r.get("status") == "ok" and r.get("pred") is not None:
            grouped.setdefault(r["benchmark"], []).append(r)
    for bench, rs in sorted(grouped.items()):
        yt = [r["label"] for r in rs]
        yp = [r["pred"] for r in rs]
        by[bench] = {
            "n": len(rs),
            "score_nll_higher_better": M.score_nll(yt, yp),
            "auroc": M.auroc(yt, yp),
            "base_rate": sum(yt) / len(yt) if yt else float("nan"),
        }
    return by


def _persist(run_dir: Path, examples, results, overall, per_bench, source) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Per-example results CSV.
    results_csv = run_dir / "results.csv"
    with results_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "condition", "label", "pred", "status"])
        for r in results["rows"]:
            w.writerow(
                [r["benchmark"], r["condition"], r["label"],
                 r.get("pred"), r.get("status")]
            )

    # 2. Metrics JSON.
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {"overall": overall, "per_benchmark": per_bench},
            indent=2,
            default=str,
        )
    )

    # 3. Run metadata (config snapshot + environment + run stats) for reference.
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data_source_used": source,
        "config": {
            "DATA_SOURCE": C.DATA_SOURCE,
            "HF_REPO_ID": C.HF_REPO_ID,
            "BENCHMARK_NAMES": C.BENCHMARK_NAMES,
            "MAX_ITEMS_PER_BENCHMARK": C.MAX_ITEMS_PER_BENCHMARK,
            "MAX_RESPONSES_PER_ITEM": C.MAX_RESPONSES_PER_ITEM,
            "MAX_TOTAL_EXAMPLES": C.MAX_TOTAL_EXAMPLES,
            "LABELS_PER_BENCHMARK": C.LABELS_PER_BENCHMARK,
            "RESTRICT_TO_KNOWN_SUBJECTS": getattr(
                C, "RESTRICT_TO_KNOWN_SUBJECTS", False
            ),
            "SEED": C.SEED,
            "SUBMISSION_DIR": str(C.SUBMISSION_DIR),
        },
        "run_stats": {
            "n_total": results["n_total"],
            "n_scored": results["n_scored"],
            "n_exceptions": results["n_exceptions"],
            "n_invalid": results["n_invalid"],
            "labeled_count": results["labeled_count"],
            "elapsed_sec": results["elapsed_sec"],
            "mean_latency_ms": results["mean_latency_ms"],
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    log.info("Persisted results -> %s", run_dir)


def _print_summary(overall, per_bench, results, source, run_dir) -> None:
    def fmt(x):
        return "n/a" if x is None or (isinstance(x, float) and x != x) else f"{x:.4f}"

    print("\n" + "=" * 68)
    print("  SUBMISSION SIMULATION SUMMARY")
    print("=" * 68)
    print(f"  data source            : {source}")
    print(f"  examples curated       : {results['n_total']}")
    print(f"  examples scored        : {results['n_scored']}"
          f"  (exceptions={results['n_exceptions']}, invalid={results['n_invalid']})")
    print(f"  labeled anchors        : {results['labeled_count']}")
    print(f"  mean predict() latency : "
          f"{fmt(results['mean_latency_ms'])} ms")
    print("-" * 68)
    print("  OVERALL METRICS")
    print(f"    val NLL (higher=better) : {fmt(overall['score_nll_higher_better'])}")
    print(f"    log-loss (lower=better) : {fmt(overall['log_loss'])}")
    print(f"    AUROC                   : {fmt(overall['auroc'])}")
    print(f"    Brier                   : {fmt(overall['brier'])}")
    print(f"    base rate / mean pred   : "
          f"{fmt(overall['base_rate'])} / {fmt(overall['mean_pred'])}")
    if "_sklearn_log_loss" in overall:
        print(f"    [sklearn check] log-loss: {fmt(overall['_sklearn_log_loss'])}")
    print("-" * 68)
    print("  PER-BENCHMARK (val NLL / AUROC / n)")
    for bench, m in per_bench.items():
        print(f"    {bench:<18} "
              f"NLL={fmt(m['score_nll_higher_better']):>8}  "
              f"AUROC={fmt(m['auroc']):>7}  n={m['n']}")
    print("-" * 68)
    print(f"  artifacts: {run_dir}/results.csv, metrics.json, metadata.json")
    print("=" * 68 + "\n")


def main() -> int:
    setup_logging()
    import curate
    import metrics as M
    import run_submission

    log.info("Starting submission simulation.")

    try:
        examples, source = curate.curate_examples()
    except Exception as e:  # noqa: BLE001
        log.error("Curation failed fatally: %r", e)
        return 2

    try:
        results = run_submission.run(examples)
    except Exception as e:  # noqa: BLE001
        log.error("Run failed fatally: %r", e)
        return 3

    overall = M.compute_all(results["y_true"], results["y_prob"])
    per_bench = _per_benchmark_breakdown(results["rows"])

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(C.OUTPUT_DIR) / f"run_{stamp}"
    _persist(run_dir, examples, results, overall, per_bench, source)
    _print_summary(overall, per_bench, results, source, run_dir)

    # Non-zero exit if nothing scored, so CI/automation can catch it.
    return 0 if results["n_scored"] > 0 else 4


if __name__ == "__main__":
    raise SystemExit(main())