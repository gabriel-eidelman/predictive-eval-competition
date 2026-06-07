"""Build subject_accuracy.json (and subject_ability_meta.json) from ALL training responses.
-------------------------------
    p_subject = (sum_correct + alpha * global_mean) / (n_responses + alpha)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Reuse the harness config + the shipped helpers verbatim.
import config as C

log = logging.getLogger("build.subject_accuracy")


# --------------------------------------------------------------------------- #
# Helpers copied/adapted from the shipped pipeline so this script is faithful
# to runtime name resolution. (render_subject_content from curate.py;
# extract_name from the submission's utils.py.)
# --------------------------------------------------------------------------- #
def render_subject_content(subject: dict, fallback_subject_id: str) -> str:
    """Replicates curate.render_subject_content / the README helper."""
    display_name = subject.get("display_name") or fallback_subject_id
    lines = [f"Name: {display_name}"]
    optional_fields = (
        ("provider", "Organization"),
        ("params", "Parameters"),
        ("release_date", "Released"),
        ("family", "Family"),
    )
    for key, label in optional_fields:
        value = subject.get(key)
        if value:
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


def extract_name(subject_content: str) -> str:
    """Replicates submission/utils.extract_name."""
    if not subject_content:
        return ""
    first = subject_content.split("\n", 1)[0].strip()
    if first.lower().startswith("name:"):
        return first.split(":", 1)[1].strip()
    return first


def _coerce_binary(value: Any) -> int | None:
    """Return 0/1 if value is a clean binary response, else None (curate.py rule)."""
    try:
        if value in C.BINARY_VALUES:
            return int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return None


def subject_name_for(subject_id: str, subj_meta: dict[str, dict]) -> str:
    """The canonical lookup key: extract_name(render_subject_content(meta))."""
    meta = subj_meta.get(str(subject_id), {})
    rendered = render_subject_content(meta, str(subject_id))
    return extract_name(rendered)


# --------------------------------------------------------------------------- #
# Source A: HuggingFace Parquet (mirrors curate._curate_hf loading, NO sampling)
# --------------------------------------------------------------------------- #
def _accumulate_hf() -> tuple[dict[str, list[int]], dict[str, dict]]:
    """Return (sum_correct/n per subject_id, subject metadata by subject_id).

    Loads ALL response parquet files and ALL responses (no per-benchmark /
    per-item subsampling). Aggregates clean binary responses per subject_id.
    """
    import pandas as pd  # noqa: F401
    from datasets import Features, Value, load_dataset
    from huggingface_hub import HfApi

    repo_id = C.HF_REPO_ID
    registry = {"subjects.parquet", "items.parquet", "benchmarks.parquet"}

    repo_files = HfApi().list_repo_files(repo_id=repo_id, repo_type="dataset")
    response_files = sorted(
        n
        for n in repo_files
        if n.endswith(".parquet")
        and n not in registry
        and not n.endswith("_traces.parquet")
    )
    if not response_files:
        raise RuntimeError("No response parquet files found in HF repo.")

    response_features = Features(
        {
            "subject_id": Value("string"),
            "item_id": Value("string"),
            "benchmark_id": Value("string"),
            "trial": Value("int64"),
            "test_condition": Value("string"),
            "response": Value("float64"),
            "correct_answer": Value("string"),
            "trace": Value("string"),
        }
    )

    log.info("Loading %d HF response files (ALL responses, no sampling)...",
             len(response_files))
    df = load_dataset(
        repo_id, data_files=response_files, features=response_features, split="train"
    ).to_pandas()
    subjects_df = load_dataset(
        repo_id, data_files="subjects.parquet", split="train"
    ).to_pandas()
    log.info("Loaded %d responses across %d subject rows.",
             len(df), len(subjects_df))

    # Subject metadata by subject_id (for render -> parse name resolution).
    subj_meta: dict[str, dict] = {}
    if "subject_id" in subjects_df.columns:
        for row in subjects_df.to_dict(orient="records"):
            subj_meta[str(row.get("subject_id"))] = row

    # Keep only clean-binary responses (vectorised), then aggregate per subject.
    df = df[["subject_id", "response"]].copy()
    df = df[df["response"].isin(list(C.BINARY_VALUES))].copy()
    if df.empty:
        raise RuntimeError("No binary responses after filtering.")
    df["label"] = df["response"].round().astype(int)

    grp = df.groupby("subject_id")["label"]
    agg = grp.agg(["sum", "count"])
    per_subject: dict[str, list[int]] = {
        str(sid): [int(r["sum"]), int(r["count"])] for sid, r in agg.iterrows()
    }
    log.info("Aggregated %d subjects from %d binary responses.",
             len(per_subject), int(agg["count"].sum()))
    return per_subject, subj_meta


# --------------------------------------------------------------------------- #
# Source B: torch_measure (mirrors curate._curate_torch_measure loading)
# --------------------------------------------------------------------------- #
def _accumulate_torch_measure() -> tuple[dict[str, list[int]], dict[str, dict]]:
    import pandas as pd  # noqa: F401
    from torch_measure.datasets import load

    def _to_pandas(obj):
        import pandas as pd

        if isinstance(obj, pd.DataFrame):
            return obj
        try:
            return obj.to_pandas()
        except AttributeError:
            return pd.DataFrame(list(obj))

    per_subject: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    subj_meta: dict[str, dict] = {}

    # Use every benchmark torch_measure can load, not just BENCHMARK_NAMES,
    # so the subject means use ALL available training data.
    names = list(C.BENCHMARK_NAMES)
    for name in names:
        try:
            data = load(name)
        except Exception as e:  # noqa: BLE001
            log.warning("[%s] load failed (%r); skipping.", name, e)
            continue

        resp = _to_pandas(data.responses)
        try:
            subjects = _to_pandas(data.subjects)
        except AttributeError:
            subjects = None

        subj_c = "subject_id" if "subject_id" in resp.columns else None
        resp_c = next(
            (c for c in ("response", "label", "score", "value") if c in resp.columns),
            None,
        )
        if subj_c is None or resp_c is None:
            log.warning("[%s] missing subject/response column; skipping.", name)
            continue

        if subjects is not None:
            s_idx = subjects
            if s_idx.index.name != "subject_id" and "subject_id" in s_idx.columns:
                s_idx = s_idx.set_index("subject_id")
            for sid, row in s_idx.iterrows():
                subj_meta.setdefault(str(sid), row.to_dict())

        for sid, val in zip(resp[subj_c].astype(str), resp[resp_c]):
            b = _coerce_binary(val)
            if b is None:
                continue
            per_subject[sid][0] += b
            per_subject[sid][1] += 1

    return dict(per_subject), subj_meta


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build(per_subject: dict[str, list[int]],
          subj_meta: dict[str, dict],
          alpha: float) -> tuple[dict[str, float], dict]:
    """Compute the response-weighted global mean and per-subject smoothed means."""
    total_correct = sum(c for c, _ in per_subject.values())
    total_n = sum(n for _, n in per_subject.values())
    if total_n == 0:
        raise RuntimeError("No binary responses aggregated; nothing to build.")
    global_mean = total_correct / total_n
    log.info("Global mean = %.6f over %d binary responses (%d subjects).",
             global_mean, total_n, len(per_subject))

    # Collapse subject_id -> canonical name (the runtime lookup key). If two
    # subject_ids resolve to the same name, merge their counts.
    by_name: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for sid, (c, n) in per_subject.items():
        name = subject_name_for(sid, subj_meta) or str(sid)
        by_name[name][0] += c
        by_name[name][1] += n

    subject_mean: dict[str, float] = {}
    for name, (c, n) in by_name.items():
        subject_mean[name] = (c + alpha * global_mean) / (n + alpha)

    meta = {
        "global_mean": global_mean,
        "alpha": alpha,
        "n_subjects": len(subject_mean),
        "n_responses": total_n,
    }
    return subject_mean, meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["auto", "hf", "torch_measure"],
                    default=C.DATA_SOURCE,
                    help="Ingestion path (default: config.DATA_SOURCE).")
    ap.add_argument("--alpha", type=float, default=10.0,
                    help="Laplace smoothing strength toward global mean (default 10).")
    ap.add_argument("--out-dir", type=Path, default=Path(C.SUBMISSION_DIR),
                    help="Where to write subject_accuracy.json / subject_ability_meta.json "
                         "(default: config.SUBMISSION_DIR).")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, C.LOG_LEVEL, logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    order = ["torch_measure", "hf"] if args.source == "auto" else [args.source]
    per_subject: dict[str, list[int]] | None = None
    subj_meta: dict[str, dict] = {}
    last_err: Exception | None = None
    for src in order:
        try:
            log.info("Accumulating ALL responses via source=%s ...", src)
            if src == "torch_measure":
                per_subject, subj_meta = _accumulate_torch_measure()
            else:
                per_subject, subj_meta = _accumulate_hf()
            if not per_subject:
                raise RuntimeError(f"source={src} produced 0 subjects.")
            log.info("Source %s succeeded.", src)
            break
        except Exception as e:  # noqa: BLE001
            log.warning("source=%s failed: %r", src, e)
            last_err = e
            per_subject = None

    if not per_subject:
        log.error("All sources failed. Last error: %r", last_err)
        return 2

    subject_mean, meta = build(per_subject, subj_meta, args.alpha)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sm_path = out_dir / "subject_accuracy.json"
    meta_path = out_dir / "subject_ability_meta.json"
    sm_path.write_text(json.dumps(subject_mean))
    meta_path.write_text(json.dumps(meta, indent=2))

    log.info("Wrote %d subject means -> %s", len(subject_mean), sm_path)
    log.info("Wrote subject_ability meta -> %s", meta_path)

    # Quick sanity peek.
    sample = list(subject_mean.items())[:5]
    log.info("Sample: %s", sample)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())