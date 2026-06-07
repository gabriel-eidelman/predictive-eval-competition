"""Curate a set of (input_dict, label) examples across benchmarks.

Two ingestion paths, both producing the SAME output: a list of curated examples
where each example is

    {
        "benchmark":        str,   # public benchmark_id (matches runtime input)
        "condition":        str,   # normalized; "none" when not applicable
        "subject_content":  str,   # "Name: ..." + optional metadata lines
        "item_content":     str,   # the question/prompt text
        "label":            int,   # ground-truth 0/1 (held out from predict)
    }

Path A ("torch_measure"): mirrors multi_benchmark.py. Loads each benchmark via
torch_measure.datasets.load(name), introspects response columns, keeps fully
binary items, subsamples, and joins item/subject text.

Path B ("hf"): mirrors the README. Loads the explicit Parquet response tables +
registry tables, joins via to_training_example / render_subject_content.

The label is the binary ground truth. It is NOT placed in the dict handed to
predict(); it is carried alongside for scoring (and for the adaptive-label
channel).
"""

from __future__ import annotations

import logging
import random
from typing import Any, Iterable

import config as C

log = logging.getLogger("simulate.curate")


# --------------------------------------------------------------------------- #
# Shared helpers (README-style subject rendering)
# --------------------------------------------------------------------------- #
def render_subject_content(subject: dict, fallback_subject_id: str) -> str:
    """Replicates the README render_subject_content helper."""
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


def _coerce_binary(value: Any) -> int | None:
    """Return 0/1 if value is a clean binary response, else None."""
    try:
        if value in C.BINARY_VALUES:
            return int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return None


def _cap_examples(examples: list[dict], seed: int) -> list[dict]:
    if len(examples) <= C.MAX_TOTAL_EXAMPLES:
        return examples
    rng = random.Random(seed)
    log.info(
        "Capping examples %d -> %d (MAX_TOTAL_EXAMPLES)",
        len(examples),
        C.MAX_TOTAL_EXAMPLES,
    )
    return rng.sample(examples, C.MAX_TOTAL_EXAMPLES)


# --------------------------------------------------------------------------- #
# Path A: torch_measure (mirrors multi_benchmark.py)
# --------------------------------------------------------------------------- #
_SUBJECT_COL_CANDIDATES = ("subject_id",)
_ITEM_COL_CANDIDATES = ("item_id",)
_RESPONSE_COL_CANDIDATES = ("response", "label", "score", "value")
_CONDITION_COL_CANDIDATES = ("test_condition", "condition")


def _resolve_col(columns: Iterable[str], candidates: tuple[str, ...], what: str) -> str:
    cols = list(columns)
    for c in candidates:
        if c in cols:
            return c
    raise KeyError(
        f"Could not find a {what} column. Looked for {candidates}; saw {cols}."
    )


def _to_pandas(obj):
    import pandas as pd

    if isinstance(obj, pd.DataFrame):
        return obj
    try:
        return obj.to_pandas()
    except AttributeError:
        return pd.DataFrame(list(obj))


def _curate_torch_measure() -> list[dict]:
    import pandas as pd  # noqa: F401
    from torch_measure.datasets import load

    rng = random.Random(C.SEED)
    examples: list[dict] = []

    for name in C.BENCHMARK_NAMES:
        try:
            data = load(name)
        except Exception as e:  # noqa: BLE001
            log.warning("[%s] load failed (%r); skipping.", name, e)
            continue

        resp = _to_pandas(data.responses)
        items = _to_pandas(data.items)
        try:
            subjects = _to_pandas(data.subjects)
        except AttributeError:
            subjects = None

        try:
            subj_c = _resolve_col(resp.columns, _SUBJECT_COL_CANDIDATES, "subject_id")
            item_c = _resolve_col(resp.columns, _ITEM_COL_CANDIDATES, "item_id")
            resp_c = _resolve_col(resp.columns, _RESPONSE_COL_CANDIDATES, "response")
        except KeyError as e:
            log.warning("[%s] %s; skipping.", name, e)
            continue
        cond_c = next((c for c in _CONDITION_COL_CANDIDATES if c in resp.columns), None)

        # Item text lookup.
        items_idx = items
        if items_idx.index.name != "item_id" and "item_id" in items_idx.columns:
            items_idx = items_idx.set_index("item_id")
        content_col = "content" if "content" in items_idx.columns else None
        item_text = {}
        if content_col:
            item_text = items_idx[content_col].astype(str).to_dict()

        # Subject text lookup (rendered like the README).
        subj_text: dict[str, str] = {}
        if subjects is not None:
            s_idx = subjects
            if s_idx.index.name != "subject_id" and "subject_id" in s_idx.columns:
                s_idx = s_idx.set_index("subject_id")
            for sid, row in s_idx.iterrows():
                subj_text[str(sid)] = render_subject_content(row.to_dict(), str(sid))

        # Keep only fully-binary items (any graded response disqualifies the item).
        bin_mask = resp[resp_c].map(lambda v: _coerce_binary(v) is not None)
        per_item_frac = bin_mask.groupby(resp[item_c]).mean()
        fully_binary = set(per_item_frac[per_item_frac == 1.0].index)
        kept = resp[resp[item_c].isin(fully_binary)].copy()
        if kept.empty:
            log.info("[%s] no fully-binary items; dropping.", name)
            continue

        # Subsample items.
        item_ids = sorted(kept[item_c].astype(str).unique())
        n_keep = min(C.MAX_ITEMS_PER_BENCHMARK, len(item_ids))
        chosen = set(rng.sample(item_ids, n_keep))
        kept = kept[kept[item_c].astype(str).isin(chosen)]

        n_before = len(examples)
        # Cap responses per item.
        for iid, grp in kept.groupby(kept[item_c].astype(str)):
            rows = grp.to_dict("records")
            rng.shuffle(rows)
            rows = rows[: C.MAX_RESPONSES_PER_ITEM]
            for r in rows:
                label = _coerce_binary(r[resp_c])
                if label is None:
                    continue
                sid = str(r[subj_c])
                cond = str(r[cond_c]).strip() if cond_c and r.get(cond_c) else ""
                examples.append(
                    {
                        "benchmark": str(name),
                        "condition": cond or "none",
                        "subject_content": subj_text.get(sid, f"Name: {sid}"),
                        "item_content": item_text.get(str(iid), str(iid)),
                        "label": int(label),
                    }
                )
        log.info("[%s] added %d examples", name, len(examples) - n_before)

    return examples


# --------------------------------------------------------------------------- #
# Path B: HuggingFace Parquet (loads like exploration.py, samples like
# multi_benchmark.py)
# --------------------------------------------------------------------------- #
def _curate_hf() -> list[dict]:
    """Load the explicit Parquet tables (exploration.py style: one .to_pandas()
    each), then apply the multi_benchmark.py sampling logic on the DataFrame:
    fully-binary item filter -> stratified per-benchmark item subsample ->
    cap responses per item. Join item/subject text via the registry tables.
    """
    import pandas as pd
    from datasets import Features, Value, load_dataset
    from huggingface_hub import HfApi

    rng = random.Random(C.SEED)
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

    # ---- Load exactly like exploration.py: one load_dataset + one to_pandas ----
    log.info("Loading %d HF response files...", len(response_files))
    df = load_dataset(
        repo_id, data_files=response_files, features=response_features, split="train"
    ).to_pandas()
    items_df = load_dataset(repo_id, data_files="items.parquet", split="train").to_pandas()
    subjects_df = load_dataset(repo_id, data_files="subjects.parquet", split="train").to_pandas()
    benchmarks_df = load_dataset(repo_id, data_files="benchmarks.parquet", split="train").to_pandas()
    log.info(
        "Loaded %d responses, %d items, %d subjects, %d benchmarks.",
        len(df), len(items_df), len(subjects_df), len(benchmarks_df),
    )

    # ---- Registry lookups for text/metadata ----
    content_col = "content" if "content" in items_df.columns else None
    if content_col is None and len(items_df.columns):
        content_col = items_df.columns[0]
    items_idx = items_df.set_index("item_id") if "item_id" in items_df.columns else items_df
    item_text = items_idx[content_col].astype(str).to_dict() if content_col else {}

    subj_text: dict[str, str] = {}
    if "subject_id" in subjects_df.columns:
        for row in subjects_df.to_dict(orient="records"):
            sid = str(row.get("subject_id"))
            subj_text[sid] = render_subject_content(row, sid)

    # public benchmark_id passthrough (exploration uses benchmark_id directly;
    # the README says use the public benchmark_id to match runtime inputs).
    bench_id_map = {}
    if "benchmark_id" in benchmarks_df.columns:
        bench_id_map = {
            str(r["benchmark_id"]): str(r["benchmark_id"])
            for r in benchmarks_df.to_dict(orient="records")
        }

    # ---- multi_benchmark.py-style processing, vectorised over the DataFrame ----
    df = df[["subject_id", "item_id", "benchmark_id", "test_condition", "response"]].copy()

    # Optionally restrict to the configured benchmark list (by benchmark_id).
    # measurement-db benchmark_ids may not match the human-readable
    # multi_benchmark names, so only filter if there is overlap; otherwise keep
    # all benchmarks present in the data.
    wanted = set(C.BENCHMARK_NAMES)
    present = set(df["benchmark_id"].unique())
    overlap = wanted & present
    if overlap:
        df = df[df["benchmark_id"].isin(overlap)].copy()
        log.info("Restricted to %d configured benchmarks present in data.", len(overlap))
    else:
        log.info(
            "Config BENCHMARK_NAMES don't match dataset benchmark_ids; "
            "using all %d benchmarks present.", len(present),
        )

    # Binary mask + fully-binary item filter (an item is kept only if ALL its
    # responses are in {0,1}) -- this is the multi_benchmark.py rule that drops
    # graded benchmarks like ultrafeedback/mtbench.
    bin_mask = df["response"].isin(list(C.BINARY_VALUES))
    per_item_frac = bin_mask.groupby(df["item_id"]).transform("mean")
    df = df[per_item_frac == 1.0].copy()
    if df.empty:
        raise RuntimeError("No fully-binary items after filtering.")
    df["label"] = df["response"].round().astype(int)

    examples: list[dict] = []
    for bench, g in df.groupby("benchmark_id"):
        # Stratified subsample: pick items within this benchmark.
        item_ids = sorted(g["item_id"].astype(str).unique())
        n_keep = min(C.MAX_ITEMS_PER_BENCHMARK, len(item_ids))
        chosen = set(rng.sample(item_ids, n_keep))
        gk = g[g["item_id"].astype(str).isin(chosen)]

        n_before = len(examples)
        for iid, rows in gk.groupby(gk["item_id"].astype(str)):
            recs = rows.to_dict("records")
            rng.shuffle(recs)
            for r in recs[: C.MAX_RESPONSES_PER_ITEM]:
                content = item_text.get(str(iid))
                if not content:
                    continue
                sid = str(r["subject_id"])
                cond = r.get("test_condition")
                cond = str(cond).strip() if cond else ""
                examples.append(
                    {
                        "benchmark": bench_id_map.get(str(bench), str(bench)),
                        "condition": cond or "none",
                        "subject_content": subj_text.get(sid, f"Name: {sid}"),
                        "item_content": content,
                        "label": int(r["label"]),
                    }
                )
        log.info("[%s] added %d examples", bench, len(examples) - n_before)

    return examples


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def curate_examples() -> tuple[list[dict], str]:
    """Return (examples, source_used)."""
    source = C.DATA_SOURCE
    order: list[str]
    if source == "auto":
        order = ["torch_measure", "hf"]
    else:
        order = [source]

    last_err: Exception | None = None
    for src in order:
        try:
            log.info("Curating examples via source=%s ...", src)
            examples = (
                _curate_torch_measure() if src == "torch_measure" else _curate_hf()
            )
            if not examples:
                raise RuntimeError(f"source={src} produced 0 examples.")
            examples = _cap_examples(examples, C.SEED)
            log.info("Curated %d examples via %s.", len(examples), src)
            return examples, src
        except Exception as e:  # noqa: BLE001
            log.warning("source=%s failed: %r", src, e)
            last_err = e

    raise RuntimeError(
        f"All curation sources failed. Last error: {last_err!r}"
    )