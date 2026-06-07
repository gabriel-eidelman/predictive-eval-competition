"""multi_benchmark.py — pool, filter, and subsample the measurement-db benchmarks.

The single-benchmark `torch_measure.datasets.load(name)` returns a LongFormData
for one benchmark. This module pools many benchmarks for the PGE pipeline:

  1. Loads each benchmark in BENCHMARK_NAMES via load(name).
  2. Drops non-binary responses (response not in {0,1}) and any item that is not
     fully binary — this excludes graded benchmarks (ultrafeedback 1-5,
     mtbench 1-10) and any stray graded rows elsewhere.
  3. Subsamples min(MAX_ITEMS_PER_BENCHMARK, available) items per benchmark,
     stratified so every benchmark is represented.
  4. Computes each item's modal test_condition and attaches it, plus benchmark_id,
     as `condition` / `benchmark` columns on a combined items DataFrame.
  5. Returns a combined items_df + a pooled ResponseMatrix.

Robustness: `data.responses` column names are introspected against candidate
sets rather than hardcoded, so a schema rename surfaces as a clear error naming
the columns actually seen, instead of a silent wrong result.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch

from torch_measure.datasets import load
from torch_measure.data.response_matrix import ResponseMatrix

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Graded benchmarks (ultrafeedback, mtbench) are dropped by the binary filter.
BENCHMARK_NAMES: list[str] = [
    "mmbench_v11",
    "ultrafeedback",   # graded 1-5 -> dropped by binary filter
    "ai2d_test",
    "mmlupro",
    "rewardbench",
    "bfcl",
    "livecodebench",
    "mathvista_mini",
    "afrimedqa",
    "matharena",
    "agentdojo",
    "swebench",
    "hle",
    "mtbench",         # graded 1-10 -> dropped by binary filter
    "cybench",
    "androidworld",
]

MAX_ITEMS_PER_BENCHMARK = 1000
SEED = 42

# Candidate column names in data.responses (introspected, first match wins).
_SUBJECT_COL_CANDIDATES = ("subject_id",)
_ITEM_COL_CANDIDATES = ("item_id",)
_RESPONSE_COL_CANDIDATES = ("response", "label", "score", "value")
_CONDITION_COL_CANDIDATES = ("test_condition", "condition")


@dataclass
class MultiBenchmarkData:
    """Bundle returned by load_multi_benchmark.

    items     : combined items DataFrame, indexed by item_id, with at least
                columns `content`, `benchmark`, `condition`.
    rm         : pooled ResponseMatrix over the kept (subject, item) pairs.
    benchmark_of_item : dict item_id -> benchmark name (for stratified split).
    """
    items: pd.DataFrame
    rm: ResponseMatrix
    benchmark_of_item: dict[str, str]


# ---------------------------------------------------------------------------
# Column resolution
# ---------------------------------------------------------------------------


def _resolve_col(df: pd.DataFrame, candidates: tuple[str, ...], what: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"Could not find a {what} column in responses. Looked for "
        f"{candidates}; saw columns {list(df.columns)}. "
        f"Update the candidate list in multi_benchmark.py."
    )


def _responses_df(data) -> pd.DataFrame:
    """Coerce LongFormData.responses to a pandas DataFrame."""
    resp = data.responses
    if isinstance(resp, pd.DataFrame):
        return resp
    # HF Dataset or list-of-dicts -> DataFrame
    try:
        return resp.to_pandas()  # datasets.Dataset
    except AttributeError:
        return pd.DataFrame(list(resp))


def _items_df(data) -> pd.DataFrame:
    items = data.items
    if isinstance(items, pd.DataFrame):
        return items
    try:
        return items.to_pandas()
    except AttributeError:
        return pd.DataFrame(list(items))


# ---------------------------------------------------------------------------
# Per-benchmark processing
# ---------------------------------------------------------------------------


def _process_one_benchmark(
    name: str,
    *,
    max_items: int,
    seed: int,
    verbose: bool,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Load one benchmark; return (kept_items_df, kept_responses_df) or None.

    kept_items_df    : indexed by item_id, columns content/benchmark/condition.
    kept_responses_df: long-form rows (subject_id, item_id, response 0/1) for the
                       kept items only.
    """
    try:
        data = load(name)
    except Exception as e:  # noqa: BLE001 — surface load failures, keep going
        if verbose:
            print(f"  [{name}] load failed ({e!r}); skipping.")
        return None

    resp = _responses_df(data)
    items = _items_df(data)

    subj_c = _resolve_col(resp, _SUBJECT_COL_CANDIDATES, "subject_id")
    item_c = _resolve_col(resp, _ITEM_COL_CANDIDATES, "item_id")
    resp_c = _resolve_col(resp, _RESPONSE_COL_CANDIDATES, "response")
    cond_c = next((c for c in _CONDITION_COL_CANDIDATES if c in resp.columns), None)

    resp = resp[[subj_c, item_c, resp_c] + ([cond_c] if cond_c else [])].copy()
    resp.columns = ["subject_id", "item_id", "response"] + (["condition"] if cond_c else [])

    # --- Binary filter: keep only 0/1 responses ---
    is_binary = resp["response"].isin([0, 0.0, 1, 1.0])
    n_total = len(resp)
    resp = resp[is_binary].copy()
    resp["response"] = resp["response"].astype(float)

    if resp.empty:
        if verbose:
            print(f"  [{name}] no binary responses ({n_total} rows, all graded); dropping benchmark.")
        return None

    # Drop items that had any non-binary response.
    graded_items = set(resp.loc[~is_binary.reindex(resp.index, fill_value=True), "item_id"]) \
        if n_total != len(resp) else set()
    full_resp = _responses_df(data)[[item_c, resp_c]].copy()
    full_resp.columns = ["item_id", "response"]
    per_item_binary_frac = (
        full_resp["response"].isin([0, 0.0, 1, 1.0])
        .groupby(full_resp["item_id"]).mean()
    )
    fully_binary_items = set(per_item_binary_frac[per_item_binary_frac == 1.0].index)
    resp = resp[resp["item_id"].isin(fully_binary_items)].copy()

    if resp.empty:
        if verbose:
            print(f"  [{name}] no fully-binary items; dropping benchmark.")
        return None

    # --- Subsample items (stratified is just this benchmark's pool) ---
    item_ids_all = pd.Series(sorted(resp["item_id"].unique()))
    n_keep = min(max_items, len(item_ids_all))
    kept_ids = set(
        item_ids_all.sample(n=n_keep, random_state=seed).tolist()
    )
    resp = resp[resp["item_id"].isin(kept_ids)].copy()

    # --- Modal condition per item ---
    if "condition" in resp.columns:
        modal = (
            resp.assign(condition=resp["condition"].fillna("none").astype(str))
            .groupby("item_id")["condition"]
            .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "none")
        )
    else:
        modal = pd.Series("none", index=sorted(kept_ids), name="condition")

    # --- Build kept items frame ---
    items = items.copy()
    if items.index.name != "item_id":
        if "item_id" in items.columns:
            items = items.set_index("item_id")
    items = items.loc[items.index.intersection(kept_ids)]
    items["benchmark"] = name
    items["condition"] = modal.reindex(items.index).fillna("none").astype(str)

    # Drop the per-response condition column from resp before returning (matrix
    # only needs subject/item/response).
    resp = resp[["subject_id", "item_id", "response"]]

    if verbose:
        print(
            f"  [{name}] kept {len(items)} items / {len(resp):,} binary responses "
            f"({len(item_ids_all)} binary items available)"
        )
    return items, resp


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def load_multi_benchmark(
    *,
    benchmark_names: list[str] = BENCHMARK_NAMES,
    max_items_per_benchmark: int = MAX_ITEMS_PER_BENCHMARK,
    seed: int = SEED,
    verbose: bool = True,
) -> MultiBenchmarkData:
    """Load, binary-filter, subsample, and pool the requested benchmarks.

    Returns a MultiBenchmarkData with a combined items DataFrame (indexed by
    item_id, carrying `content`/`benchmark`/`condition`) and a pooled
    ResponseMatrix. Subjects are the union across benchmarks; cells absent for a
    given (subject, item) are NaN (unobserved), which the EM loop already
    handles.
    """
    SEP = "-" * 60
    if verbose:
        print(f"{SEP}\nMulti-benchmark load: {len(benchmark_names)} benchmarks\n{SEP}")

    items_frames: list[pd.DataFrame] = []
    resp_frames: list[pd.DataFrame] = []

    for name in benchmark_names:
        out = _process_one_benchmark(
            name, max_items=max_items_per_benchmark, seed=seed, verbose=verbose
        )
        if out is None:
            continue
        items_one, resp_one = out
        items_frames.append(items_one)
        resp_frames.append(resp_one)

    if not items_frames:
        raise RuntimeError("No benchmarks yielded any binary items; nothing to train on.")

    items_df = pd.concat(items_frames, axis=0)
    resp_df = pd.concat(resp_frames, axis=0, ignore_index=True)

    # item_ids must be globally unique to pool safely.
    if items_df.index.duplicated().any():
        dupes = items_df.index[items_df.index.duplicated()].unique()[:5].tolist()
        raise ValueError(
            f"Duplicate item_ids across benchmarks (e.g. {dupes}); "
            f"item ids must be globally unique to pool safely."
        )

    # keep item_id as both index and column — both access patterns are used downstream
    items_df.index.name = "item_id"
    items_df["item_id"] = items_df.index.astype(str)

    benchmark_of_item = items_df["benchmark"].to_dict()

    # --- Build the pooled ResponseMatrix from long form ---
    rm = _response_matrix_from_long(resp_df)

    if verbose:
        bin_counts = items_df["benchmark"].value_counts()
        print(f"{SEP}")
        print(f"Pooled: {len(items_df)} items across {bin_counts.size} benchmarks, "
              f"{rm.data.shape[0]} subjects")
        print(f"  per-benchmark item counts:\n{bin_counts.to_string()}")
        print(SEP)

    return MultiBenchmarkData(items=items_df, rm=rm, benchmark_of_item=benchmark_of_item)


def _response_matrix_from_long(resp_df: pd.DataFrame) -> ResponseMatrix:
    """Pivot a long (subject_id, item_id, response) frame to a ResponseMatrix.

    Uses a pandas pivot to a dense (subjects x items) matrix with NaN for
    unobserved cells, then wraps it. If multiple responses exist for the same
    (subject, item) pair (repeated trials), they are averaged then re-binarised
    by rounding — a pragmatic collapse; trials are not modelled separately here.
    """
    # Average duplicate (subject,item) cells, then round back to {0,1}.
    pivot = resp_df.pivot_table(
        index="subject_id",
        columns="item_id",
        values="response",
        aggfunc="mean",
    )
    # re-binarize averaged cells (e.g. 2/3 trials correct -> 1), keeping NaN
    # ResponseMatrix.data must be a float32 tensor (observed_mask calls torch.isnan)
    data = torch.as_tensor(pivot.to_numpy(), dtype=torch.float32)
    observed = ~torch.isnan(data)
    data[observed] = torch.round(data[observed])

    subject_ids = list(pivot.index.astype(str))
    item_ids = list(pivot.columns.astype(str))

    # ResponseMatrix has no public dense constructor; set attributes directly
    rm = ResponseMatrix.__new__(ResponseMatrix)
    rm.data = data
    rm.subject_ids = subject_ids
    rm.item_ids = item_ids
    return rm


# ---------------------------------------------------------------------------
# Stratified within-benchmark item split
# ---------------------------------------------------------------------------


def stratified_item_split(
    item_ids: list[str],
    benchmark_of_item: dict[str, str],
    eval_frac: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Split items into train/eval, stratified so each benchmark appears in both.

    Within every benchmark, eval_frac of its items go to eval. Order within each
    benchmark is shuffled deterministically by seed.
    """
    import random

    by_bench: dict[str, list[str]] = {}
    for it in item_ids:
        by_bench.setdefault(benchmark_of_item[it], []).append(it)

    rng = random.Random(seed)
    train_ids: list[str] = []
    eval_ids: list[str] = []
    for bench, ids in sorted(by_bench.items()):
        ids = sorted(ids)
        rng.shuffle(ids)
        n_eval = max(1, int(eval_frac * len(ids))) if len(ids) > 1 else 0
        eval_ids.extend(ids[:n_eval])
        train_ids.extend(ids[n_eval:])
    return train_ids, eval_ids