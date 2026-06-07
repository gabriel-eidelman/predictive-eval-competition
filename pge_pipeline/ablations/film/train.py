"""train.py — PGE train phase, FiLM ablation.

FiLM joint-EM on pooled benchmarks: stage 2a regressor is conditioned on
(benchmark, condition) via FiLM modulation.
"""

from __future__ import annotations

import sys
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import pandas as pd

from src.pge.stage1 import run_stage1
from src.pge.stage2a import run_stage2a
from src.pge import checkpoint as ckpt
from src.pge.pipeline_common import embed_texts, get_texts, embed_dim
from src.pge.multi_benchmark import (
    load_multi_benchmark,
    stratified_item_split,
    BENCHMARK_NAMES,
    MAX_ITEMS_PER_BENCHMARK,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATASET = "measurement-db-multi"  # run label only
EVAL_ITEM_FRAC = 0.2
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_BATCH_SIZE = 256
N_FACTORS = 4
STAGE1_EPOCHS = 150
STAGE1_LR = 0.05

STAGE2A_EPOCHS = 500
STAGE2A_LR = 1e-2
STAGE2A_ALPHA = 1e-3
STAGE2A_HIDDEN_DIM = 128
STAGE2A_CAT_EMB_DIM = 64
STAGE2A_N_RESIDUAL_LAYERS = 2

CONDITION_COL = "condition"
BENCHMARK_COL = "benchmark"

# --- Benchmark-description ablation -----------------------------------------
# When True, prepend each benchmark's description to its item texts before
# embedding. Set False for the baseline (no descriptions) ablation arm.
INCLUDE_BENCHMARK_DESCRIPTION = False
# JSON mapping benchmark name -> description string. Keys must match the values
# in items_df[BENCHMARK_COL]. Separator inserted between description and item.
DESCRIPTIONS_PATH = Path(__file__).resolve().parent / "dataset_descriptions.json"
DESCRIPTION_SEP = "\n\n"

MAX_ITEMS = MAX_ITEMS_PER_BENCHMARK

SEED = 42
DEVICE = "cpu"

CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"


def train(
    *,
    benchmark_names: list[str] = BENCHMARK_NAMES,
    max_items_per_benchmark: int = MAX_ITEMS,
    dataset_label: str = DATASET,
    eval_item_frac: float = EVAL_ITEM_FRAC,
    embed_model: str = EMBED_MODEL,
    embed_batch_size: int = EMBED_BATCH_SIZE,
    n_factors: int = N_FACTORS,
    stage1_epochs: int = STAGE1_EPOCHS,
    stage1_lr: float = STAGE1_LR,
    stage2a_epochs: int = STAGE2A_EPOCHS,
    stage2a_lr: float = STAGE2A_LR,
    stage2a_alpha: float = STAGE2A_ALPHA,
    stage2a_hidden_dim: int = STAGE2A_HIDDEN_DIM,
    stage2a_cat_emb_dim: int = STAGE2A_CAT_EMB_DIM,
    stage2a_n_residual_layers: int = STAGE2A_N_RESIDUAL_LAYERS,
    condition_col: str | None = CONDITION_COL,
    benchmark_col: str | None = BENCHMARK_COL,
    include_benchmark_description: bool = INCLUDE_BENCHMARK_DESCRIPTION,
    descriptions_path: Path = DESCRIPTIONS_PATH,
    description_sep: str = DESCRIPTION_SEP,
    seed: int = SEED,
    device: str = DEVICE,
    ckpt_dir: Path = CKPT_DIR,
    verbose: bool = True,
) -> Path:
    """Fit stage 1 + stage 2a (FiLM joint EM) on pooled benchmarks. Returns run dir."""
    SEP = "=" * 60

    # --- 0. Load + pool + filter + subsample benchmarks ---
    if verbose:
        print(f"\n{SEP}\nTRAIN STEP 0: load + pool benchmarks\n{SEP}")
    mb = load_multi_benchmark(
        benchmark_names=benchmark_names,
        max_items_per_benchmark=max_items_per_benchmark,
        seed=seed,
        verbose=verbose,
    )
    items_df: pd.DataFrame = mb.items
    rm = mb.rm
    all_item_ids: list[str] = rm.item_ids

    # WITHIN-BENCHMARK split: each benchmark in both train and eval.
    train_item_ids, eval_item_ids = stratified_item_split(
        all_item_ids, mb.benchmark_of_item, eval_item_frac, seed
    )
    if verbose:
        print(f"  Total {len(all_item_ids)} | train {len(train_item_ids)} | "
              f"eval {len(eval_item_ids)} (stratified within benchmark)")

    # --- 0b. Build GLOBAL categorical id maps over ALL pooled items ---
    condition_map = _build_id_map(items_df, condition_col)
    benchmark_map = _build_id_map(items_df, benchmark_col)
    n_conditions = _vocab_size(condition_map)
    n_benchmarks = _vocab_size(benchmark_map)
    if verbose:
        print(f"  condition vocab: {n_conditions} (col={condition_col!r}) | "
              f"benchmark vocab: {n_benchmarks} (col={benchmark_col!r})")

    # --- 0c. Load benchmark descriptions (ablation) ---
    descriptions = _load_descriptions(
        descriptions_path, include_benchmark_description, verbose
    )
    if verbose:
        print(f"  benchmark description prepend: {include_benchmark_description} "
              f"({len(descriptions)} descriptions loaded)")

    # --- 1. Stage 1 ---
    if verbose:
        print(f"\n{SEP}\nTRAIN STEP 1: stage 1 factor model\n{SEP}")
    stage1_out = run_stage1(
        train_item_ids=train_item_ids,
        response_matrix=rm,
        n_factors=n_factors,
        max_epochs=stage1_epochs,
        lr=stage1_lr,
        seed=seed,
        device=device,
        verbose=verbose,
    )

    # --- 1b. Build the raw TRAIN response matrix for the EM loop ---
    if verbose:
        print(f"\n{SEP}\nTRAIN STEP 1b: build train response matrix for EM\n{SEP}")
    train_responses = _build_train_responses(
        rm,
        subject_ids=stage1_out["subject_ids"],
        train_item_ids=stage1_out["item_ids"],
    )
    stage1_out["responses"] = train_responses
    if verbose:
        obs = ~torch.isnan(train_responses)
        print(f"  Train matrix {tuple(train_responses.shape)} | "
              f"{int(obs.sum()):,} obs ({obs.float().mean():.1%})")

    # --- 1c. Encode TRAIN-item categorical ids (stage-1 item order) ---
    train_condition_ids = _encode_ids(items_df, condition_col, condition_map, stage1_out["item_ids"])
    train_benchmark_ids = _encode_ids(items_df, benchmark_col, benchmark_map, stage1_out["item_ids"])
    stage1_out["condition_ids"] = train_condition_ids
    stage1_out["benchmark_ids"] = train_benchmark_ids
    stage1_out["n_conditions"] = n_conditions
    stage1_out["n_benchmarks"] = n_benchmarks

    # --- 2. Embed TRAIN item texts only ---
    if verbose:
        print(f"\n{SEP}\nTRAIN STEP 2: embed train item texts\n{SEP}")
    train_texts = get_texts(items_df, stage1_out["item_ids"])
    train_texts = _prepend_benchmark_descriptions(
        train_texts,
        item_ids=stage1_out["item_ids"],
        items_df=items_df,
        benchmark_col=benchmark_col,
        descriptions=descriptions,
        enabled=include_benchmark_description,
        sep=description_sep,
        verbose=verbose,
    )
    train_embeddings = embed_texts(
        train_texts, embed_model, embed_batch_size, device, verbose
    )

    # --- 3. Stage 2a — Joint EM (FiLM-modulated) ---
    if verbose:
        print(f"\n{SEP}\nTRAIN STEP 3: stage 2a FiLM joint-EM item-param regressor\n{SEP}")
    stage2a_out = run_stage2a(
        stage1_out,
        train_embeddings,
        condition_ids=train_condition_ids,
        benchmark_ids=train_benchmark_ids,
        n_conditions=n_conditions,
        n_benchmarks=n_benchmarks,
        max_epochs=stage2a_epochs,
        lr=stage2a_lr,
        alpha=stage2a_alpha,
        hidden_dim=stage2a_hidden_dim,
        cat_emb_dim=stage2a_cat_emb_dim,
        n_residual_layers=stage2a_n_residual_layers,
        seed=seed,
        device=device,
        verbose=verbose,
        use_condition=False,
        use_benchmark=True,    
    )

    # --- 4. Save checkpoint ---
    if verbose:
        print(f"\n{SEP}\nTRAIN STEP 4: save checkpoint\n{SEP}")
    ts = datetime.now(timezone.utc)
    run_id = f"{dataset_label}_{ts.strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = ckpt_dir / run_id

    ckpt.save_stage1(stage1_out, run_dir / "stage1.pt")
    ckpt.save_stage2a(stage2a_out, run_dir / "stage2a.pt")

    manifest = {
        "run_id": run_id,
        "timestamp_utc": ts.isoformat(),
        "dataset": dataset_label,
        "benchmark_names": benchmark_names,
        "train_item_ids": stage1_out["item_ids"],
        "eval_item_ids": eval_item_ids,
        "config": {
            "eval_item_frac": eval_item_frac,
            "max_items_per_benchmark": max_items_per_benchmark,
            "embed_model": embed_model,
            "embed_batch_size": embed_batch_size,
            "n_factors": n_factors,
            "stage1_epochs": stage1_epochs,
            "stage1_lr": stage1_lr,
            "stage2a_epochs": stage2a_epochs,
            "stage2a_lr": stage2a_lr,
            "stage2a_alpha": stage2a_alpha,
            "stage2a_hidden_dim": stage2a_hidden_dim,
            "stage2a_cat_emb_dim": stage2a_cat_emb_dim,
            "stage2a_n_residual_layers": stage2a_n_residual_layers,
            "stage2a_variant": "amortized_em_film_multibench",
            "split": "within_benchmark_stratified",
            "condition_col": condition_col,
            "benchmark_col": benchmark_col,
            "n_conditions": n_conditions,
            "n_benchmarks": n_benchmarks,
            "include_benchmark_description": include_benchmark_description,
            "description_sep": description_sep,
            "seed": seed,
            "device": device,
            "embedding_dim": embed_dim(train_embeddings),
        },
        "categorical_maps": {
            "condition": condition_map,
            "benchmark": benchmark_map,
        },
        "dataset_split": {
            "n_subjects": len(stage1_out["subject_ids"]),
            "n_train_items": len(stage1_out["item_ids"]),
            "n_eval_items": len(eval_item_ids),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
        "paths": {"stage1": "stage1.pt", "stage2a": "stage2a.pt"},
    }
    ckpt.save_manifest(manifest, run_dir / "manifest.json")

    if verbose:
        print(f"\n{SEP}\nTRAIN COMPLETE — checkpoint at:\n  {run_dir}\n{SEP}")
    return run_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_id_map(items_df: pd.DataFrame, col: str | None) -> dict[str, int]:
    """label -> int (codes start at 1; 0 reserved for DEFAULT_CAT_ID fallback)."""
    if col is None or col not in items_df.columns:
        return {}
    labels = sorted(items_df[col].dropna().astype(str).unique().tolist())
    return {label: i + 1 for i, label in enumerate(labels)}


def _vocab_size(id_map: dict[str, int]) -> int:
    return max(1, len(id_map))


def _encode_ids(
    items_df: pd.DataFrame,
    col: str | None,
    id_map: dict[str, int],
    item_ids: list[str],
) -> torch.Tensor:
    """Encode item_ids (in order) to category codes; missing/unseen -> 0."""
    n = len(item_ids)
    if col is None or col not in items_df.columns or not id_map:
        return torch.zeros(n, dtype=torch.long)
    # items_df is indexed by item_id (multi_benchmark guarantees this).
    labels = items_df[col].reindex([str(it) for it in item_ids])
    codes = [
        id_map.get(str(lbl), 0) if pd.notna(lbl) else 0
        for lbl in labels.tolist()
    ]
    return torch.tensor(codes, dtype=torch.long)


def _load_descriptions(
    descriptions_path: Path,
    enabled: bool,
    verbose: bool = True,
) -> dict[str, str]:
    """Load benchmark_name -> description JSON. Empty dict if disabled."""
    if not enabled:
        return {}
    if not Path(descriptions_path).exists():
        raise FileNotFoundError(
            f"INCLUDE_BENCHMARK_DESCRIPTION is set but descriptions file not "
            f"found at {descriptions_path}"
        )
    with Path(descriptions_path).open() as f:
        descriptions = json.load(f)
    if not isinstance(descriptions, dict):
        raise ValueError(
            f"Expected a JSON object mapping benchmark name -> description in "
            f"{descriptions_path}, got {type(descriptions).__name__}"
        )
    return {str(k): str(v) for k, v in descriptions.items()}


def _prepend_benchmark_descriptions(
    texts: list[str],
    *,
    item_ids: list[str],
    items_df: pd.DataFrame,
    benchmark_col: str | None,
    descriptions: dict[str, str],
    enabled: bool,
    sep: str,
    verbose: bool = True,
) -> list[str]:
    """Prepend each item's benchmark description to its text (in item_ids order).

    No-op when disabled or no descriptions are loaded. Items whose benchmark has
    no matching description are left unchanged.
    """
    if not enabled or not descriptions or not benchmark_col \
            or benchmark_col not in items_df.columns:
        return texts

    labels = items_df[benchmark_col].reindex([str(it) for it in item_ids]).tolist()
    out: list[str] = []
    n_prepended = 0
    n_missing = 0
    for text, lbl in zip(texts, labels):
        key = str(lbl) if pd.notna(lbl) else None
        desc = descriptions.get(key) if key is not None else None
        if desc:
            out.append(f"{desc}{sep}{text}")
            n_prepended += 1
        else:
            out.append(text)
            n_missing += 1
    if verbose:
        print(f"  prepended descriptions to {n_prepended}/{len(texts)} items "
              f"({n_missing} without a matching description)")
    return out


def _build_train_responses(
    rm,
    *,
    subject_ids: list[str],
    train_item_ids: list[str],
) -> torch.Tensor:
    """Slice the pooled response matrix to (subjects x train_items), NaN-padded."""
    full = torch.as_tensor(rm.data, dtype=torch.float32)

    s_pos = {s: i for i, s in enumerate(rm.subject_ids)}
    i_pos = {it: j for j, it in enumerate(rm.item_ids)}

    missing_subj = [s for s in subject_ids if s not in s_pos]
    missing_item = [it for it in train_item_ids if it not in i_pos]
    if missing_subj or missing_item:
        raise KeyError(
            f"Alignment mismatch: {len(missing_subj)} subject(s) and "
            f"{len(missing_item)} item(s) not in the pooled ResponseMatrix. "
            f"First missing subject: {missing_subj[:1]}, item: {missing_item[:1]}"
        )

    row_idx = torch.tensor([s_pos[s] for s in subject_ids], dtype=torch.long)
    col_idx = torch.tensor([i_pos[it] for it in train_item_ids], dtype=torch.long)
    return full.index_select(0, row_idx).index_select(1, col_idx)


if __name__ == "__main__":
    train()