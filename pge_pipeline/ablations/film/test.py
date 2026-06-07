"""test.py — PGE test phase, FiLM ablation.

Loads a checkpoint from train.py, rebuilds the pooled benchmark data
deterministically, encodes eval items' (benchmark, condition) ids from the
manifest's global maps, and runs stage 3.
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

from src.pge.stage3 import run_stage3
from src.pge import checkpoint as ckpt
from src.pge.pipeline_common import embed_texts, get_texts, build_eval_tensor
from src.pge.multi_benchmark import load_multi_benchmark, BENCHMARK_NAMES

CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

DESCRIPTIONS_PATH = Path(__file__).resolve().parent / "dataset_descriptions.json"


def _latest_run_dir(ckpt_dir: Path) -> Path:
    candidates = [p for p in ckpt_dir.iterdir() if (p / "manifest.json").exists()]
    if not candidates:
        raise FileNotFoundError(f"No checkpoints with a manifest under {ckpt_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def test(
    *,
    run_dir: Path | None = None,
    latest_in: Path | None = None,
    descriptions_path: Path = DESCRIPTIONS_PATH,
    device: str = "cpu",
    results_dir: Path = RESULTS_DIR,
    verbose: bool = True,
) -> dict:
    """Load a checkpoint and evaluate stage 3 on the held-out eval items."""
    SEP = "=" * 60

    if run_dir is None:
        run_dir = _latest_run_dir(latest_in or CKPT_DIR)
    if verbose:
        print(f"\n{SEP}\nTEST: using checkpoint\n  {run_dir}\n{SEP}")

    # --- 0. Load manifest + models ---
    manifest = ckpt.load_manifest(run_dir / "manifest.json")
    cfg = manifest["config"]
    eval_item_ids: list[str] = manifest["eval_item_ids"]

    cat_maps = manifest.get("categorical_maps", {})
    condition_map = cat_maps.get("condition", {})
    benchmark_map = cat_maps.get("benchmark", {})
    condition_col = cfg.get("condition_col")
    benchmark_col = cfg.get("benchmark_col")

    # Description ablation: reuse EXACTLY what training used.
    include_benchmark_description = bool(cfg.get("include_benchmark_description", False))
    description_sep = cfg.get("description_sep", "\n\n")

    benchmark_names = manifest.get("benchmark_names", BENCHMARK_NAMES)
    max_items = cfg.get("max_items_per_benchmark")
    seed = cfg.get("seed", 42)

    stage1_out = ckpt.load_stage1(run_dir / "stage1.pt", device=device)
    stage2a_out = ckpt.load_stage2a(run_dir / "stage2a.pt", device=device)

    # --- 1. Rebuild the pooled data (deterministic) + eval matrix ---
    if verbose:
        print(f"\n{SEP}\nTEST STEP 1: rebuild pooled data, build eval matrix\n{SEP}")
    mb = load_multi_benchmark(
        benchmark_names=benchmark_names,
        max_items_per_benchmark=max_items,
        seed=seed,
        verbose=verbose,
    )
    items_df: pd.DataFrame = mb.items
    rm = mb.rm

    # Sanity: every manifest eval id should be present in the rebuilt pool.
    missing = [it for it in eval_item_ids if it not in set(rm.item_ids)]
    if missing:
        raise RuntimeError(
            f"{len(missing)} eval item(s) from the manifest are absent from the "
            f"rebuilt pool (e.g. {missing[:3]}). The benchmark data or seed may "
            f"have changed since training; results would be inconsistent."
        )

    eval_responses, eval_mask = build_eval_tensor(
        rm, subject_ids=stage1_out["subject_ids"], eval_item_ids=eval_item_ids
    )
    if verbose:
        n_obs = int(eval_mask.sum())
        print(f"  Eval matrix {tuple(eval_responses.shape)} | {n_obs:,} observed "
              f"({eval_mask.float().mean():.1%})")

    # --- 1b. Load descriptions iff training used them ---
    descriptions = _load_descriptions(
        descriptions_path, include_benchmark_description, verbose
    )
    if verbose:
        print(f"  benchmark description prepend: {include_benchmark_description} "
              f"({len(descriptions)} descriptions loaded)")

    # --- 2. Embed eval items ---
    if verbose:
        print(f"\n{SEP}\nTEST STEP 2: embed eval items\n{SEP}")
    eval_texts = get_texts(items_df, eval_item_ids)
    eval_texts = _prepend_benchmark_descriptions(
        eval_texts,
        item_ids=eval_item_ids,
        items_df=items_df,
        benchmark_col=benchmark_col,
        descriptions=descriptions,
        enabled=include_benchmark_description,
        sep=description_sep,
        verbose=verbose,
    )
    eval_embeddings = embed_texts(
        eval_texts, cfg["embed_model"], cfg["embed_batch_size"], device, verbose
    )

    # --- 2b. Encode eval-item categorical ids with the GLOBAL train maps ---
    eval_condition_ids = _encode_ids(items_df, condition_col, condition_map, eval_item_ids)
    eval_benchmark_ids = _encode_ids(items_df, benchmark_col, benchmark_map, eval_item_ids)
    if verbose:
        n_cond_fb = int((eval_condition_ids == 0).sum())
        n_bench_fb = int((eval_benchmark_ids == 0).sum())
        print(f"  eval ids encoded | condition fallbacks: {n_cond_fb} | "
              f"benchmark fallbacks: {n_bench_fb}")

    # --- 3. Stage 3 ---
    if verbose:
        print(f"\n{SEP}\nTEST STEP 3: stage 3 scoring\n{SEP}")
    stage3_out = run_stage3(
        stage1_out,
        stage2a_out,
        eval_embeddings,
        eval_responses,
        eval_mask,
        condition_ids=eval_condition_ids,
        benchmark_ids=eval_benchmark_ids,
        verbose=verbose,
    )

    # --- 4. Save results ---
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc)
    results_path = results_dir / f"{manifest['run_id']}_eval_{ts.strftime('%Y%m%dT%H%M%SZ')}.json"
    results = {
        "run_id": manifest["run_id"],
        "checkpoint_dir": str(run_dir),
        "timestamp_utc": ts.isoformat(),
        "config": cfg,
        "dataset_split": {
            **manifest["dataset_split"],
            "n_observed_eval_responses": int(eval_mask.sum().item()),
            "eval_density": float(eval_mask.float().mean().item()),
        },
        "results": {
            "brier": float(stage3_out["brier"]),
            "ece": float(stage3_out["ece"]),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
    }
    with results_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)

    if verbose:
        print(f"\n{SEP}\nTEST COMPLETE")
        print(f"  Brier {stage3_out['brier']:.4f} | ECE {stage3_out['ece']:.4f}")
        print(f"  Results -> {results_path}\n{SEP}")

    return {**stage3_out, "results_path": results_path}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _encode_ids(items_df, col, id_map, item_ids) -> torch.Tensor:
    """Encode item_ids to category codes using the global train map; unseen -> 0."""
    n = len(item_ids)
    if col is None or col not in items_df.columns or not id_map:
        return torch.zeros(n, dtype=torch.long)
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
    """Load benchmark_name -> description JSON. Returns empty dict if disabled."""
    if not enabled:
        return {}
    if not Path(descriptions_path).exists():
        raise FileNotFoundError(
            f"Manifest records include_benchmark_description=True but the "
            f"descriptions file was not found at {descriptions_path}. Point "
            f"--descriptions_path at the same file used for training."
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

    No-op when disabled or no descriptions are loaded.
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


if __name__ == "__main__":
    test()