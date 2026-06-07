"""Shared helpers for the PGE train/test split.
"""

from __future__ import annotations

import torch
from sentence_transformers import SentenceTransformer


def split_item_ids(
    item_ids: list[str], eval_frac: float, seed: int
) -> tuple[list[str], list[str]]:
    """Randomly split item_ids into (train_item_ids, eval_item_ids).

    Deterministic given (item_ids order, eval_frac, seed), so the test phase
    can recompute the same split — though we also persist the lists in the
    manifest to avoid relying on that.
    """
    n = len(item_ids)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    n_eval = max(1, int(eval_frac * n))
    eval_idx = set(perm[:n_eval])
    train_ids = [item_ids[i] for i in range(n) if i not in eval_idx]
    eval_ids = [item_ids[i] for i in range(n) if i in eval_idx]
    return train_ids, eval_ids


def embed_texts(
    texts: list[str],
    model_name: str,
    batch_size: int,
    device: str = "cpu",
    verbose: bool = True,
) -> torch.Tensor:
    """Encode a list of strings with a sentence-transformer.

    Returns a (n, embed_dim) float32 tensor row-aligned with `texts`.
    """
    if verbose:
        print(f"Embedding {len(texts)} texts with '{model_name}'...")
    encoder = SentenceTransformer(model_name, device=device)
    vecs = encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=verbose,
        convert_to_numpy=True,
    )
    return torch.tensor(vecs, dtype=torch.float32)


def build_eval_tensor(
    response_matrix,
    subject_ids: list[str],
    eval_item_ids: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract the (all subjects) x (eval items) response sub-matrix.

    Returns (responses, eval_mask): responses has NaN where unobserved,
    eval_mask is True where a response exists.
    """
    all_item_ids: list[str] = response_matrix.item_ids
    item_id_to_col = {iid: j for j, iid in enumerate(all_item_ids)}
    subj_id_to_row = {sid: i for i, sid in enumerate(response_matrix.subject_ids)}

    n_subj = len(subject_ids)
    n_eval = len(eval_item_ids)
    responses = torch.full((n_subj, n_eval), float("nan"))

    for j_eval, iid in enumerate(eval_item_ids):
        j_full = item_id_to_col.get(iid)
        if j_full is None:
            continue
        for i_subj, sid in enumerate(subject_ids):
            i_full = subj_id_to_row.get(sid)
            if i_full is None:
                continue
            responses[i_subj, j_eval] = response_matrix.data[i_full, j_full]

    eval_mask = ~torch.isnan(responses)
    return responses, eval_mask


def get_texts(items_df, ids: list[str]) -> list[str]:
    """Look up raw item text by id from the items registry DataFrame."""
    id_to_text = dict(zip(items_df["item_id"], items_df["content"]))
    return [id_to_text.get(iid, "") for iid in ids]


def embed_dim(*tensors: torch.Tensor) -> int | None:
    """Return embedding dim from the first non-empty 2D tensor, else None."""
    for t in tensors:
        if t is not None and t.ndim == 2 and t.shape[0] > 0:
            return int(t.shape[1])
    return None
