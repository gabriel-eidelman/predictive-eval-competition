"""Stage 1 of the PGE pipeline: fit a LogisticFM factor model to observed data.

Loads a benchmark dataset (or accepts a pre-built ResponseMatrix), pivots to a
response matrix, accepts a pre-determined item split (train_item_ids) so the
factor model only sees training items. Returns fitted parameters plus the
item_ids it was trained on so downstream stages stay aligned.

MULTI-BENCHMARK: when `response_matrix` is supplied, the by-name load is
skipped and the given matrix is used directly. This lets train.py pass a pooled
matrix built across many benchmarks. The matrix only needs `.data`,
`.subject_ids`, `.item_ids`; an `observed_mask` is derived from NaNs if the
attribute is absent.
"""

from __future__ import annotations

import torch

from torch_measure.datasets import load
from torch_measure.metrics import brier_score, expected_calibration_error
from torch_measure.models import LogisticFM, predict_dense

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATASET = "afrimedqa"
N_FACTORS = 1
MAX_EPOCHS = 150
LR = 0.05
SEED = 42
DEVICE = "cpu"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def train_test_split_mask(
    observed: torch.Tensor,
    train_frac: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split observed cells into train and test boolean masks (cell-level)."""
    obs_idx = observed.nonzero(as_tuple=False)
    n_obs = obs_idx.shape[0]
    perm = torch.randperm(n_obs, generator=torch.Generator().manual_seed(seed))
    n_train = int(train_frac * n_obs)

    train_mask = torch.zeros_like(observed)
    test_mask = torch.zeros_like(observed)
    train_idx = obs_idx[perm[:n_train]]
    test_idx = obs_idx[perm[n_train:]]
    train_mask[train_idx[:, 0], train_idx[:, 1]] = True
    test_mask[test_idx[:, 0], test_idx[:, 1]] = True
    return train_mask, test_mask


def _observed_mask(rm) -> torch.Tensor:
    """Observed-cell mask for a ResponseMatrix.

    Uses rm.observed_mask if present; otherwise derives it from non-NaN cells,
    so a matrix built outside the standard constructor (e.g. the pooled
    multi-benchmark matrix) works without that attribute.
    """
    if hasattr(rm, "observed_mask") and rm.observed_mask is not None:
        m = rm.observed_mask
        return m if isinstance(m, torch.Tensor) else torch.as_tensor(m)
    data = rm.data if isinstance(rm.data, torch.Tensor) else torch.as_tensor(rm.data)
    return ~torch.isnan(data)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_stage1(
    train_item_ids: list[str] | None = None,
    *,
    response_matrix=None,
    dataset: str = DATASET,
    n_factors: int = N_FACTORS,
    max_epochs: int = MAX_EPOCHS,
    lr: float = LR,
    seed: int = SEED,
    device: str = DEVICE,
    verbose: bool = True,
) -> dict:
    """Fit LogisticFM on the training items and return a results dict.

    Parameters
    ----------
    train_item_ids
        If provided, the factor model is fit only on these items (columns).
        The pipeline passes this to ensure stage 1 never sees eval items.
        If None, all items are used (standalone / debug mode).
    response_matrix
        Pre-built ResponseMatrix (or compatible object exposing `.data`,
        `.subject_ids`, `.item_ids`). When supplied, the by-name `load(dataset)`
        is skipped and this matrix is used directly — the multi-benchmark path.
    dataset, n_factors, max_epochs, lr, seed, device, verbose
        Model and training hyperparameters. `dataset` is ignored when
        `response_matrix` is given.

    Returns
    -------
    dict with keys:
        model, responses, train_mask, test_mask, history, probs,
        subject_ids, item_ids
    """
    torch.manual_seed(seed)

    # --- Acquire the response matrix (pooled or by-name) ---
    if response_matrix is not None:
        rm = response_matrix
        if verbose:
            print(f"Using pre-built response matrix "
                  f"({len(rm.subject_ids)} subjects x {len(rm.item_ids)} items)")
    else:
        if verbose:
            print(f"Loading '{dataset}'...")
        data = load(dataset)
        rm = data.to_response_matrix()
        if verbose:
            print(f"  {rm}")

    # Ensure rm.data is a tensor for the rest of the function.
    rm_data = rm.data if isinstance(rm.data, torch.Tensor) else torch.as_tensor(rm.data, dtype=torch.float32)

    full_observed = _observed_mask(rm)
    if verbose:
        print(f"  Overall accuracy: {rm_data[full_observed].mean():.3f}")

    # --- Resolve item subset ---
    all_item_ids: list[str] = list(rm.item_ids)
    all_subject_ids: list[str] = list(rm.subject_ids)

    if train_item_ids is not None:
        train_item_set = set(train_item_ids)
        col_idx = [
            i for i, iid in enumerate(all_item_ids) if iid in train_item_set
        ]
        if len(col_idx) == 0:
            raise ValueError("None of the provided train_item_ids were found in the dataset.")
        responses = rm_data[:, torch.tensor(col_idx, dtype=torch.long)]
        item_ids = [all_item_ids[i] for i in col_idx]
        if verbose:
            print(f"\nRestricted to {len(col_idx)} training items")
    else:
        responses = rm_data
        item_ids = list(all_item_ids)

    subject_ids = list(all_subject_ids)

    observed = ~torch.isnan(responses)
    n_obs = int(observed.sum().item())
    if verbose:
        print(
            f"Matrix: {responses.shape[0]} subjects x {responses.shape[1]} items  "
            f"({n_obs:,} observed, density {observed.float().mean():.1%})"
        )

    # Cell-level train/test split (within training items only, for diagnostics)
    train_mask, test_mask = train_test_split_mask(observed, train_frac=0.8, seed=seed)
    if verbose:
        print(f"Cell split: {train_mask.sum().item():,} train  |  {test_mask.sum().item():,} test")

    # --- Fit ---
    n_subjects, n_items = responses.shape
    if verbose:
        print(f"\nFitting LogisticFM  (K={n_factors}, epochs={max_epochs}, lr={lr})...")
    model = LogisticFM(n_subjects, n_items, n_factors=n_factors, device=device)
    history = model.fit(
        responses,
        mask=train_mask,
        max_epochs=max_epochs,
        lr=lr,
        verbose=verbose,
    )
    if verbose:
        print(f"Final training loss: {history['losses'][-1]:.4f}")

    # --- Evaluate (cell-level held-out split within training items) ---
    with torch.no_grad():
        probs = predict_dense(model)

    if verbose:
        train_brier = brier_score(probs, responses, mask=train_mask)
        train_ece   = expected_calibration_error(probs, responses, mask=train_mask)
        test_brier  = brier_score(probs, responses, mask=test_mask)
        test_ece    = expected_calibration_error(probs, responses, mask=test_mask)
        print(f"\n{'Split':<8}  {'Brier':>8}  {'ECE':>8}")
        print("-" * 30)
        print(f"{'Train':<8}  {train_brier:>8.4f}  {train_ece:>8.4f}")
        print(f"{'Test':<8}  {test_brier:>8.4f}  {test_ece:>8.4f}")

    return {
        "model":        model,
        "responses":    responses,
        "train_mask":   train_mask,
        "test_mask":    test_mask,
        "history":      history,
        "probs":        probs,
        "subject_ids":  subject_ids,
        "item_ids":     item_ids,
    }


if __name__ == "__main__":
    run_stage1()