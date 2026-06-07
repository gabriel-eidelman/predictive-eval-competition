"""Stage 3 of the PGE pipeline: score and evaluate on held-out eval items.

Accepts stage1_out and stage2a_out from either a live run or checkpoint.
Calls stage2a's predict callable to get item parameters, then scores all
(subject, eval-item) pairs.
"""

from __future__ import annotations

import torch
from torch_measure.metrics import brier_score, expected_calibration_error


def run_stage3(
    stage1_out: dict,
    stage2a_out: dict,
    eval_item_embeddings: torch.Tensor,
    eval_responses: torch.Tensor,
    eval_mask: torch.Tensor,
    *,
    verbose: bool = True,
) -> dict:
    """Predict P(correct) for all (subject, eval-item) pairs and evaluate.

    Parameters
    ----------
    stage1_out
        Output of run_stage1 (or checkpoint.load_stage1). Used for index
        alignment; subject abilities come from stage2a_out["ability"].
    stage2a_out
        Output of run_stage2a (or checkpoint.load_stage2a). Provides the
        EM-fitted `ability` matrix and the `predict` callable for items.
    eval_item_embeddings
        (n_eval_items, embed_dim) embeddings for the held-out items.
    eval_responses
        (n_subjects, n_eval_items) ground-truth responses (NaN = unobserved).
    eval_mask
        Boolean mask, True where eval_responses is observed.

    Returns
    -------
    dict with keys:
        probs       — (n_subjects, n_eval_items) predicted probabilities
        brier       — scalar Brier score on observed eval cells
        ece         — scalar ECE on observed eval cells
    """
    ability = stage2a_out["ability"]

    # Predict item parameters for eval items via the stage 2a regressor.
    pred_loadings, pred_difficulty = stage2a_out["predict"](eval_item_embeddings)
    # pred_loadings  : (n_eval_items, K)
    # pred_difficulty: (n_eval_items,)

    # align devices in case checkpoint and ability were loaded to different devices
    pred_loadings = pred_loadings.to(ability.device)
    pred_difficulty = pred_difficulty.to(ability.device)

    # logit_{ij} = u_i @ v_j + b_j  =>  (n_subjects, n_eval_items)
    logits = ability @ pred_loadings.T + pred_difficulty.unsqueeze(0)
    probs = torch.sigmoid(logits)

    bs = brier_score(probs, eval_responses, mask=eval_mask)
    ece = expected_calibration_error(probs, eval_responses, mask=eval_mask)

    if verbose:
        n_obs = int(eval_mask.sum())
        print("\nStage 3 evaluation")
        print(f"  Eval cells observed : {n_obs:,}")
        print(f"  Brier score         : {bs:.4f}")
        print(f"  ECE                 : {ece:.4f}")

    return {"probs": probs, "brier": bs, "ece": ece}