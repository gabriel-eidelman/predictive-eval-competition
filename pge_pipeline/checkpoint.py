"""Checkpointing for the PGE pipeline.

Saves and loads the train-phase outputs (stage 1 + stage 2a) so stage 3 can
score without re-fitting.

* Stage 1  : fitted LogisticFM + subject_ids + item_ids (for index alignment).
* Stage 2a : item-param regressor (via get_save_dict) + normalization stats
             (feat_mean / feat_std) + EM-fitted ability matrix. The `predict`
             closure is rebuilt at load time from the saved regressor.
* Manifest : train/eval item id lists, embed model name, and run config.

Two files per run:
    <ckpt_dir>/<run_id>/stage1.pt
    <ckpt_dir>/<run_id>/stage2a.pt
    <ckpt_dir>/<run_id>/manifest.json

The regressor save dict carries a `model_type` tag plus constructor dimensions,
so net_from_save_dict reconstructs the right class. feat_std is clamped to >= 1e-3
on save.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.pge.stage2a import net_from_save_dict


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def save_stage1(stage1_out: dict, path: Path) -> None:
    """Save the fitted LogisticFM and its id alignment.

    Saves the full module (not just state_dict) so constructor dimensions are
    pickled alongside weights. The raw responses tensor is not persisted.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": stage1_out["model"],
        "subject_ids": stage1_out["subject_ids"],
        "item_ids": stage1_out["item_ids"],
    }
    torch.save(payload, path)


def save_stage2a(stage2a_out: dict, path: Path) -> None:
    """Save the item-param regressor, normalization stats, and abilities.

    The `predict` closure is intentionally dropped — it's rebuilt on load from
    the regressor + feat_mean/feat_std. The EM-fitted ability matrix is added
    so stage 3 can score eval items.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    net: torch.nn.Module = stage2a_out["model"]
    payload = {
        **net.get_save_dict(),
        "feat_mean": stage2a_out["feat_mean"].cpu(),
        "feat_std": stage2a_out["feat_std"].cpu().clamp(min=1e-3),
        "ability": stage2a_out["ability"].cpu(),  # (n_subjects, K)
        "history": stage2a_out.get("history"),
    }
    torch.save(payload, path)


def save_manifest(manifest: dict, path: Path) -> None:
    """Write the human-readable split + config manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(manifest, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_stage1(path: Path, device: str = "cpu") -> dict:
    """Load the stage-1 payload into a stage3-compatible dict.

    Returns a dict shaped like run_stage1's output but containing only what
    stage 3 needs for index alignment: the model, subject_ids, item_ids.
    """
    payload = torch.load(path, map_location=device, weights_only=False)
    payload["model"].to(device)
    payload["model"].eval()
    return payload


def load_stage2a(path: Path, device: str = "cpu") -> dict:
    """Load the stage-2a payload and rebuild a `predict` callable.

    Reconstructs the regressor (MLP or linear, dispatched on the saved
    `model_type` tag) from get_save_dict() output, moves the normalization stats
    and ability matrix onto `device`, and closes over them in a `predict`
    function identical in behavior to run_stage2a's: it returns (loadings,
    difficulty) for raw (L2-normalised, NOT z-scored) embeddings.
    """
    payload = torch.load(path, map_location=device, weights_only=False)

    net = net_from_save_dict(payload).to(device)
    net.eval()

    feat_mean = payload["feat_mean"].to(device)
    feat_std = payload["feat_std"].to(device)
    ability = payload["ability"].to(device)

    def predict(
        new_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict (loadings, difficulty) for new item embeddings.

        new_embeddings : (m, embed_dim) — raw (L2-normalised, NOT z-scored).
        """
        with torch.no_grad():
            new_embeddings = new_embeddings.to(device).float()
            normed = (new_embeddings - feat_mean) / feat_std
            return net.eval_predict(normed)

    return {
        "model": net,
        "model_type": payload.get("model_type", "mlp"),
        "predict": predict,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "ability": ability,
        "history": payload.get("history"),
        "n_factors": payload["n_factors"],
        "embed_dim": payload["embed_dim"],
    }


def load_manifest(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)