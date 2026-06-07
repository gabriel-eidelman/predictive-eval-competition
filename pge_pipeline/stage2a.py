"""Stage 2a of the PGE pipeline: joint EM training for amortized calibration.

Trains a text-to-item-params network and subject ability vectors jointly by
optimizing the Bernoulli log-likelihood of the response matrix.

model_type selects the regressor architecture:
  "mlp"    -> MLPItemParamNetwork (Linear->LayerNorm->SiLU base + residual path)
  "linear" -> ItemParamLinear (a single Linear(embed_dim -> n_factors+1))

Architecture:
    f_W : text_embedding -> [loadings (K); difficulty (1)]
    θ   : free (n_subjects, K) ability matrix

  MLP variant:
    Base     : Linear -> LayerNorm -> SiLU  (embed_dim -> hidden)
    Residual : n_residual_layers × (Linear+SiLU) + skip from base
    Head     : Linear(hidden -> n_factors + 1)
  Linear variant:
    Linear(embed_dim -> n_factors + 1)

EM loop: E-step is an implicit forward pass (item params = f_W(...)); M-step is a
joint Adam step on θ and W against the observed-cell Bernoulli NLL. Returns the
best-val-epoch checkpoint of net and ability.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Config defaults (all overridable via run_stage2a kwargs)
# ---------------------------------------------------------------------------

ALPHA = 1e-2
VAL_FRAC = 0.2
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_EPOCHS = 500
LR = 1e-2
ABILITY_LR_SCALE = 0.3
ABILITY_WEIGHT_DECAY = 1e-2
PATIENCE = 15
MIN_DELTA = 1e-4

# Architecture defaults
HIDDEN_DIM = 128
N_RESIDUAL_LAYERS = 1

MODEL_TYPE = "mlp"   # "mlp" | "linear"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _standardizer(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = x.mean(dim=0)
    std = x.std(dim=0).clamp_min(1e-6)
    return mean, std


def _train_val_item_split(
    n_items: int, val_frac: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split *items* (columns) into train/val for held-out item evaluation."""
    perm = torch.randperm(n_items, generator=torch.Generator().manual_seed(seed))
    n_val = max(1, int(val_frac * n_items))
    return perm[n_val:], perm[:n_val]


def _bernoulli_nll(
    logits: torch.Tensor,
    responses: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Mean negative Bernoulli log-likelihood over observed cells."""
    return F.binary_cross_entropy_with_logits(
        logits[mask], responses[mask], reduction="mean"
    )


# ---------------------------------------------------------------------------
# Item-parameter networks
# ---------------------------------------------------------------------------


class MLPItemParamNetwork(torch.nn.Module):
    """Maps standardized_embedding -> [loadings; difficulty].

    Linear->LayerNorm->SiLU base with a residual MLP path and a linear output head.
    """

    def __init__(
        self,
        embed_dim: int,
        n_factors: int,
        *,
        hidden_dim: int = HIDDEN_DIM,
        n_residual_layers: int = N_RESIDUAL_LAYERS,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_factors = n_factors
        self.hidden_dim = hidden_dim
        self.n_residual_layers = n_residual_layers

        # --- Base content network ---
        self.base_linear = torch.nn.Linear(embed_dim, hidden_dim)
        self.base_norm = torch.nn.LayerNorm(hidden_dim)
        self.base_act = torch.nn.SiLU()

        # --- Residual path ---
        self.residual_layers = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(n_residual_layers)]
        )
        self.residual_act = torch.nn.SiLU()

        # --- Output head ---
        self.output_head = torch.nn.Linear(hidden_dim, n_factors + 1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Small init keeps logits near 0 (prob 0.5) at the start of EM."""
        torch.nn.init.normal_(self.base_linear.weight, std=0.01)
        torch.nn.init.zeros_(self.base_linear.bias)

        for layer in self.residual_layers:
            torch.nn.init.normal_(layer.weight, std=0.01)
            torch.nn.init.zeros_(layer.bias)

        torch.nn.init.normal_(self.output_head.weight, std=0.01)
        torch.nn.init.zeros_(self.output_head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (batch, embed_dim) -> (batch, n_factors + 1)."""
        base = self.base_act(self.base_norm(self.base_linear(x)))  # (B, H)

        h = base
        for layer in self.residual_layers:
            h = self.residual_act(layer(h))
        h = h + base                           # residual skip-connection

        return self.output_head(h)             # (B, n_factors + 1)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def get_save_dict(self) -> dict:
        return {
            "model_type": "mlp",
            "state_dict": self.state_dict(),
            "embed_dim": self.embed_dim,
            "n_factors": self.n_factors,
            "hidden_dim": self.hidden_dim,
            "n_residual_layers": self.n_residual_layers,
        }

    @classmethod
    def from_save_dict(cls, d: dict) -> "MLPItemParamNetwork":
        net = cls(
            d["embed_dim"],
            d["n_factors"],
            hidden_dim=d["hidden_dim"],
            n_residual_layers=d["n_residual_layers"],
        )
        net.load_state_dict(d["state_dict"])
        return net

    def eval_predict(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (loadings, difficulty) without gradient tracking."""
        with torch.no_grad():
            out = self(x)
        return out[:, : self.n_factors], out[:, self.n_factors]


class ItemParamLinear(torch.nn.Module):
    """Differentiable linear map: standardized_embedding -> [loadings; difficulty].

    Implemented as a torch.nn.Linear so gradients flow through W naturally during
    the joint EM optimisation loop.

    Input  : (batch, embed_dim)      — L2-normalised, then z-scored
    Output : (batch, n_factors + 1)  — last column is difficulty b_j
    """

    def __init__(self, embed_dim: int, n_factors: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_factors = n_factors
        self.linear = torch.nn.Linear(embed_dim, n_factors + 1)
        # Small init keeps logits well-conditioned at the start.
        torch.nn.init.normal_(self.linear.weight, std=0.01)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (batch, embed_dim) -> (batch, n_factors + 1)."""
        return self.linear(x)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def get_save_dict(self) -> dict:
        return {
            "model_type": "linear",
            "state_dict": self.state_dict(),
            "embed_dim": self.embed_dim,
            "n_factors": self.n_factors,
        }

    @classmethod
    def from_save_dict(cls, d: dict) -> "ItemParamLinear":
        net = cls(d["embed_dim"], d["n_factors"])
        net.load_state_dict(d["state_dict"])
        return net

    def eval_predict(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (loadings, difficulty) without gradient tracking."""
        with torch.no_grad():
            out = self(x)
        return out[:, : self.n_factors], out[:, self.n_factors]


def _build_net(
    model_type: str,
    embed_dim: int,
    n_factors: int,
    *,
    hidden_dim: int,
    n_residual_layers: int,
) -> torch.nn.Module:
    """Construct the requested item-param network."""
    if model_type == "mlp":
        return MLPItemParamNetwork(
            embed_dim,
            n_factors,
            hidden_dim=hidden_dim,
            n_residual_layers=n_residual_layers,
        )
    if model_type == "linear":
        return ItemParamLinear(embed_dim, n_factors)
    raise ValueError(
        f"Unknown model_type {model_type!r}; expected 'mlp' or 'linear'."
    )


def net_from_save_dict(d: dict) -> torch.nn.Module:
    """Reconstruct an item-param network from its save dict (dispatch on tag)."""
    model_type = d.get("model_type", "mlp")
    if model_type == "mlp":
        return MLPItemParamNetwork.from_save_dict(d)
    if model_type == "linear":
        return ItemParamLinear.from_save_dict(d)
    raise ValueError(
        f"Unknown model_type {model_type!r} in save dict; expected 'mlp' or 'linear'."
    )


# ---------------------------------------------------------------------------
# Joint EM training loop
# ---------------------------------------------------------------------------


def _compute_logits(
    ability: torch.Tensor,
    loadings: torch.Tensor,
    difficulty: torch.Tensor,
) -> torch.Tensor:
    """IRT logit: θ_i · v_j + b_j  →  (n_subjects, n_items)."""
    return ability @ loadings.T + difficulty.unsqueeze(0)


def _run_em(
    responses: torch.Tensor,
    item_embeddings: torch.Tensor,
    val_responses: torch.Tensor | None,
    val_embeddings: torch.Tensor | None,
    n_factors: int,
    *,
    model_type: str,
    alpha: float,
    max_epochs: int,
    lr: float,
    ability_lr_scale: float,
    ability_weight_decay: float,
    patience: int,
    min_delta: float,
    seed: int,
    device: str,
    verbose: bool,
    hidden_dim: int,
    n_residual_layers: int,
) -> tuple[torch.nn.Module, torch.Tensor, list[float], list[float]]:
    """Core joint EM loop. Returns the BEST-VAL-EPOCH net + ability."""
    torch.manual_seed(seed)
    n_subjects, n_items_train = responses.shape
    embed_dim = item_embeddings.shape[1]

    responses = responses.to(device)
    item_embeddings = item_embeddings.to(device)
    train_mask = ~torch.isnan(responses)

    if val_responses is not None:
        val_responses = val_responses.to(device)
        val_embeddings = val_embeddings.to(device)
        val_mask = ~torch.isnan(val_responses)
    else:
        val_mask = None

    # ----- Parameters -----
    ability = torch.nn.Parameter(
        torch.randn(n_subjects, n_factors, device=device) * 0.1
    )
    net = _build_net(
        model_type,
        embed_dim,
        n_factors,
        hidden_dim=hidden_dim,
        n_residual_layers=n_residual_layers,
    ).to(device)

    # ----- Optimiser -----
    optimizer = torch.optim.Adam(
        [
            {
                "params": [ability],
                "lr": lr * ability_lr_scale,
                "weight_decay": ability_weight_decay,
            },
            {"params": net.parameters(), "lr": lr, "weight_decay": alpha},
        ]
    )

    train_hist: list[float] = []
    val_hist: list[float] = []
    best_val = math.inf
    best_epoch = 0
    no_improve = 0
    best_net_state: dict | None = None
    best_ability: torch.Tensor | None = None

    if verbose:
        print(
            f"\nJoint EM — model_type={model_type}, n_subjects={n_subjects}, "
            f"n_items_train={n_items_train}, embed_dim={embed_dim}, K={n_factors}, "
            f"hidden={hidden_dim}, n_resid={n_residual_layers}, "
            f"alpha={alpha}, ability_wd={ability_weight_decay}, "
            f"ability_lr_scale={ability_lr_scale}, lr={lr}, "
            f"patience={patience}, max_epochs={max_epochs}"
        )

    for epoch in range(1, max_epochs + 1):
        net.train()
        optimizer.zero_grad()

        item_params = net(item_embeddings)
        loadings = item_params[:, : n_factors]
        difficulty = item_params[:, n_factors]
        logits = _compute_logits(ability, loadings, difficulty)
        loss = _bernoulli_nll(logits, responses, train_mask)

        loss.backward()
        optimizer.step()

        train_nll = loss.item()
        train_hist.append(train_nll)

        val_nll: float | None = None
        if val_responses is not None:
            with torch.no_grad():
                net.eval()
                val_params = net(val_embeddings)
                val_logits = _compute_logits(
                    ability, val_params[:, :n_factors], val_params[:, n_factors]
                )
                val_nll = _bernoulli_nll(val_logits, val_responses, val_mask).item()
            val_hist.append(val_nll)

            if val_nll < best_val - min_delta:
                best_val = val_nll
                best_epoch = epoch
                no_improve = 0
                best_net_state = copy.deepcopy(
                    {k: v.detach().cpu() for k, v in net.state_dict().items()}
                )
                best_ability = ability.detach().cpu().clone()
            else:
                no_improve += 1
                if no_improve >= patience:
                    if verbose:
                        print(f"  Early stop at epoch {epoch} (patience={patience}); "
                              f"best val NLL {best_val:.4f} @ epoch {best_epoch}")
                    break

        if verbose and (epoch == 1 or epoch % 25 == 0):
            val_str = f"  val NLL: {val_nll:.4f}" if val_nll is not None else ""
            print(f"  epoch {epoch:4d}  train NLL: {train_nll:.4f}{val_str}")

    if verbose:
        print("Joint EM done.")

    if best_net_state is not None:
        net.load_state_dict(best_net_state)
        ability_out = best_ability
        if verbose:
            print(f"  Restored best-val model from epoch {best_epoch} "
                  f"(val NLL {best_val:.4f}).")
    else:
        ability_out = ability.detach().cpu()

    return net.cpu(), ability_out, train_hist, val_hist


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_stage2a(
    stage1_out: dict,
    item_embeddings: torch.Tensor,
    *,
    model_type: str = MODEL_TYPE,
    alpha: float = ALPHA,
    val_frac: float = VAL_FRAC,
    seed: int = SEED,
    device: str = DEVICE,
    verbose: bool = True,
    max_epochs: int = MAX_EPOCHS,
    lr: float = LR,
    ability_lr_scale: float = ABILITY_LR_SCALE,
    ability_weight_decay: float = ABILITY_WEIGHT_DECAY,
    patience: int = PATIENCE,
    hidden_dim: int = HIDDEN_DIM,
    n_residual_layers: int = N_RESIDUAL_LAYERS,
    weight_decay: float = 0.0,   # ignored — alpha is the L2 knob here
) -> dict:
    """Fit via joint EM (Amortized Calibration).

    model_type selects the regressor: "mlp" or "linear".
    Returns a dict with model, ability (best-val snapshot), predict callable,
    and normalization stats.
    """
    torch.manual_seed(seed)

    # --- Resolve the response matrix ---
    if "responses" in stage1_out:
        responses: torch.Tensor = stage1_out["responses"].float()
    elif hasattr(stage1_out.get("model", None), "responses"):
        responses = stage1_out["model"].responses.float()
    else:
        raise ValueError(
            "stage1_out must contain a 'responses' key (n_subjects x n_items "
            "tensor with NaN for unobserved cells)."
        )

    item_ids: list[str] = stage1_out["item_ids"]
    n_items = len(item_ids)

    if item_embeddings.shape[0] != n_items:
        raise ValueError(
            f"item_embeddings has {item_embeddings.shape[0]} rows but "
            f"stage1_out['item_ids'] has {n_items} entries; must be row-aligned."
        )

    # --- Standardise embeddings (fit stats on train split only) ---
    train_idx, val_idx = _train_val_item_split(n_items, val_frac, seed)
    feat_mean, feat_std = _standardizer(item_embeddings[train_idx])

    def _norm(x: torch.Tensor) -> torch.Tensor:
        return (x - feat_mean.to(x.device)) / feat_std.to(x.device)

    emb_norm = _norm(item_embeddings)

    # --- Split response matrix by train/val items ---
    train_responses = responses[:, train_idx]
    val_responses = responses[:, val_idx] if len(val_idx) > 0 else None
    train_emb = emb_norm[train_idx]
    val_emb = emb_norm[val_idx] if len(val_idx) > 0 else None

    # Infer n_factors.
    n_factors: int = 1
    if "model" in stage1_out and hasattr(stage1_out["model"], "loadings"):
        n_factors = stage1_out["model"].loadings.shape[1]
    elif "n_factors" in stage1_out:
        n_factors = int(stage1_out["n_factors"])

    embed_dim = item_embeddings.shape[1]

    if verbose:
        obs = (~torch.isnan(responses)).sum().item()
        print(
            f"\nStage 2a (Amortized Calibration / Joint EM)\n"
            f"  model_type={model_type}\n"
            f"  {responses.shape[0]} subjects × {n_items} items  "
            f"({obs:,} observed cells)\n"
            f"  train items: {len(train_idx)}  val items: {len(val_idx)}\n"
            f"  K={n_factors}  embed_dim={embed_dim}"
        )

    net, ability, train_hist, val_hist = _run_em(
        train_responses,
        train_emb,
        val_responses,
        val_emb,
        n_factors,
        model_type=model_type,
        alpha=alpha,
        max_epochs=max_epochs,
        lr=lr,
        ability_lr_scale=ability_lr_scale,
        ability_weight_decay=ability_weight_decay,
        patience=patience,
        min_delta=MIN_DELTA,
        seed=seed,
        device=device,
        verbose=verbose,
        hidden_dim=hidden_dim,
        n_residual_layers=n_residual_layers,
    )

    net.eval()

    feat_mean_cpu = feat_mean.cpu()
    feat_std_cpu = feat_std.cpu()

    def predict(
        new_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict (loadings, difficulty) for new item embeddings.

        new_embeddings : (m, embed_dim) — raw (L2-normalised, NOT z-scored).
        """
        with torch.no_grad():
            new_embeddings = new_embeddings.cpu().float()
            normed = (new_embeddings - feat_mean_cpu) / feat_std_cpu
            return net.eval_predict(normed)

    return {
        "model":      net,
        "model_type": model_type,
        "predict":    predict,
        "feat_mean":  feat_mean_cpu,
        "feat_std":   feat_std_cpu,
        "history":    {"train_loss": train_hist, "val_loss": val_hist},
        "n_factors":  n_factors,
        "embed_dim":  embed_dim,
        "ability":    ability,
    }