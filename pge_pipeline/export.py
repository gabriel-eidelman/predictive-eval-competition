"""export.py — offline artifact builder for the Predictive AI Evaluation Challenge.

Run with:  modal run export.py

  1. Loads every benchmark's response matrix and stitches them into one global
     (all_subjects x all_items) matrix.
  2. [Optional] Fits Stage 1 (LogisticFM) to infer n_factors.
  3. Embeds every item's text with the sentence embedder declared in models.txt.
  4. Fits Stage 2a via joint EM: linear text->item-params regressor and subject
     ability matrix trained together on Bernoulli log-likelihood.
  5. Bundles everything predict() needs into a zip.

Bundle contents (inside submission_artifacts.zip):
    abilities.pt        - (n_subjects, K) float tensor — from EM joint fit
    subject_lookup.json - {normalized_display_name: row_index, ...} + subject_ids
    stage2a.pt          - linear regressor state_dict + feat_mean/std + dims
    config.json         - embed model id, n_factors, embed_dim, max_seq_len, etc.

IMPORTANT: EMBED_MODEL here MUST equal the single line in models.txt and the
EMBED_MODEL constant in model.py.
"""

from __future__ import annotations

import json
import zipfile

import modal

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ID = "aims-foundations/measurement-db"
REGISTRY_FILES = {"subjects.parquet", "items.parquet", "benchmarks.parquet"}

EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"  # MUST match models.txt
N_FACTORS = 4

# Set SKIP_STAGE1 = True to skip Stage 1 and use N_FACTORS directly.
SKIP_STAGE1 = True
STAGE1_EPOCHS = 150
STAGE1_LR = 0.05

STAGE2A_EPOCHS = 500
STAGE2A_LR = 1e-2
STAGE2A_ALPHA = 1e-3  # L2 regularization on the regressor

SEED = 42
DEVICE = "cuda"
GPU = "A10G"
TIMEOUT = 60 * 60       # 1 h
EMBED_BATCH_SIZE = 256

RATING_BENCHMARKS = {
    "ultrafeedback": 4.0,
    "mtbench":       7.0,
}

BUILD_DIR = "/root/build"
ZIP_PATH_REMOTE = "/root/submission_artifacts.zip"
ZIP_PATH_LOCAL = "submission_artifacts.zip"

ARTIFACT_NAMES = ["abilities.pt", "subject_lookup.json", "stage2a.pt", "config.json"]

DATASET_DESCRIPTIONS_PATH = "/root/data/dataset_descriptions.json"


# ---------------------------------------------------------------------------
# Modal app / image
# ---------------------------------------------------------------------------

app = modal.App("pge-artifact-builder")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "pandas",
        "datasets",
        "huggingface_hub",
        "sentence-transformers",
    )
    .env({"PYTHONPATH": "/root/src"})
    .add_local_python_source("torch_measure")
    .add_local_python_source("src")
    .add_local_file("data/dataset_descriptions.json", DATASET_DESCRIPTIONS_PATH)
)


# ---------------------------------------------------------------------------
# Subject-name normalisation — MUST be identical in model.py
# ---------------------------------------------------------------------------


def normalize_name(name: str) -> str:
    return " ".join(str(name).strip().lower().split())


def parse_subject_name(subject_content: str) -> str:
    for line in str(subject_content).splitlines():
        line = line.strip()
        if line.lower().startswith("name:"):
            return line.split(":", 1)[1].strip()
    for line in str(subject_content).splitlines():
        if line.strip():
            return line.strip()
    return ""


# ---------------------------------------------------------------------------
# Global response matrix
# ---------------------------------------------------------------------------


def load_global_matrix(verbose: bool = True):
    import torch
    from datasets import Features, Value, load_dataset
    from huggingface_hub import HfApi

    repo_files = HfApi().list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    response_files = sorted(
        n for n in repo_files
        if n.endswith(".parquet")
        and n not in REGISTRY_FILES
        and not n.endswith("_traces.parquet")
    )
    response_features = Features({
        "subject_id": Value("string"), "item_id": Value("string"),
        "benchmark_id": Value("string"), "trial": Value("int64"),
        "test_condition": Value("string"), "response": Value("float64"),
        "correct_answer": Value("string"), "trace": Value("string"),
    })

    if verbose:
        print(f"Loading {len(response_files)} response parquet files ...")
    responses_ds = load_dataset(REPO_ID, data_files=response_files,
                                features=response_features, split="train")
    items_ds = load_dataset(REPO_ID, data_files="items.parquet", split="train")
    subjects_ds = load_dataset(REPO_ID, data_files="subjects.parquet", split="train")

    df = responses_ds.to_pandas()
    items_df = items_ds.to_pandas()
    subjects_df = subjects_ds.to_pandas()

    def binarize(row):
        thr = RATING_BENCHMARKS.get(row["benchmark_id"])
        if thr is not None:
            return float(row["response"] >= thr)
        return float(row["response"] >= 0.5)
    df["binary"] = df.apply(binarize, axis=1)

    cell = df.groupby(["subject_id", "item_id"])["binary"].mean().reset_index()
    cell["binary"] = (cell["binary"] >= 0.5).astype(float)

    subject_ids = sorted(cell["subject_id"].unique().tolist())
    item_ids = sorted(cell["item_id"].unique().tolist())
    s_idx = {s: i for i, s in enumerate(subject_ids)}
    i_idx = {it: j for j, it in enumerate(item_ids)}

    responses = torch.full((len(subject_ids), len(item_ids)), float("nan"))
    for sid, iid, v in cell.itertuples(index=False):
        responses[s_idx[sid], i_idx[iid]] = v

    text_col = "content" if "content" in items_df.columns else items_df.columns[0]
    item_text = {str(r["item_id"]): str(r[text_col]) for _, r in items_df.iterrows()}

    if "benchmark_id" in items_df.columns:
        item_benchmark = {
            str(r["item_id"]): str(r["benchmark_id"]) for _, r in items_df.iterrows()
        }
    else:
        bm_map = df.drop_duplicates("item_id").set_index("item_id")["benchmark_id"]
        item_benchmark = {str(k): str(v) for k, v in bm_map.items()}

    sub_by_id = {r["subject_id"]: r for _, r in subjects_df.iterrows()}
    name_to_idx_extra: dict[str, int] = {}
    for sid, idx in s_idx.items():
        rec = sub_by_id.get(sid, {})
        keys = {sid}
        dn = rec.get("display_name")
        if dn:
            keys.add(dn)
        aliases = rec.get("raw_labels_seen")
        if aliases is not None:
            try:
                keys.update(str(a) for a in aliases if a)
            except TypeError:
                pass
        for k in keys:
            name_to_idx_extra.setdefault(normalize_name(k), idx)

    if verbose:
        obs = ~torch.isnan(responses)
        print(f"Global matrix: {len(subject_ids)} x {len(item_ids)} "
              f"({int(obs.sum()):,} obs, density {obs.float().mean():.2%})")
    return responses, subject_ids, item_ids, item_text, item_benchmark, name_to_idx_extra


# ---------------------------------------------------------------------------
# Item embedding
# ---------------------------------------------------------------------------


def embed_items(
    item_ids: list[str],
    item_text: dict[str, str],
    item_benchmark: dict[str, str],
    dataset_descriptions: dict[str, str],
    model_id: str,
    device: str,
    batch_size: int = 256,
    verbose: bool = True,
):
    from sentence_transformers import SentenceTransformer

    if verbose:
        print(f"\nEmbedding {len(item_ids)} items with {model_id} ...")

    def _build_text(iid: str) -> str:
        content = item_text.get(iid, "") or ""
        bm_id = item_benchmark.get(iid, "")
        description = dataset_descriptions.get(bm_id, "")
        if description:
            return f"{description}\n\n{content}"
        return content

    texts = [_build_text(iid) for iid in item_ids]
    st = SentenceTransformer(model_id, device=device)
    emb = st.encode(
        texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=verbose,
    ).cpu()
    return emb.float(), int(st.max_seq_length)


# ---------------------------------------------------------------------------
# Remote GPU build
# ---------------------------------------------------------------------------


@app.function(image=image, gpu=GPU, timeout=TIMEOUT)
def build_artifacts() -> dict[str, bytes]:
    """Run the full pipeline on a Modal GPU and return artifact bytes."""
    import io
    from pathlib import Path

    import torch

    from pge.stage2a import run_stage2a

    torch.manual_seed(SEED)
    Path(BUILD_DIR).mkdir(parents=True, exist_ok=True)

    # --- Load benchmark descriptions ---
    with open(DATASET_DESCRIPTIONS_PATH) as f:
        dataset_descriptions: dict[str, str] = json.load(f)
    print(f"Loaded descriptions for {len(dataset_descriptions)} benchmarks.")

    # --- 1. Global matrix ---
    responses, subject_ids, item_ids, item_text, item_benchmark, name_to_idx = (
        load_global_matrix()
    )
    responses_cpu = responses.cpu()

    # --- 2. [Optional] Stage 1 (LogisticFM) — only to infer n_factors ---
    n_factors = N_FACTORS
    stage1_model = None

    if not SKIP_STAGE1:
        from torch_measure.models import LogisticFM

        print(f"\nFitting Stage 1 (LogisticFM) for n_factors inference ...")
        responses_gpu = responses.to(DEVICE)
        observed = ~torch.isnan(responses_gpu)
        stage1_model = LogisticFM(
            responses.shape[0], responses.shape[1],
            n_factors=N_FACTORS, device=DEVICE,
        )
        stage1_model.fit(
            responses_gpu,
            mask=observed,
            max_epochs=STAGE1_EPOCHS,
            lr=STAGE1_LR,
            verbose=True,
        )
        n_factors = stage1_model.loadings.shape[1]
        print(f"  Stage 1 done. n_factors={n_factors}")
        del responses_gpu   # free GPU memory before embedding pass

    # --- 3. Embed all items ---
    embeddings, max_seq_len = embed_items(
        item_ids, item_text, item_benchmark, dataset_descriptions,
        EMBED_MODEL, DEVICE, batch_size=EMBED_BATCH_SIZE,
    )
    embed_dim = embeddings.shape[1]

    # --- 4. Stage 2a ---
    stage1_out = {
        "model":       stage1_model,  # None if SKIP_STAGE1
        "n_factors":   n_factors,
        "item_ids":    item_ids,
        "subject_ids": subject_ids,
        "responses":   responses_cpu,
    }
    stage2a_out = run_stage2a(
        stage1_out,
        embeddings,                    # raw L2-normalised embeddings (not yet z-scored)
        max_epochs=STAGE2A_EPOCHS,
        lr=STAGE2A_LR,
        alpha=STAGE2A_ALPHA,
        seed=SEED,
        device=DEVICE,
        verbose=True,
    )

    # --- 5. Serialize artifacts ---
    print("\nSerializing artifacts ...")
    artifacts: dict[str, bytes] = {}

    # abilities.pt
    ability: torch.Tensor = stage2a_out["ability"]  # (n_subjects, K)
    buf = io.BytesIO()
    torch.save({"ability": ability, "n_factors": n_factors}, buf)
    artifacts["abilities.pt"] = buf.getvalue()

    # subject_lookup.json
    name_to_idx_out: dict[str, int] = {}
    for idx, sid in enumerate(subject_ids):
        name_to_idx_out.setdefault(normalize_name(sid), idx)
    for k, idx in name_to_idx.items():
        name_to_idx_out.setdefault(k, idx)
    artifacts["subject_lookup.json"] = json.dumps(
        {"name_to_idx": name_to_idx_out, "subject_ids": subject_ids}
    ).encode()

    # stage2a.pt
    net = stage2a_out["model"]
    buf = io.BytesIO()
    torch.save(
        {
            **net.get_save_dict(),
            "feat_mean": stage2a_out["feat_mean"].cpu(),
            "feat_std":  stage2a_out["feat_std"].cpu().clamp(min=1e-3),
        },
        buf,
    )
    artifacts["stage2a.pt"] = buf.getvalue()

    # config.json
    artifacts["config.json"] = json.dumps(
        {
            "embed_model":          EMBED_MODEL,
            "n_factors":            n_factors,
            "embed_dim":            embed_dim,
            "max_seq_len":          max_seq_len,
            "normalize_embeddings": True,
            "pooling":              "mean",
            "stage2a_variant":      "amortized_em_linear",   # for traceability
        },
        indent=2,
    ).encode()

    return artifacts


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main() -> None:
    artifacts = build_artifacts.remote()

    print("\nZipping artifacts locally ...")
    with zipfile.ZipFile(ZIP_PATH_LOCAL, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in ARTIFACT_NAMES:
            zf.writestr(name, artifacts[name])
    print(f"\nDONE. Bundle -> {ZIP_PATH_LOCAL}")
    print("Unzip its contents into your submission's hf_submission/ next to model.py.")