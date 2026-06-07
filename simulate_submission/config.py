"""Configuration for the submission simulation harness.

Tweak these without touching the rest of the code. The two ingestion paths are:

  source = "torch_measure"   -> mirrors multi_benchmark.py (the competition's
                                own loading path). Uses torch_measure.datasets.load.
  source = "hf"              -> mirrors the README helpers (to_training_example /
                                render_subject_content) reading the raw HuggingFace
                                Parquet tables directly.

"auto" tries torch_measure first, falls back to hf.
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent  # folder that *contains* simulate_submission/ and submission/

# --------------------------------------------------------------------------- #
# Where the official submission lives. The sandbox imports submission/model.py;
# we import it the same way (model.predict).
# --------------------------------------------------------------------------- #
# Default assumes layout:
#   <root>/
#     submission/model.py
#     simulate_submission/...   <- this folder
SUBMISSION_DIR = REPO_ROOT / "submission"

# Output directory for results + metadata.
OUTPUT_DIR = HERE / "runs"

# --------------------------------------------------------------------------- #
# Data ingestion
# --------------------------------------------------------------------------- #
DATA_SOURCE = "auto"  # "torch_measure" | "hf" | "auto"

HF_REPO_ID = "aims-foundations/measurement-db"

# Benchmarks to pull (from multi_benchmark.py EDA list). Graded ones are kept;
# the binary filter drops them. 
BENCHMARK_NAMES: list[str] = [
    "mmbench_v11",
    "ultrafeedback",   
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

# Curation knobs.
MAX_ITEMS_PER_BENCHMARK = 300      # items sampled per benchmark
MAX_RESPONSES_PER_ITEM = 3         # cap (subject,item) rows kept per item
MAX_TOTAL_EXAMPLES = 5000          # hard cap on examples actually scored
SEED = 42

# Binary label set used to filter responses to {0,1}.
BINARY_VALUES = (0, 0.0, 1, 1.0)

# Adaptive labeling: how many labeled anchors to reveal to predict() per
# benchmark
LABELS_PER_BENCHMARK = 5
RESTRICT_TO_KNOWN_SUBJECTS = False
LOG_LEVEL = "INFO"

SUBJECT_MEAN_MODE = "new_all"

# Global fallback mean used by modes that fall back (2b, 4). Set to your
# Stage 1 global mean; None means "let the model decide its own fallback".
GLOBAL_MEAN = None