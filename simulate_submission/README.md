# simulate_submission

Offline harness that simulates an official **Predictive AI Evaluation Challenge**
submission end-to-end: it curates real (subject, item) examples across
benchmarks, runs them through your `submission/model.py:predict` exactly as the
Codabench sandbox would, and scores the result with the leaderboard metrics
(val NLL, higher-is-better, plus AUROC).

## Layout assumed

```
<root>/
  submission/           # your real submission (must contain model.py)
    model.py
    config.py
    subject_ability.py  calibrate.py  ...
  simulate_submission/  # this folder
    config.py
    curate.py
    metrics.py
    run_submission.py
    main.py
    runs/               # created on first run; one subfolder per run
```

If your `submission/` lives elsewhere, set `SUBMISSION_DIR` in `config.py`.

## Run

```bash
cd simulate_submission
python main.py
```

## What it does

1. **Curate** (`curate.py`) — pulls examples across the benchmarks listed in
   `config.BENCHMARK_NAMES`. Two interchangeable sources (`DATA_SOURCE`):
   - `torch_measure` — mirrors `multi_benchmark.py` (`torch_measure.datasets.load`,
     binary filter that drops graded benchmarks, per-benchmark subsampling).
   - `hf` — mirrors the README: loads the explicit Parquet response + registry
     tables and joins via `render_subject_content` / `to_training_example`.
   - `auto` (default) — try `torch_measure`, fall back to `hf`.

   Each curated example carries the four `predict()` input fields **plus** the
   ground-truth binary `label` (held out from `predict`, used only for scoring
   and the adaptive-label channel).

2. **Convert** — each example is reduced to exactly
   `{benchmark, condition, subject_content, item_content}` (the contract's four
   string keys) before being handed to `predict()`.

3. **Run** (`run_submission.py`) — imports `submission/model.py` the way the
   sandbox does (parent of `submission/` on `sys.path`, then
   `import submission.model`), builds the **labeled channel** (up to
   `LABELS_PER_BENCHMARK` real labels per benchmark, reused on every call as the
   spec requires), and calls `predict()` once per example. Exceptions, invalid
   outputs (NaN/inf/out-of-range/non-coercible), and numpy/torch scalars are all
   caught per-example and logged, so one bad call never aborts the run.

4. **Score** (`metrics.py`) — computes:
   - **val NLL** = mean log-likelihood = `-log_loss` (higher is better; the
     leaderboard convention),
   - **AUROC** (secondary), via a tie-aware rank statistic (no sklearn needed),
   - plus log-loss, Brier, base rate, mean prediction. If `sklearn` is present
     it cross-checks log-loss/AUROC.

5. **Report + persist** (`main.py`) — prints an overall + per-benchmark summary
   and writes to `runs/run_<UTC-timestamp>/`:
   - `results.csv` — per-example `benchmark, condition, label, pred, status`,
   - `metrics.json` — overall + per-benchmark metrics,
   - `metadata.json` — config snapshot, run stats, environment.

## Config knobs (`config.py`)

| key | meaning |
| --- | --- |
| `DATA_SOURCE` | `torch_measure` / `hf` / `auto` |
| `SUBMISSION_DIR` | path to the folder containing `model.py` |
| `BENCHMARK_NAMES` | benchmarks to pull |
| `MAX_ITEMS_PER_BENCHMARK` | items sampled per benchmark |
| `MAX_RESPONSES_PER_ITEM` | cap on (subject, item) rows per item |
| `MAX_TOTAL_EXAMPLES` | hard cap on examples scored (default 5000, matching the round size) |
| `LABELS_PER_BENCHMARK` | K real labels revealed to `predict()` per benchmark; `0` = cold/no-label |
| `SEED` | determinism for sampling |

## Notes

- This is a **proxy** for the hidden test set, not the real thing: it scores on
  examples whose responses appear in the public training data, so a model that
  memorizes training responses will look better here than on the true cold-start
  hidden slice. Use it for plumbing/sanity/calibration checks and relative
  comparisons, not as a leaderboard oracle.
- `AUROC` is `n/a` when a slice is single-class (undefined), matching the
  competition's debugging-only treatment of the secondary metric.
- Requires whatever your submission imports (e.g. `torch`), plus `pandas`,
  `datasets`/`huggingface_hub` (for the `hf` path) or `torch_measure` (for the
  `torch_measure` path). `sklearn` is optional.
