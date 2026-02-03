# Walmart Demand Forecasting (M5)

Predict 28 days of daily item‑level demand using the M5 Forecasting dataset. This project is written as a portfolio-ready case study: start with a strong local baseline, then scale to a global model that learns shared patterns across ~30k series.

This repo is designed for two audiences:
- **Readers:** follow the narrative in the notebooks.
- **Reviewers/recruiters:** reproduce the exact results via `make` or the CLI and inspect the generated artifacts (metrics + timing + saved models).

## Narrative (recommended reading order)

1) **Act 1 — Local Dynamics:** [Notebooks/01_local_dynamics.ipynb](Notebooks/01_local_dynamics.ipynb)
- Focus: a high‑volatility subset (**CA‑FOODS**) as a stress test.
- Baseline: local, feature‑engineered **LightGBM**.
- Challenger: local **N‑BEATSx** with quantile training.
- Outcome: **N‑BEATSx slightly wins locally** on this split (a reminder that feature engineering isn’t the only path to strong results, even on a volatile subset).

2) **Act 2 — Global Scale:** [Notebooks/02_global_scale.ipynb](Notebooks/02_global_scale.ipynb)
- Focus: a RAM‑aware pipeline for the full M5 item‑store panel.
- Control: global LightGBM with lags/rolling/calendar/price/event features.
- Final model: global **N‑BEATSx** leveraging cross‑learning.
- Outcome: N‑BEATSx slightly improves RMSE and item‑level WRMSSE on the same split, with less feature‑engineering overhead.

## Evaluation contract (what “good” means here)

- **Horizon:** 28 days.
- **Splits:** contiguous time split(s) with the final 28 days held out as test.
- **Reporting metrics:**
	- **RMSE** (reported on point forecasts; for quantile-trained models, RMSE is computed on the **median** forecast).
	- **WRMSSE (item-level / Level 12 only)** in Act 2 for a revenue‑weighted view. Full hierarchical WRMSSE is intentionally out of scope for laptop RAM/time.

## Results (from notebook outputs)

These numbers come from the printed outputs in the notebooks (the project’s source of truth).

| Setting | LightGBM RMSE | N‑BEATSx RMSE (median) | Item‑level WRMSSE (N‑BEATSx / LightGBM) | Reference runtime (wall‑clock) |
|---|---:|---:|---:|---:|
| **Act 1 (Local: CA‑FOODS)** | **2.6355** | **2.6286** | — | **LGBM ~4.3 min**, **N‑BEATSx ~6.2 min** |
| **Act 2 (Global: full panel)** | **2.2209** | **2.1814** | **0.8568 / 0.8669** | **LGBM ~20.6 min**, **N‑BEATSx ~29.5 min** |

Notes:
- The local gap is intentionally small (this is a “realistic” comparison, not a contrived demo).
- WRMSSE here is **Level 12 only** (item‑store). Full 12‑level WRMSSE is not computed.
- Runtimes come from the `timing.json` files produced by one reference run on an Apple Silicon Mac (MPS enabled). Your hardware will vary.

## Expected outputs (quick scan)

If you want a fast “does this work?” check without opening notebooks, reproduce runs and compare the emitted JSON artifacts.

### Act 1 — Local (CA‑FOODS)

After running:

```bash
make run_local_lgbm
make run_local_nbeatsx
```

You should see metrics similar to:

```json
// Artifacts/runs/<...>/metrics_local_lgbm.json
{ "horizon": 28, "rmse": 2.6355, "start_day": 1000 }
```

```json
// Artifacts/runs/<...>/metrics_local_nbeatsx.json
{ "horizon": 28, "rmse": 2.6286, "accelerator": "mps", "start_day": 1000 }
```

Local runs intentionally do **not** include WRMSSE.

### Act 2 — Global (full panel)

After running:

```bash
make run_global_lgbm
make run_global_nbeatsx
```

You should see metrics similar to:

```json
// Artifacts/runs/<...>/metrics_global_lgbm.json
{ "horizon": 28, "rmse": 2.2209, "wrmsse_item_level": 0.8669, "start_day": 1200 }
```

```json
// Artifacts/runs/<...>/metrics_global_nbeatsx.json
{ "horizon": 28, "rmse": 2.1814, "wrmsse_item_level": 0.8568, "accelerator": "mps", "start_day": 1200 }
```

## Why this project is interesting

- Demonstrates end‑to‑end thinking: data pipeline constraints, strong baselines, model selection, and business‑oriented evaluation.
- Keeps the outcome honest: on this specific local split, the deep model is allowed to be competitive.
- Makes the tradeoff explicit: **accuracy vs engineering maintenance** at scale.

## How to run

### Option A — Conda (recommended)

1) Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate Zak_Las_Env
```

2) Editable install (run once per environment):

```bash
python -m pip install -e .
```

3) Verify data is present:

```bash
make verify_m5
```

4) Run the notebooks (recommended reading order):

1) [Notebooks/01_local_dynamics.ipynb](Notebooks/01_local_dynamics.ipynb)
2) [Notebooks/02_global_scale.ipynb](Notebooks/02_global_scale.ipynb)

Notebook kernel note: notebooks run in the currently selected Jupyter kernel. If the kernel is not the same environment where you installed the package, imports may fail. The notebooks include a small “editable install if missing” setup cell to self‑heal when `walmart_demand_forecasting` is not importable.

### Option B — Dev Container (VS Code)

If you open this repo in VS Code with the **Dev Containers** extension installed, VS Code will detect [.devcontainer/devcontainer.json](.devcontainer/devcontainer.json) and prompt you to **Reopen in Container**.

Inside the container terminal:

```bash
python -m pip install -e .
make verify_m5
```

Then run notebooks or CLI/Make repro as shown below.

## Data (M5 download)

This repo expects the core M5 CSVs to exist in the repo-root `data/` folder.

- Download via Kaggle (recommended):
	- `make download_m5`
- Verify required files exist:
	- `make verify_m5`

Notes:
- You must have Kaggle credentials at `~/.kaggle/kaggle.json` and have accepted the competition rules.
- `data/` is intentionally gitignored (raw data should not be committed).
- Common failure mode: Kaggle occasionally yields a ZIP archive saved with a `.csv` name. Use `make verify_m5` / `make repair_m5` to fix this at staging time.

## Reproduce results (recruiter checklist)

This section provides copy‑paste commands to reproduce all four models and inspect the artifacts produced by the CLI.

### Reproduce via `make` (fastest)

Run these one by one (each command creates a fresh run folder under `Artifacts/runs/`):

```bash
make run_local_lgbm
make run_local_nbeatsx
make run_global_lgbm
make run_global_nbeatsx
```

Inspect the most recent run for each model:

```bash
make show_local_lgbm
make show_local_nbeatsx
make show_global_lgbm
make show_global_nbeatsx
```

### Reproduce via CLI (exact commands)

Each run writes:
- `Artifacts/runs/<run_name_or_timestamp>/metrics_*.json`
- `Artifacts/runs/<run_name_or_timestamp>/timing.json`
- `Artifacts/runs/<run_name_or_timestamp>/run_meta.json`

Run once:

```bash
python -m pip install -e .
```

Then run each model (copy/paste lines):

```bash
python -m walmart_demand_forecasting.cli --config configs/local.toml  --run-name repro_local_lgbm  local-lgbm
python -m walmart_demand_forecasting.cli --config configs/local.toml  --run-name repro_local_nbeatsx  local-nbeatsx
python -m walmart_demand_forecasting.cli --config configs/global.toml --run-name repro_global_lgbm global-lgbm
python -m walmart_demand_forecasting.cli --config configs/global.toml --run-name repro_global_nbeatsx global-nbeatsx
```

Inspect artifacts directly:

```bash
cat Artifacts/runs/repro_local_lgbm/metrics_local_lgbm.json
cat Artifacts/runs/repro_local_lgbm/timing.json

cat Artifacts/runs/repro_local_nbeatsx/metrics_local_nbeatsx.json
cat Artifacts/runs/repro_local_nbeatsx/timing.json

cat Artifacts/runs/repro_global_lgbm/metrics_global_lgbm.json
cat Artifacts/runs/repro_global_lgbm/timing.json

cat Artifacts/runs/repro_global_nbeatsx/metrics_global_nbeatsx.json
cat Artifacts/runs/repro_global_nbeatsx/timing.json
```

## Artifacts

All training outputs go under `Artifacts/` (local-only by default):
- `Artifacts/runs/<run_name_or_timestamp>/`: per-run metrics, timing, run metadata, predictions, and model files
- `Artifacts/model_saves/`: longer-lived saved models
- `Artifacts/logs/`: training logs

The repo does not commit raw data or large training artifacts by default (see `.gitignore`). It is configured to allow committing **only** lightweight run evidence under `Artifacts/runs/`:
- `Artifacts/runs/*/metrics_*.json`
- `Artifacts/runs/*/timing.json`

## Tests

This repo includes a small, fast unit test suite focused on the reusable helpers (features, splits, metrics, and CSV memory squeezing).

- Recommended (ensures editable install first):
	- `make test`
- Direct (if you already did an editable install):
	- `python -m pytest -q`

Notes:
- `make test` runs `python -m pip install -e .` and then `python -m pytest -q` using the same interpreter.
- Tests are designed to be lightweight (no full model training).

## Artifacts

Training logs and saved model artifacts live under `Artifacts/` (e.g., `Artifacts/logs/` and `Artifacts/model_saves/`).
These are intentionally not committed to Git.

## Planned next steps

- Expand unit test coverage (datasets + model helper edges).
- Add CI to run `make test` + static checks on PRs.
- Add a small “quick inference” demo that loads the saved models and generates a 28-day forecast without retraining.
- Deploy the global N‑BEATSx inference path as a lightweight API (planned target: **AWS Lambda**, with model artifacts stored in object storage).

## Notes / limitations

- Results are tied to the specific split protocol in the notebooks; alternative splits can change rankings.
- ROI-style numbers (if included) are illustrative and depend on explicit inventory policy assumptions.

## Repo structure (high-level)
This repo uses a **src-layout** installable package plus **notebook-first** narrative.

- `Notebooks/`: narrative + orchestration (portfolio-friendly)
	- `01_local_dynamics.ipynb`: Act 1 (local CA-FOODS baseline vs N-BEATSx)
	- `02_global_scale.ipynb`: Act 2 (full-panel RAM-aware pipeline + global models)
- `.devcontainer/`: Dev Container definition for 1-click VS Code setup
- `src/walmart_demand_forecasting/`: reusable code (import as `walmart_demand_forecasting.*`)
	- `datasets/m5_loaders.py`: M5 loaders (CA-FOODS + global) and `Paths` defaults
	- `features/m5_features.py`: feature engineering (lags, rolling means, time features)
	- `evaluation/split.py`: contiguous split helpers used in notebooks
	- `evaluation/metrics.py`: RMSE + item-level WRMSSE helper (`compute_item_level_wrmsse`)
	- `models/lgbm/`: Optuna tuning + training + inference helpers
	- `models/nbeatsx/`: NeuralForecast/NBEATSx pipeline helpers + logging utilities
	- `visualization/`: plotting helpers (e.g., Lightning CSV learning curves)
- `configs/`: run configs for CLI pipelines (`local.toml`, `global.toml`, `smoke.toml`)
- `data/`: M5 dataset files (CSV)
- `Artifacts/`: local artifacts (training logs + saved models; not tracked by Git)
- `environment.yml`: conda environment definition (most runtime deps live here)
- `pyproject.toml`: minimal packaging metadata for editable installs
- `Makefile`: reproducible entrypoints (`run_*` and `show_*` targets)

### Data & code flow (conceptual)
- Notebooks call loaders in `walmart_demand_forecasting.datasets.m5_loaders` to produce model-ready panels.
- Feature engineering happens via `walmart_demand_forecasting.features.m5_features`.
- Splits + reporting metrics come from `walmart_demand_forecasting.evaluation.*`.
- Models are trained/evaluated via `walmart_demand_forecasting.models.lgbm.*` and `walmart_demand_forecasting.models.nbeatsx.*`.
