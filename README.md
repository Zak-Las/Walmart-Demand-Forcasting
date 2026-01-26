# Walmart Demand Forecasting (M5)

Predict 28 days of daily item‑level demand using the M5 Forecasting dataset. This project is written as a portfolio-ready case study: start with a strong local baseline, then scale to a global model that learns shared patterns across ~30k series.

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

| Setting | LightGBM RMSE | N‑BEATSx RMSE (median) | Item‑level WRMSSE (N‑BEATSx / LightGBM) |
|---|---:|---:|---:|
| **Act 1 (Local: CA‑FOODS)** | **2.6355** | **2.6286** | — |
| **Act 2 (Global: full panel)** | **2.2209** | **2.1814** | **0.8568 / 0.8669** |

Notes:
- The local gap is intentionally small (this is a “realistic” comparison, not a contrived demo).
- WRMSSE here is **Level 12 only** (item‑store). Full 12‑level WRMSSE is not computed.

## Why this project is interesting

- Demonstrates end‑to‑end thinking: data pipeline constraints, strong baselines, model selection, and business‑oriented evaluation.
- Keeps the outcome honest: on this specific local split, the deep model is allowed to be competitive.
- Makes the tradeoff explicit: **accuracy vs engineering maintenance** at scale.

## How to run

- Recommended: run the notebooks in order:
	1) [Notebooks/01_local_dynamics.ipynb](Notebooks/01_local_dynamics.ipynb)
	2) [Notebooks/02_global_scale.ipynb](Notebooks/02_global_scale.ipynb)

- Environment:
	- Create the conda env from `environment.yml`.
	- Recommended: install the package in editable mode so imports work everywhere:
		- `pip install -e .`
	- **Notebook kernel note:** notebooks run in the currently selected Jupyter kernel. If the kernel is not the same environment where you installed the package, imports may fail.
		- The notebooks include a small “editable install if missing” setup cell to self-heal when `walmart_demand_forecasting` is not importable in that kernel.

	## Tests

	This repo includes a small, fast unit test suite focused on the reusable helpers (features, splits, metrics, and CSV memory squeezing).

	- Recommended (ensures editable install first):
		- `make test`
	- Direct (if you already did an editable install):
		- `python -m pytest -q`

	Notes:
	- `make test` runs `python -m pip install -e .` and then `python -m pytest -q` using the same interpreter.
	- Tests are designed to be lightweight (no full model training).

## Planned next steps

- Expand unit test coverage (datasets + model helper edges).
- Add a small CLI or runner script for reproducible training/inference without opening notebooks.
- Deploy the global N‑BEATSx inference path as a lightweight API (planned target: **AWS Lambda**, with model artifacts stored in object storage).

## Notes / limitations

- Results are tied to the specific split protocol in the notebooks; alternative splits can change rankings.
- ROI-style numbers (if included) are illustrative and depend on explicit inventory policy assumptions.

## Repo structure (high-level)


This repo uses a **src-layout** installable package plus **notebook-first** narrative.

- `Notebooks/`: narrative + orchestration (portfolio-friendly)
	- `01_local_dynamics.ipynb`: Act 1 (local CA-FOODS baseline vs N-BEATSx)
	- `02_global_scale.ipynb`: Act 2 (full-panel RAM-aware pipeline + global models)
- `src/walmart_demand_forecasting/`: reusable code (import as `walmart_demand_forecasting.*`)
	- `datasets/m5.py`: M5 loaders (CA-FOODS + global) and `Paths` defaults
	- `features/m5_features.py`: feature engineering (lags, rolling means, time features)
	- `evaluation/split.py`: contiguous split helpers used in notebooks
	- `evaluation/metrics.py`: RMSE + item-level WRMSSE helper (`compute_item_level_wrmsse`)
	- `models/lgbm/`: Optuna tuning + training + inference helpers
	- `models/nbeatsx/`: NeuralForecast/NBEATSx pipeline helpers + logging utilities
	- `visualization/`: plotting helpers (e.g., Lightning CSV learning curves)
- `data/`: M5 dataset files (CSV)
- `model_saves/`: will be used to save the global models (not tracked by Git)
- `environment.yml`: conda environment definition (most runtime deps live here)
- `pyproject.toml`: minimal packaging metadata for editable installs

### Data & code flow (conceptual)
- Notebooks call loaders in `walmart_demand_forecasting.datasets.m5` to produce model-ready panels.
- Feature engineering happens via `walmart_demand_forecasting.features.m5_features`.
- Splits + reporting metrics come from `walmart_demand_forecasting.evaluation.*`.
- Models are trained/evaluated via `walmart_demand_forecasting.models.lgbm.*` and `walmart_demand_forecasting.models.nbeatsx.*`.
