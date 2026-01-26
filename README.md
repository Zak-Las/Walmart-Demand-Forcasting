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
	- If running scripts outside notebooks, set `PYTHONPATH` to the repo root so `src.*` imports work.

## Planned next steps

- Add unit tests around data loading, feature generation, and scoring.
- Add a small CLI or runner script for reproducible training/inference without opening notebooks.
- Deploy the global N‑BEATSx inference path as a lightweight API (planned target: **AWS Lambda**, with model artifacts stored in object storage).

## Notes / limitations

- Results are tied to the specific split protocol in the notebooks; alternative splits can change rankings.
- ROI-style numbers (if included) are illustrative and depend on explicit inventory policy assumptions.

## Repo structure (high-level)

- `Notebooks/`: narrative + orchestration (portfolio-friendly).
- `src/m5/`: reusable helpers (loading/splitting/features/metrics + LightGBM + NeuralForecast/N‑BEATSx utilities).
- `data/`: M5 dataset files.
