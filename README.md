# Walmart Demand Forecasting (M5)

Predict 28 days of daily item‑level demand using the M5 Forecasting dataset. This project is written as a portfolio-ready case study: start with a strong local baseline, then scale to a global model that learns shared patterns across ~30k series.

## Narrative (recommended reading order)

1) **Act 1 — Local Dynamics:** [Notebooks/01_local_dynamics.ipynb](Notebooks/01_local_dynamics.ipynb)
- Focus: a high‑volatility subset (**CA‑FOODS**) as a stress test.
- Baseline: local, feature‑engineered **LightGBM**.
- Challenger: local **N‑BEATSx** with quantile training.
- Outcome: LightGBM wins locally (illustrates that scale/diversity matter).

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

## Why this project is interesting

- Demonstrates end‑to‑end thinking: data pipeline constraints, strong baselines, model selection, and business‑oriented evaluation.
- Shows a credible failure mode: deep learning is not forced to “win” in the local regime.
- Makes the tradeoff explicit: **accuracy vs engineering maintenance** at scale.

## Planned next steps

- Refactor notebooks into a reusable training/inference package.
- Add unit tests around data loading, feature generation, and scoring.
- Deploy the global N‑BEATSx inference path as a lightweight API (planned target: **AWS Lambda**, with model artifacts stored in object storage).

## Notes / limitations

- Results are tied to the specific split protocol in the notebooks; alternative splits can change rankings.
- ROI-style numbers (if included) are illustrative and depend on explicit inventory policy assumptions.
