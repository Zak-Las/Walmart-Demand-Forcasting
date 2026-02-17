from __future__ import annotations

from typing import Literal

import pandas as pd


def tune_lightgbm_optuna(
    df_tune: pd.DataFrame,
    *,
    features: list[str],
    y_col: str,
    cat_feats: list[str] | None = None,
    seed: int = 42,
    objective: str = "tweedie",
    metric: str = "rmse",
    tweedie_variance_power_range: tuple[float, float] = (1.1, 1.5),
    learning_rate_range: tuple[float, float] = (0.01, 0.1),
    num_boost_round: int = 500,
    nfold: int = 3,
    n_trials: int = 15,
    study_name: str = "M5_LGBM_Global",
    direction: Literal["minimize", "maximize"] = "minimize",
    num_threads: int = -1,
    verbose: int = -1,
    free_raw_data: bool = False,
    return_tune_data: bool = False,
):
    """Tune a LightGBM model with Optuna using LightGBM CV + pruning.

    This is a notebook-friendly extraction of the Optuna tuning block used in
    `Notebooks/02_global_scale.ipynb`.

    Parameters mirror the notebook defaults (TPE sampler seeded, Tweedie objective,
    RMSE metric, and pruning callback on the CV metric).

    Returns
    -------
    study or (study, tune_data)
        If `return_tune_data=True`, also returns the LightGBM Dataset used for CV.
    """

    import lightgbm as lgb
    import numpy as np
    import optuna

    try:
        from optuna.integration import LightGBMPruningCallback
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Optuna LightGBM integration is unavailable. "
            "Install `optuna-integration[lightgbm]` or use a compatible Optuna build."
        ) from exc

    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    if num_boost_round <= 0:
        raise ValueError("num_boost_round must be positive")
    if nfold <= 1:
        raise ValueError("nfold must be >= 2")

    tune_data = lgb.Dataset(
        df_tune[features],
        label=df_tune[y_col],
        categorical_feature=cat_feats,
        free_raw_data=free_raw_data,
    )

    min_power, max_power = tweedie_variance_power_range
    min_lr, max_lr = learning_rate_range

    def objective_fn(trial: optuna.Trial) -> float:
        param_grid = {
            "objective": objective,
            "metric": metric,
            "tweedie_variance_power": trial.suggest_float(
                "tweedie_variance_power", min_power, max_power
            ),
            "learning_rate": trial.suggest_float("learning_rate", min_lr, max_lr),
            "seed": seed,
            "num_threads": num_threads,
            "verbose": verbose,
        }

        cv_results = lgb.cv(
            param_grid,
            tune_data,
            num_boost_round=num_boost_round,
            nfold=nfold,
            stratified=False,
            callbacks=[LightGBMPruningCallback(trial, metric)],
        )

        # Same as the notebook: best (last) CV metric value.
        # For RMSE, lower is better.
        key = f"valid {metric}-mean"
        if key not in cv_results:
            raise KeyError(
                f"Expected LightGBM CV results to include {key!r}. Available keys: {list(cv_results.keys())}"
            )
        return float(cv_results[key][-1])

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction=direction, study_name=study_name, sampler=sampler)
    study.optimize(objective_fn, n_trials=n_trials)

    if return_tune_data:
        return study, tune_data
    return study
