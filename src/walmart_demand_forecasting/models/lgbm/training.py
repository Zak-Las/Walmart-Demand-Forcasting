from __future__ import annotations

from typing import Any


def train_lightgbm_with_best_params(
    *,
    train_data: "Any",
    valid_data: "Any",
    best_params: dict[str, Any],
    seed: int = 42,
    objective: str = "tweedie",
    metric: str = "rmse",
    bin_construct_sample_cnt: int = 200000,
    num_threads: int = -1,
    verbose: int = -1,
    num_boost_round: int = 1500,
    early_stopping_rounds: int = 100,
    log_evaluation: int = 50,
) -> tuple["Any", dict[str, Any]]:
    """Train LightGBM using tuned params plus fixed defaults.

    This extracts the final training block from `Notebooks/02_global_scale.ipynb`.

    Notes
    -----
    - `best_params` is typically `study.best_params` from Optuna, but this helper
      intentionally accepts a plain dict to avoid a hard dependency on Optuna.
    - Returns both the trained Booster and the final params dict for traceability.
    """

    import lightgbm as lgb

    params = dict(best_params)
    params.update(
        {
            "objective": objective,
            "metric": metric,
            "bin_construct_sample_cnt": bin_construct_sample_cnt,
            "num_threads": num_threads,
            "verbose": verbose,
            "seed": seed,
        }
    )

    model = lgb.train(
        params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[train_data, valid_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds),
            lgb.log_evaluation(log_evaluation),
        ],
    )

    return model, params
