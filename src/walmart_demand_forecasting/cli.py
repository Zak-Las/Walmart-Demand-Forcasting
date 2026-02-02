from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import tomllib  # py311+
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    models_dir: Path
    preds_dir: Path
    logs_root: Path


def _load_toml(path: Path) -> dict[str, Any]:
    if tomllib is None:  # pragma: no cover
        raise RuntimeError("tomllib unavailable; requires Python 3.11+")
    with path.open("rb") as f:
        return tomllib.load(f)


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = dict(base)
    for key, val in updates.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], val)
        else:
            out[key] = val
    return out


def _now_run_id(prefix: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"


def _ensure_run_dirs(*, artifacts_dir: Path, run_name: str) -> RunPaths:
    run_dir = artifacts_dir / "runs" / run_name
    models_dir = run_dir / "models"
    preds_dir = run_dir / "predictions"
    logs_root = run_dir / "logs"

    for p in [run_dir, models_dir, preds_dir, logs_root]:
        p.mkdir(parents=True, exist_ok=True)

    return RunPaths(run_dir=run_dir, models_dir=models_dir, preds_dir=preds_dir, logs_root=logs_root)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _maybe_git_sha() -> str | None:
    import subprocess

    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        return sha or None
    except Exception:
        return None


def _print_kv(title: str, items: dict[str, Any]) -> None:
    print(title)
    for k, v in items.items():
        print(f"  - {k}: {v}")


def _set_reproducibility(seed: int) -> None:
    """Best-effort seeding to match notebook runs.

    Notes
    -----
    - `PYTHONHASHSEED` only fully applies if set before the interpreter starts,
      but we still set it for traceability.
    """

    import random

    try:
        import numpy as np
    except Exception:  # pragma: no cover
        np = None  # type: ignore[assignment]

    random.seed(seed)
    if np is not None:
        np.random.seed(seed)

    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    # Optional: torch (covers NeuralForecast / Lightning)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _import_lightgbm():
    """Import LightGBM with a helpful error on macOS when OpenMP is missing."""

    try:
        import lightgbm as lgb

        return lgb
    except OSError as exc:
        msg = str(exc)
        if "libomp" in msg or "OpenMP" in msg or "@rpath/libomp.dylib" in msg:
            raise RuntimeError(
                "LightGBM failed to load because OpenMP (libomp) is missing.\n\n"
                "Fix (recommended for conda envs):\n"
                "  conda install -c conda-forge libomp lightgbm\n\n"
                "Fix (Homebrew alternative):\n"
                "  brew install libomp\n\n"
                "Then re-run the same `wdf ... lgbm` command."
            ) from exc
        raise


def _run_global_lgbm(*, cfg: dict[str, Any], paths: RunPaths) -> None:
    import numpy as np
    import pandas as pd

    lgb = _import_lightgbm()

    from walmart_demand_forecasting.datasets.m5_loaders import Paths as M5Paths
    from walmart_demand_forecasting.datasets.m5_loaders import load_global_lgbm
    from walmart_demand_forecasting.evaluation.split import contiguous_day_split
    from walmart_demand_forecasting.features.m5_features import (
        add_cyclical_time_features,
        add_group_lags,
        add_group_rolling_means,
        add_snap_active,
    )
    from walmart_demand_forecasting.models.lgbm.inference import evaluate_lgbm_on_test_mask
    from walmart_demand_forecasting.models.lgbm.io import save_lgbm_model
    from walmart_demand_forecasting.models.lgbm.training import train_lightgbm_with_best_params
    from walmart_demand_forecasting.models.lgbm.tuning import tune_lightgbm_optuna

    params = cfg["global"]["lgbm"]

    horizon = int(params["horizon"])
    start_day = int(params["start_day"])
    buffer = int(params.get("buffer", 60))

    m5_input_dir = Path(cfg["paths"].get("m5_input_dir", "data"))
    m5_paths = M5Paths(input_dir=m5_input_dir)

    seed = int(params.get("seed", 42))
    _set_reproducibility(seed)

    print("Loading global LightGBM dataset...")
    df = load_global_lgbm(start_day=start_day, buffer=buffer, paths=m5_paths)

    # Feature engineering (match notebook conventions)
    add_cyclical_time_features(df, wday_col="wday", month_col="month")
    add_snap_active(df, state_col="state_id")

    event_cols = ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]
    for col in event_cols:
        df[col] = df[col].astype("object").fillna("NoEvent")

    cat_feats = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "snap_active"] + event_cols
    for col in cat_feats:
        df[col] = df[col].astype("category")

    lags = list(params.get("lags", [28, 35, 42, 49]))
    df, lag_cols = add_group_lags(df, group_col="id", value_col="sales", lags=lags, dtype="float32")

    windows = list(params.get("rolling_windows", [7, 28, 49]))
    shift = int(params.get("rolling_shift", 28))
    df, rolling_cols = add_group_rolling_means(
        df,
        group_col="id",
        value_col="sales",
        shift=shift,
        windows=windows,
        dtype="float32",
    )

    # Cut off buffer (only needed for lag/rolling warmup)
    df = df[df["d_int"] >= start_day]
    df = df.dropna(subset=lag_cols + rolling_cols)
    df["date"] = pd.to_datetime(df["date"])

    split = contiguous_day_split(df, horizon=horizon, d_int_col="d_int")

    drop_cols = [
        "sales",
        "d",
        "date",
        "wm_yr_wk",
        "month",
        "wday",
        "snap_CA",
        "snap_TX",
        "snap_WI",
    ]
    features = [c for c in df.columns if c not in drop_cols]

    train_data = lgb.Dataset(
        df[split.train_mask][features],
        label=df[split.train_mask]["sales"],
        categorical_feature=cat_feats,
        free_raw_data=True,
    )
    valid_data = lgb.Dataset(
        df[split.valid_mask][features],
        label=df[split.valid_mask]["sales"],
        reference=train_data,
        categorical_feature=cat_feats,
        free_raw_data=True,
    )

    tune = bool(params.get("tune", True))
    if tune:
        print("Tuning LightGBM with Optuna...")
        tune_frac = float(params.get("tune_frac", 0.1))
        n_trials = int(params.get("n_trials", 15))
        num_boost_round_cv = int(params.get("num_boost_round_cv", 500))
        nfold = int(params.get("nfold", 3))

        df_tune = df[split.train_mask].sample(frac=tune_frac, random_state=seed)
        study, _ = tune_lightgbm_optuna(
            df_tune,
            features=features,
            y_col="sales",
            cat_feats=cat_feats,
            seed=seed,
            n_trials=n_trials,
            nfold=nfold,
            num_boost_round=num_boost_round_cv,
            free_raw_data=False,
            return_tune_data=True,
        )
        best_params = dict(study.best_params)
    else:
        best_params = dict(params.get("fixed_params", {}))

    print("Training final LightGBM model...")
    model, final_params = train_lightgbm_with_best_params(
        train_data=train_data,
        valid_data=valid_data,
        best_params=best_params,
        seed=seed,
        objective=str(params.get("objective", "tweedie")),
        metric=str(params.get("metric", "rmse")),
        bin_construct_sample_cnt=int(params.get("bin_construct_sample_cnt", 200000)),
        num_threads=int(params.get("num_threads", -1)),
        verbose=int(params.get("verbose", -1)),
        num_boost_round=int(params.get("num_boost_round", 1500)),
        early_stopping_rounds=int(params.get("early_stopping_rounds", 100)),
        log_evaluation=int(params.get("log_evaluation", 50)),
    )

    model_path = paths.models_dir / "lgbm_global_model.txt"
    save_lgbm_model(model, model_path)

    df_preds, rmse_value = evaluate_lgbm_on_test_mask(
        model=model,
        df=df,
        test_mask=split.test_mask,
        features=features,
        id_col="id",
        date_col="date",
        y_col="sales",
        d_col="d",
    )

    preds_path = paths.preds_dir / "lgbm_global_predictions.csv.gz"
    df_preds.to_csv(preds_path, index=False, compression="gzip")

    metrics = {
        "rmse": float(rmse_value),
        "horizon": horizon,
        "start_day": start_day,
        "buffer": buffer,
    }

    _write_json(paths.run_dir / "metrics_global_lgbm.json", metrics)
    _write_json(paths.run_dir / "params_global_lgbm.json", final_params)

    _print_kv("Global LightGBM complete:", {"rmse": metrics["rmse"], "model": str(model_path), "preds": str(preds_path)})


def _run_global_nbeatsx(*, cfg: dict[str, Any], paths: RunPaths) -> None:
    import pandas as pd

    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

    from walmart_demand_forecasting.datasets.m5_loaders import Paths as M5Paths
    from walmart_demand_forecasting.datasets.m5_loaders import load_global_nf
    from walmart_demand_forecasting.evaluation.metrics import compute_item_level_wrmsse
    from walmart_demand_forecasting.evaluation.split import split_date_for_last_horizon, train_test_split_by_date
    from walmart_demand_forecasting.features.m5_features import add_snap_active
    from walmart_demand_forecasting.models.nbeatsx.pipeline import (
        default_lightning_accelerator,
        fit_neuralforecast,
        make_csv_logger,
        make_nbtx_mql_model,
        make_neuralforecast,
        merge_forecasts_with_actuals,
        predict_neuralforecast,
        reset_logs_dir,
        rmse_on_column,
        save_neuralforecast,
    )

    params = cfg["global"]["nbeatsx"]

    seed = int(params.get("seed", 42))
    _set_reproducibility(seed)

    horizon = int(params["horizon"])
    start_day = int(params["start_day"])
    alias = str(params.get("alias", "NBEATSx_MQLoss_Global"))

    m5_input_dir = Path(cfg["paths"].get("m5_input_dir", "data"))
    m5_paths = M5Paths(input_dir=m5_input_dir)

    print("Loading global NeuralForecast dataset...")
    m5_df = load_global_nf(start_day=None, paths=m5_paths)
    add_snap_active(m5_df, state_col="state_id")

    cols = [
        "unique_id",
        "ds",
        "y",
        "sell_price",
        "is_available",
        "snap_active",
        "is_event",
        "d_int",
    ]
    Y_df = m5_df[cols].copy()

    split_date = split_date_for_last_horizon(Y_df, horizon=horizon, d_int_col="d_int", ds_col="ds")

    print(f"Filtering history < Day {start_day}...")
    Y_trainable = Y_df[Y_df["d_int"] >= start_day].drop(columns=["d_int"])
    train_df, test_df = train_test_split_by_date(Y_trainable, split_date=split_date, ds_col="ds")

    futr_exog_list = list(params.get("futr_exog_list", ["sell_price", "is_available", "snap_active", "is_event"]))
    futr_exog_df = test_df[["unique_id", "ds", *futr_exog_list]].copy()

    log_name = str(params.get("log_name", "nbeats_mqloss_global"))
    log_dir = paths.logs_root / log_name
    reset_logs_dir(str(log_dir))
    logger = make_csv_logger(str(paths.logs_root), name=log_name)

    common_params: dict[str, Any] = {
        "h": horizon,
        "input_size": int(params.get("input_size", 370)),
        "scaler_type": str(params.get("scaler_type", "robust")),
        "max_steps": int(params.get("max_steps", 8000)),
        "val_check_steps": int(params.get("val_check_steps", 100)),
        "early_stop_patience_steps": int(params.get("early_stop_patience_steps", 15)),
        "dropout_prob_theta": float(params.get("dropout_prob_theta", 0.2)),
    }

    # Keep these as explicit keys to match notebook semantics.
    common_params["futr_exog_list"] = futr_exog_list
    common_params["exclude_insample_y"] = bool(params.get("exclude_insample_y", False))
    common_params["accelerator"] = default_lightning_accelerator(prefer_mps=bool(params.get("prefer_mps", True)))
    common_params["devices"] = int(params.get("devices", 1))
    common_params["batch_size"] = int(params.get("batch_size", 512))
    common_params["enable_progress_bar"] = bool(params.get("enable_progress_bar", False))

    model = make_nbtx_mql_model(
        common_params,
        learning_rate=float(params.get("learning_rate", 5e-4)),
        quantiles=tuple(params.get("quantiles", [0.5, 0.8, 0.9])),
        alias=alias,
        logger=logger,
        lr_scheduler=CosineAnnealingWarmRestarts,
        lr_scheduler_kwargs=dict(params.get("lr_scheduler_kwargs", {"T_0": 2000, "T_mult": 4, "eta_min": 1e-6})),
    )

    nf = make_neuralforecast(models=[model], freq=str(params.get("freq", "D")))

    print("Training global N-BEATSx...")
    fit_neuralforecast(nf=nf, df=train_df, val_size=int(params.get("val_size", horizon)))

    model_dir = paths.models_dir / "nbeatsx_global"
    save_neuralforecast(nf=nf, path=str(model_dir), overwrite=True, save_dataset=True)

    forecasts = predict_neuralforecast(nf=nf, futr_df=futr_exog_df)
    results = merge_forecasts_with_actuals(
        forecasts=forecasts,
        actuals=test_df[["unique_id", "ds", "y"]],
        id_col="unique_id",
        ds_col="ds",
        y_col="y",
        how="left",
    )

    pred_col = f"{alias}-median"
    rmse_value = rmse_on_column(df=results, y_col="y", pred_col=pred_col)

    preds_path = paths.preds_dir / "nbeatsx_global_predictions.csv.gz"
    results.to_csv(preds_path, index=False, compression="gzip")

    wrmsse = compute_item_level_wrmsse(
        m5_df=Y_df.drop(columns=["d_int"]),
        score_df=results,
        horizon=horizon,
        pred_cols={"N-BEATSx": pred_col},
        return_weights=False,
    )["N-BEATSx"]

    metrics = {
        "rmse": float(rmse_value),
        "wrmsse_item_level": float(wrmsse),
        "horizon": horizon,
        "start_day": start_day,
        "alias": alias,
        "accelerator": common_params["accelerator"],
    }

    _write_json(paths.run_dir / "metrics_global_nbeatsx.json", metrics)
    _print_kv(
        "Global N-BEATSx complete:",
        {"rmse": metrics["rmse"], "wrmsse_item_level": metrics["wrmsse_item_level"], "model": str(model_dir), "preds": str(preds_path)},
    )


def _run_local_lgbm(*, cfg: dict[str, Any], paths: RunPaths) -> None:
    import numpy as np
    import pandas as pd

    lgb = _import_lightgbm()

    from walmart_demand_forecasting.datasets.m5_loaders import Paths as M5Paths
    from walmart_demand_forecasting.datasets.m5_loaders import load_ca_foods_lgbm
    from walmart_demand_forecasting.evaluation.split import contiguous_day_split
    from walmart_demand_forecasting.features.m5_features import add_cyclical_time_features, add_group_lags, add_group_rolling_means
    from walmart_demand_forecasting.models.lgbm.inference import evaluate_lgbm_on_test_mask
    from walmart_demand_forecasting.models.lgbm.io import save_lgbm_model
    from walmart_demand_forecasting.models.lgbm.training import train_lightgbm_with_best_params
    from walmart_demand_forecasting.models.lgbm.tuning import tune_lightgbm_optuna

    params = cfg["local"]["lgbm"]

    horizon = int(params["horizon"])
    start_day = int(params["start_day"])

    m5_input_dir = Path(cfg["paths"].get("m5_input_dir", "data"))
    m5_paths = M5Paths(input_dir=m5_input_dir)

    seed = int(params.get("seed", 42))
    _set_reproducibility(seed)

    print("Loading local (CA-FOODS) LightGBM dataset...")
    df, cat_feats = load_ca_foods_lgbm(start_day=start_day, paths=m5_paths)

    add_cyclical_time_features(df, wday_col="wday", month_col="month")

    lags = list(params.get("lags", [28, 35, 42, 49]))
    df, lag_cols = add_group_lags(df, group_col="id", value_col="sales", lags=lags, dtype="float32")

    windows = list(params.get("rolling_windows", [7, 28, 49]))
    shift = int(params.get("rolling_shift", 28))
    df, rolling_cols = add_group_rolling_means(
        df,
        group_col="id",
        value_col="sales",
        shift=shift,
        windows=windows,
        dtype="float32",
    )

    df = df.dropna(subset=lag_cols + rolling_cols)
    df["date"] = pd.to_datetime(df["date"])

    split = contiguous_day_split(df, horizon=horizon, d_int_col="d_int")

    drop_cols = ["sales", "d", "date", "wm_yr_wk", "month", "wday"]
    features = [c for c in df.columns if c not in drop_cols]

    # Safety: LightGBM cannot ingest object dtype columns.
    # Only cast/mark categoricals that are actually in the feature matrix.
    obj_in_features = [c for c in features if str(df[c].dtype) == "object"]
    for c in obj_in_features:
        df[c] = df[c].astype("category")
        if c not in cat_feats:
            cat_feats.append(c)

    # LightGBM requires categorical_feature to be present in the data matrix.
    cat_feats = [c for c in cat_feats if c in features]

    train_data = lgb.Dataset(
        df[split.train_mask][features],
        label=df[split.train_mask]["sales"],
        categorical_feature=cat_feats,
        free_raw_data=True,
    )
    valid_data = lgb.Dataset(
        df[split.valid_mask][features],
        label=df[split.valid_mask]["sales"],
        reference=train_data,
        categorical_feature=cat_feats,
        free_raw_data=True,
    )

    tune = bool(params.get("tune", True))
    if tune:
        print("Tuning local LightGBM with Optuna...")
        tune_frac = float(params.get("tune_frac", 0.1))
        n_trials = int(params.get("n_trials", 15))
        num_boost_round_cv = int(params.get("num_boost_round_cv", 500))
        nfold = int(params.get("nfold", 3))
        study_name = str(params.get("study_name", "M5_LGBM_Local"))

        df_tune = df[split.train_mask].sample(frac=tune_frac, random_state=seed)
        study, _ = tune_lightgbm_optuna(
            df_tune,
            features=features,
            y_col="sales",
            cat_feats=cat_feats,
            seed=seed,
            n_trials=n_trials,
            nfold=nfold,
            num_boost_round=num_boost_round_cv,
            study_name=study_name,
            free_raw_data=False,
            return_tune_data=True,
        )
        best_params = dict(study.best_params)
    else:
        best_params = dict(params.get("fixed_params", {}))

    print("Training final local LightGBM model...")
    model, final_params = train_lightgbm_with_best_params(
        train_data=train_data,
        valid_data=valid_data,
        best_params=best_params,
        seed=seed,
        objective=str(params.get("objective", "tweedie")),
        metric=str(params.get("metric", "rmse")),
        num_boost_round=int(params.get("num_boost_round", 1500)),
        early_stopping_rounds=int(params.get("early_stopping_rounds", 100)),
        log_evaluation=int(params.get("log_evaluation", 50)),
    )

    model_path = paths.models_dir / "lgbm_local_model.txt"
    save_lgbm_model(model, model_path)

    df_preds, rmse_value = evaluate_lgbm_on_test_mask(
        model=model,
        df=df,
        test_mask=split.test_mask,
        features=features,
        id_col="id",
        date_col="date",
        y_col="sales",
        d_col="d",
    )

    preds_path = paths.preds_dir / "lgbm_local_predictions.csv.gz"
    df_preds.to_csv(preds_path, index=False, compression="gzip")

    metrics = {"rmse": float(rmse_value), "horizon": horizon, "start_day": start_day}
    _write_json(paths.run_dir / "metrics_local_lgbm.json", metrics)
    _write_json(paths.run_dir / "params_local_lgbm.json", final_params)

    _print_kv("Local LightGBM complete:", {"rmse": metrics["rmse"], "model": str(model_path), "preds": str(preds_path)})


def _run_local_nbeatsx(*, cfg: dict[str, Any], paths: RunPaths) -> None:
    import pandas as pd

    from walmart_demand_forecasting.datasets.m5_loaders import Paths as M5Paths
    from walmart_demand_forecasting.datasets.m5_loaders import load_ca_foods_nf
    from walmart_demand_forecasting.evaluation.metrics import compute_item_level_wrmsse
    from walmart_demand_forecasting.evaluation.split import split_date_for_last_horizon, train_test_split_by_date
    from walmart_demand_forecasting.models.nbeatsx.pipeline import (
        default_lightning_accelerator,
        fit_neuralforecast,
        make_csv_logger,
        make_nbtx_mql_model,
        make_neuralforecast,
        merge_forecasts_with_actuals,
        predict_neuralforecast,
        reset_logs_dir,
        rmse_on_column,
        save_neuralforecast,
    )

    params = cfg["local"]["nbeatsx"]

    seed = int(params.get("seed", 42))
    _set_reproducibility(seed)

    horizon = int(params["horizon"])
    start_day = int(params["start_day"])
    alias = str(params.get("alias", "NBEATSx_MQLoss_Local"))

    m5_input_dir = Path(cfg["paths"].get("m5_input_dir", "data"))
    m5_paths = M5Paths(input_dir=m5_input_dir)

    print("Loading local (CA-FOODS) NeuralForecast dataset...")
    # `load_ca_foods_nf` requires an int start_day; use a very early cutoff for "full history".
    keep_full_history = bool(params.get("keep_full_history", True))
    loader_start_day = 1 if keep_full_history else start_day
    m5_df = load_ca_foods_nf(start_day=loader_start_day, paths=m5_paths)

    cols = ["unique_id", "ds", "y", "sell_price", "is_available", "snap_CA", "is_event", "d_int"]
    Y_df = m5_df[cols].copy()

    split_date = split_date_for_last_horizon(Y_df, horizon=horizon, d_int_col="d_int", ds_col="ds")

    Y_trainable = Y_df[Y_df["d_int"] >= start_day].drop(columns=["d_int"])
    train_df, test_df = train_test_split_by_date(Y_trainable, split_date=split_date, ds_col="ds")

    futr_exog_list = list(params.get("futr_exog_list", ["sell_price", "is_available", "snap_CA", "is_event"]))
    futr_exog_df = test_df[["unique_id", "ds", *futr_exog_list]].copy()

    log_name = str(params.get("log_name", "nbeats_mqloss_local"))
    log_dir = paths.logs_root / log_name
    reset_logs_dir(str(log_dir))
    logger = make_csv_logger(str(paths.logs_root), name=log_name)

    common_params: dict[str, Any] = {
        "h": horizon,
        "input_size": int(params.get("input_size", horizon*13)),
        "scaler_type": str(params.get("scaler_type", "robust")),
        "max_steps": int(params.get("max_steps", 3500)),
        "val_check_steps": int(params.get("val_check_steps", 12)),
        "early_stop_patience_steps": int(params.get("early_stop_patience_steps", 20)),
        "dropout_prob_theta": float(params.get("dropout_prob_theta", 0.2)),
    }

    common_params["futr_exog_list"] = futr_exog_list
    common_params["exclude_insample_y"] = bool(params.get("exclude_insample_y", False))
    common_params["accelerator"] = default_lightning_accelerator(prefer_mps=bool(params.get("prefer_mps", True)))
    common_params["devices"] = int(params.get("devices", 1))
    common_params["batch_size"] = int(params.get("batch_size", 512))
    common_params["enable_progress_bar"] = bool(params.get("enable_progress_bar", True))

    model = make_nbtx_mql_model(
        common_params,
        learning_rate=float(params.get("learning_rate", 5e-4)),
        quantiles=tuple(params.get("quantiles", [0.5, 0.8, 0.9])),
        alias=alias,
        logger=logger,
    )

    nf = make_neuralforecast(models=[model], freq=str(params.get("freq", "D")))

    print("Training local N-BEATSx...")
    fit_neuralforecast(nf=nf, df=train_df, val_size=int(params.get("val_size", horizon)))

    model_dir = paths.models_dir / "nbeatsx_local"
    save_neuralforecast(nf=nf, path=str(model_dir), overwrite=True, save_dataset=True)

    forecasts = predict_neuralforecast(nf=nf, futr_df=futr_exog_df)
    results = merge_forecasts_with_actuals(
        forecasts=forecasts,
        actuals=test_df[["unique_id", "ds", "y"]],
        id_col="unique_id",
        ds_col="ds",
        y_col="y",
        how="left",
    )

    pred_col = f"{alias}-median"
    rmse_value = rmse_on_column(df=results, y_col="y", pred_col=pred_col)

    preds_path = paths.preds_dir / "nbeatsx_local_predictions.csv.gz"
    results.to_csv(preds_path, index=False, compression="gzip")

    # wrmsse = compute_item_level_wrmsse(
    #     m5_df=Y_df.drop(columns=["d_int"]),
    #     score_df=results,
    #     horizon=horizon,
    #     pred_cols={"N-BEATSx": pred_col},
    #     return_weights=False,
    # )["N-BEATSx"]

    metrics = {
        "rmse": float(rmse_value),
        # "wrmsse_item_level": float(wrmsse),
        "horizon": horizon,
        "start_day": start_day,
        "alias": alias,
        "accelerator": common_params["accelerator"],
    }

    _write_json(paths.run_dir / "metrics_local_nbeatsx.json", metrics)
    _print_kv(
        "Local N-BEATSx complete:",
        {"rmse": metrics["rmse"], "model": str(model_dir), "preds": str(preds_path)},
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="wdf",
        description="Terminal-first reproducibility CLI for Walmart Demand Forecasting (M5).",
    )

    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a TOML config file (e.g., configs/global.toml).",
    )
    p.add_argument(
        "--artifacts-dir",
        type=str,
        default="Artifacts",
        help="Artifacts directory (default: Artifacts).",
    )
    p.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run folder name under Artifacts/runs/. Default is timestamped.",
    )
    p.add_argument(
        "--m5-input-dir",
        type=str,
        default=None,
        help="Override M5 input dir (folder containing calendar.csv, sell_prices.csv, sales_train_evaluation.csv).",
    )

    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("global-lgbm", help="Run global LightGBM baseline (Act 2 control)")
    sub.add_parser("global-nbeatsx", help="Run global N-BEATSx (Act 2 challenger)")
    sub.add_parser("local-lgbm", help="Run local (CA-FOODS) LightGBM baseline (Act 1)")
    sub.add_parser("local-nbeatsx", help="Run local (CA-FOODS) N-BEATSx (Act 1)")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    config_path = Path(args.config) if args.config else None
    cfg: dict[str, Any] = {}
    if config_path is not None:
        cfg = _load_toml(config_path)

    # Normalize / defaults
    cfg.setdefault("paths", {})
    if args.m5_input_dir:
        cfg["paths"]["m5_input_dir"] = args.m5_input_dir

    artifacts_dir = Path(args.artifacts_dir)
    run_name = args.run_name or _now_run_id(args.command.replace("_", "-"))
    run_paths = _ensure_run_dirs(artifacts_dir=artifacts_dir, run_name=run_name)

    meta = {
        "run_name": run_name,
        "command": args.command,
        "utc": datetime.utcnow().isoformat() + "Z",
        "cwd": os.getcwd(),
        "git_sha": _maybe_git_sha(),
        "config": str(config_path) if config_path else None,
    }
    _write_json(run_paths.run_dir / "run_meta.json", meta)

    # Persist the merged config for traceability
    _write_json(run_paths.run_dir / "config_resolved.json", cfg)

    if args.command == "global-lgbm":
        _run_global_lgbm(cfg=cfg, paths=run_paths)
    elif args.command == "global-nbeatsx":
        _run_global_nbeatsx(cfg=cfg, paths=run_paths)
    elif args.command == "local-lgbm":
        _run_local_lgbm(cfg=cfg, paths=run_paths)
    elif args.command == "local-nbeatsx":
        _run_local_nbeatsx(cfg=cfg, paths=run_paths)
    else:  # pragma: no cover
        raise ValueError(f"Unknown command: {args.command}")

    print(f"\nRun artifacts: {run_paths.run_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
