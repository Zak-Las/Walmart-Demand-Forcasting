from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from walmart_demand_forecasting.evaluation.metrics import rmse


def default_lightning_accelerator(*, prefer_mps: bool = True) -> str:
    """Return a sensible PyTorch Lightning accelerator for the current machine.

    Intended for laptop-friendly reproducibility:
    - Uses Apple Silicon MPS when available (if prefer_mps=True)
    - Otherwise uses CUDA when available
    - Falls back to CPU

    This keeps training code portable across macOS (MPS), Linux/Windows (CUDA), and CPU-only.
    """

    try:
        import torch
    except Exception:
        return "cpu"

    mps_ok = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
    cuda_ok = torch.cuda.is_available()

    if prefer_mps and mps_ok:
        return "mps"
    if cuda_ok:
        return "cuda"
    if mps_ok:
        return "mps"
    return "cpu"


def reset_logs_dir(log_dir: str) -> None:
    """Delete a Lightning log directory if it exists.

    Notebook-friendly helper to keep training curves reproducible.
    """

    import os
    import shutil

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)


def make_csv_logger(log_root: str, *, name: str) -> Any:
    """Create a PyTorch Lightning CSVLogger."""

    from pytorch_lightning.loggers import CSVLogger

    return CSVLogger(log_root, name=name)


def make_nbtx_loss_comparison_models(
    common_params: dict[str, Any],
    *,
    mql_learning_rate: float = 1e-4,
    quantiles: Iterable[float] = (0.5, 0.8, 0.9),
    alias_rmse: str = "NBEATSx_RMSE",
    alias_huber: str = "NBEATSx_HUBER",
    alias_mql: str = "NBEATSx_MQL",
) -> list[Any]:
    """Create the 3-model NBEATSx lineup used in the local notebook (RMSE/Huber/MQLoss)."""

    from neuralforecast.models import NBEATSx
    from neuralforecast.losses.pytorch import HuberLoss, MQLoss, RMSE

    return [
        NBEATSx(**common_params, loss=RMSE(), alias=alias_rmse),
        NBEATSx(**common_params, loss=HuberLoss(), alias=alias_huber),
        NBEATSx(
            **common_params,
            learning_rate=mql_learning_rate,
            loss=MQLoss(quantiles=list(quantiles)),
            alias=alias_mql,
        ),
    ]


def make_nbtx_mql_model(
    common_params: dict[str, Any],
    *,
    learning_rate: float,
    quantiles: Iterable[float] = (0.5, 0.8, 0.9),
    alias: str = "NBEATSx_MQL",
    logger: Any | None = None,
    lr_scheduler: Any | None = None,
    lr_scheduler_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Create a single NBEATSx model trained with MQLoss."""

    from neuralforecast.models import NBEATSx
    from neuralforecast.losses.pytorch import MQLoss

    kwargs: dict[str, Any] = dict(common_params)
    kwargs.update(
        {
            "learning_rate": learning_rate,
            "loss": MQLoss(quantiles=list(quantiles)),
            "alias": alias,
        }
    )

    if logger is not None:
        kwargs["logger"] = logger
    if lr_scheduler is not None:
        kwargs["lr_scheduler"] = lr_scheduler
    if lr_scheduler_kwargs is not None:
        kwargs["lr_scheduler_kwargs"] = lr_scheduler_kwargs

    return NBEATSx(**kwargs)


def make_neuralforecast(models: list[Any], *, freq: str = "D") -> Any:
    """Construct a NeuralForecast wrapper."""

    from neuralforecast import NeuralForecast

    return NeuralForecast(models=models, freq=freq)


def cross_validate_neuralforecast(
    *,
    models: list[Any],
    df: pd.DataFrame,
    val_size: int,
    n_windows: int,
    step_size: int,
    freq: str = "D",
) -> pd.DataFrame:
    """Run NeuralForecast cross-validation and return the CV results dataframe."""

    nf = make_neuralforecast(models, freq=freq)
    return nf.cross_validation(
        df=df,
        val_size=val_size,
        n_windows=n_windows,
        step_size=step_size,
    )


def fit_neuralforecast(*, nf: Any, df: pd.DataFrame, val_size: int) -> Any:
    """Fit a NeuralForecast model (single split)."""

    nf.fit(df=df, val_size=val_size)
    return nf


def save_neuralforecast(
    *,
    nf: Any,
    path: str,
    overwrite: bool = True,
    save_dataset: bool = True,
) -> None:
    """Save a NeuralForecast model to disk."""

    nf.save(path=path, overwrite=overwrite, save_dataset=save_dataset)


def load_neuralforecast(*, path: str) -> Any:
    """Load a NeuralForecast model from disk."""

    from neuralforecast import NeuralForecast

    return NeuralForecast.load(path=path)


def predict_neuralforecast(*, nf: Any, futr_df: pd.DataFrame) -> pd.DataFrame:
    """Run NeuralForecast prediction."""

    return nf.predict(futr_df=futr_df)


def merge_forecasts_with_actuals(
    *,
    forecasts: pd.DataFrame,
    actuals: pd.DataFrame,
    id_col: str = "unique_id",
    ds_col: str = "ds",
    y_col: str = "y",
    how: str = "left",
) -> pd.DataFrame:
    """Merge forecast outputs with ground truth actuals."""

    out = forecasts.merge(
        actuals[[id_col, ds_col, y_col]],
        on=[id_col, ds_col],
        how=how,
    )

    out[ds_col] = pd.to_datetime(out[ds_col])
    return out


def rmse_on_column(*, df: pd.DataFrame, y_col: str, pred_col: str) -> float:
    """Compute RMSE for a given prediction column."""

    return float(rmse(df[y_col], df[pred_col]))
