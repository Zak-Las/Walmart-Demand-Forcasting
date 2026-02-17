from __future__ import annotations

import glob
import os
from typing import Iterable, Optional

import pandas as pd


def _default_train_cols() -> tuple[str, ...]:
    return (
        "train_loss_step",
        "train_loss_epoch",
        "train_loss",
    )


def _default_valid_cols() -> tuple[str, ...]:
    return (
        "valid_loss",
        "val_loss",
        "validation_loss",
    )


def _first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def iter_lightning_metrics_csv_paths(log_dir: str, version: str = "*") -> list[str]:
    """Return metrics.csv paths for a Lightning CSVLogger run.

    Expected structure (CSVLogger):
      {log_dir}/version_{n}/metrics.csv

    Args:
        log_dir: e.g. "logs/nbeats_mqloss_local".
        version: version selector, e.g. "0" or "*".

    Returns:
        Sorted list of metrics.csv paths.
    """
    version_dirs = sorted(glob.glob(os.path.join(log_dir, f"version_{version}")))
    paths: list[str] = []
    for v_dir in version_dirs:
        metrics_path = os.path.join(v_dir, "metrics.csv")
        if os.path.exists(metrics_path):
            paths.append(metrics_path)
    return paths


def plot_nf_learning_curves(
    *,
    model_name: str,
    log_dir: str,
    version: str = "*",
    max_versions: int = 3,
    x_col: str = "epoch",
    train_col_candidates: Iterable[str] = _default_train_cols(),
    valid_col_candidates: Iterable[str] = _default_valid_cols(),
    figsize: tuple[int, int] = (16, 6),
    title: Optional[str] = None,
):
    """Plot training/validation loss from Lightning CSVLogger metrics.csv.

    This stays notebook-friendly: it imports matplotlib lazily.

    Args:
        model_name: Display name used in the plot title.
        log_dir: e.g. "logs/nbeats_mqloss_local".
        version: version selector, e.g. "0" or "*".
        max_versions: Plot up to N versions.
        x_col: Usually "epoch".
        train_col_candidates: Candidate column names for train loss.
        valid_col_candidates: Candidate column names for val loss.
        figsize: Matplotlib figure size.
        title: Optional override title.

    Returns:
        The created matplotlib figure.
    """
    import matplotlib.pyplot as plt

    metrics_paths = iter_lightning_metrics_csv_paths(log_dir=log_dir, version=version)

    if not metrics_paths:
        print(f"WARNING: No metrics.csv found under {log_dir}/version_{version}")
        return None

    fig = plt.figure(figsize=figsize)

    for metrics_path in metrics_paths[:max_versions]:
        m_df = pd.read_csv(metrics_path)

        if x_col not in m_df.columns:
            raise ValueError(f"Expected column '{x_col}' in {metrics_path}. Found: {list(m_df.columns)}")

        train_col = _first_existing_column(m_df, train_col_candidates)
        valid_col = _first_existing_column(m_df, valid_col_candidates)

        if train_col is None and valid_col is None:
            raise ValueError(
                f"No train/valid loss columns found in {metrics_path}. "
                f"Tried train={list(train_col_candidates)} valid={list(valid_col_candidates)}"
            )

        if train_col is not None:
            train_df = m_df[[x_col, train_col]].dropna()
            plt.plot(train_df[x_col], train_df[train_col], label=f"Train ({train_col})", alpha=0.5)

        if valid_col is not None:
            valid_df = m_df[[x_col, valid_col]].dropna()
            plt.plot(valid_df[x_col], valid_df[valid_col], label=f"Valid ({valid_col})", color="red", linewidth=1.5)

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()

    if title is None:
        title = f"Training Dynamics: {model_name}"
    plt.title(title)

    plt.tight_layout()
    plt.show()
    return fig
