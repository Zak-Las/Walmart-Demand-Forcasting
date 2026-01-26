from __future__ import annotations

import numpy as np
import pandas as pd


def add_cyclical_time_features(df: pd.DataFrame, wday_col: str = "wday", month_col: str = "month") -> pd.DataFrame:
    """Add cyclical sin/cos encodings for weekly and yearly cycles.

    This matches the notebook logic:
    - wday assumed 1..7
    - month assumed 1..12

    Returns the same dataframe for chaining.
    """

    df["wday_sin"] = np.sin(2 * np.pi * df[wday_col] / 7).astype("float32")
    df["wday_cos"] = np.cos(2 * np.pi * df[wday_col] / 7).astype("float32")

    df["month_sin"] = np.sin(2 * np.pi * df[month_col] / 12).astype("float32")
    df["month_cos"] = np.cos(2 * np.pi * df[month_col] / 12).astype("float32")

    return df


def add_snap_active(df: pd.DataFrame, state_col: str = "state_id") -> pd.DataFrame:
    """Create a single SNAP flag for a row's state.

    Expects columns snap_CA, snap_TX, snap_WI.
    """

    conditions = [
        df[state_col] == "CA",
        df[state_col] == "TX",
        df[state_col] == "WI",
    ]
    choices = [df["snap_CA"], df["snap_TX"], df["snap_WI"]]
    df["snap_active"] = np.select(conditions, choices, default=0).astype("int8")
    return df


def add_group_lags(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    lags: list[int],
    dtype: str = "float32",
) -> tuple[pd.DataFrame, list[str]]:
    """Add group-wise lag columns using pandas shift."""

    lag_cols: list[str] = []
    for lag in lags:
        col_name = f"lag_{lag}"
        df[col_name] = df.groupby(group_col, observed=True)[value_col].shift(lag).astype(dtype)
        lag_cols.append(col_name)
    return df, lag_cols


def add_group_rolling_means(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    shift: int,
    windows: list[int],
    dtype: str = "float32",
) -> tuple[pd.DataFrame, list[str]]:
    """Add group-wise rolling means of shifted series.

    Example (notebook-equivalent): shift=28, windows=[7, 28]
    """

    rolling_cols: list[str] = []
    for window in windows:
        col_name = f"rolling_mean_{window}"
        df[col_name] = (
            df.groupby(group_col, observed=True)[value_col]
            .transform(lambda x: x.shift(shift).rolling(window).mean())
            .astype(dtype)
        )
        rolling_cols.append(col_name)
    return df, rolling_cols
