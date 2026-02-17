from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ContiguousDaySplit:
    max_d: int
    valid_start: int
    test_start: int
    train_mask: pd.Series
    valid_mask: pd.Series
    test_mask: pd.Series


def contiguous_day_split(df: pd.DataFrame, horizon: int, d_int_col: str = "d_int") -> ContiguousDaySplit:
    """Create contiguous train/valid/test boolean masks from an integer day index.

    Protocol:
    - test = last `horizon` days
    - valid = the `horizon` days before test
    - train = everything before valid

    Matches the logic currently in the notebooks.
    """

    max_d = int(df[d_int_col].max())
    test_start = max_d - int(horizon)
    valid_start = test_start - int(horizon)

    train_mask = df[d_int_col] <= valid_start
    valid_mask = (df[d_int_col] > valid_start) & (df[d_int_col] <= test_start)
    test_mask = df[d_int_col] > test_start

    return ContiguousDaySplit(
        max_d=max_d,
        valid_start=valid_start,
        test_start=test_start,
        train_mask=train_mask,
        valid_mask=valid_mask,
        test_mask=test_mask,
    )


def split_date_for_last_horizon(
    df: pd.DataFrame,
    horizon: int,
    d_int_col: str = "d_int",
    ds_col: str = "ds",
) -> pd.Timestamp:
    """Return the first timestamp at (max_d - horizon).

    This is the split logic used in the NeuralForecast sections.
    """

    max_d = int(df[d_int_col].max())
    split_day = max_d - int(horizon)
    split_date = df.loc[df[d_int_col] == split_day, ds_col].min()
    if pd.isna(split_date):
        raise ValueError(f"split_date is NaN (no rows with {d_int_col} == {split_day}).")
    return pd.to_datetime(split_date)


def train_test_split_by_date(
    df: pd.DataFrame,
    split_date: pd.Timestamp,
    ds_col: str = "ds",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataframe into train/test by a timestamp cutoff (inclusive/exclusive)."""

    split_date = pd.to_datetime(split_date)
    train_df = df[df[ds_col] <= split_date]
    test_df = df[df[ds_col] > split_date]
    return train_df, test_df
