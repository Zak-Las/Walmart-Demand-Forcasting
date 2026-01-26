from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """Root mean squared error."""

    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))


def merge_on_id_date(
    left: pd.DataFrame,
    right: pd.DataFrame,
    id_col: str = "unique_id",
    ds_col: str = "ds",
    how: str = "left",
) -> pd.DataFrame:
    """Small helper to standardize forecast/actual merges."""

    return left.merge(right, on=[id_col, ds_col], how=how)


def compute_item_level_wrmsse(
    *,
    m5_df: pd.DataFrame,
    score_df: pd.DataFrame,
    pred_cols: dict[str, str],
    horizon: int = 28,
    id_col: str = "unique_id",
    ds_col: str = "ds",
    y_col: str = "y",
    price_col: str = "sell_price",
    availability_col: str = "is_available",
    return_weights: bool = False,
) -> dict[str, float] | tuple[dict[str, float], pd.Series]:
    """Compute item-level (Level 12) WRMSSE for one or more prediction columns.

    This mirrors the notebook's approach:
    - Scaling factor per item = RMSE of naive forecast on training history.
    - Dollar weights = revenue share over the last `horizon` days of training history.
    - WRMSSE = sum_i (weights_i * RMSSE_i).

    Parameters
    ----------
    m5_df:
        Full (or sufficiently long) history with at least columns:
        [id_col, ds_col, y_col, price_col, availability_col].
    score_df:
        Evaluation set with at least columns: [id_col, y_col] and each prediction column.
    pred_cols:
        Mapping from a label (e.g. "N-BEATSx") to the prediction column name in score_df.

    Returns
    -------
    dict[str, float]
        Mapping from model label to WRMSSE.
    """

    if horizon <= 0:
        raise ValueError("horizon must be positive")

    # Ensure ds is datetime for comparisons.
    m5_ds = pd.to_datetime(m5_df[ds_col])
    score_ds_present = ds_col in score_df.columns
    if score_ds_present:
        _ = pd.to_datetime(score_df[ds_col])

    m5_sorted = m5_df.sort_values([id_col, ds_col])

    # A) Scaling factors: naive one-step RMSE per item, computed on training history.
    train_end = pd.to_datetime(m5_ds.max()) - pd.Timedelta(days=horizon)
    train_hist = m5_sorted.loc[pd.to_datetime(m5_sorted[ds_col]) <= train_end].copy()
    train_hist = train_hist.loc[train_hist[availability_col] != 0].copy()

    train_hist["diff_sq"] = train_hist.groupby(id_col)[y_col].diff().pow(2)
    scaling_factors = train_hist.groupby(id_col)["diff_sq"].mean().pow(0.5)
    scaling_factors = scaling_factors.replace(0, np.nan)

    # B) Dollar weights: revenue share over the last `horizon` days of training history.
    last_train_end = pd.to_datetime(pd.to_datetime(train_hist[ds_col]).max())
    last_train_start = last_train_end - pd.Timedelta(days=horizon)
    last_train_window = train_hist.loc[pd.to_datetime(train_hist[ds_col]) > last_train_start].copy()

    last_train_window["revenue"] = last_train_window[y_col] * last_train_window[price_col]
    total_revenue = float(last_train_window["revenue"].sum())
    if total_revenue <= 0:
        raise ValueError("total_revenue is non-positive; cannot compute WRMSSE weights")

    item_revenue = last_train_window.groupby(id_col)["revenue"].sum()
    weights = item_revenue / total_revenue

    # C) RMSSE + WRMSSE per model.
    out: dict[str, float] = {}
    for label, pred_col in pred_cols.items():
        if pred_col not in score_df.columns:
            raise KeyError(f"Missing prediction column in score_df: {pred_col}")

        tmp = score_df[[id_col, y_col, pred_col]].copy()
        tmp["sq_err"] = (tmp[y_col] - tmp[pred_col]) ** 2
        mse = tmp.groupby(id_col)["sq_err"].mean()
        rmse_series = mse.pow(0.5)

        rmsse = rmse_series / scaling_factors
        wrmsse = float((rmsse * weights).sum())
        out[label] = wrmsse

    if return_weights:
        return out, weights
    return out
