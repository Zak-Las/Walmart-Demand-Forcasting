from __future__ import annotations

from typing import Any

import pandas as pd

from walmart_demand_forecasting.evaluation.metrics import rmse


def predict_lgbm_on_test_mask(
    *,
    model: Any,
    df: pd.DataFrame,
    test_mask: pd.Series,
    features: list[str],
    id_col: str = "id",
    date_col: str = "date",
    y_col: str = "sales",
    d_col: str = "d",
) -> pd.DataFrame:
    """Create a standard prediction dataframe for a LightGBM panel split.

    Returns columns: unique_id, ds, y, lgbm_pred (+ d if present).
    """

    cols = [id_col, date_col, y_col]
    if d_col in df.columns:
        cols.insert(1, d_col)

    out = df.loc[test_mask, cols].copy()

    rename_map = {id_col: "unique_id", date_col: "ds", y_col: "y"}
    out = out.rename(columns=rename_map)
    out["ds"] = pd.to_datetime(out["ds"])

    preds = model.predict(df.loc[test_mask, features])
    out["lgbm_pred"] = preds

    return out


def evaluate_lgbm_on_test_mask(
    *,
    model: Any,
    df: pd.DataFrame,
    test_mask: pd.Series,
    features: list[str],
    id_col: str = "id",
    date_col: str = "date",
    y_col: str = "sales",
    d_col: str = "d",
) -> tuple[pd.DataFrame, float]:
    """Predict + compute RMSE for the test mask.

    Returns
    -------
    df_preds, rmse_value
    """

    df_preds = predict_lgbm_on_test_mask(
        model=model,
        df=df,
        test_mask=test_mask,
        features=features,
        id_col=id_col,
        date_col=date_col,
        y_col=y_col,
        d_col=d_col,
    )

    rmse_value = rmse(df_preds["y"], df_preds["lgbm_pred"])
    return df_preds, rmse_value
