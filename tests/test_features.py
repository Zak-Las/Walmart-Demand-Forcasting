import numpy as np
import pandas as pd

from walmart_demand_forecasting.features.m5_features import (
    add_cyclical_time_features,
    add_group_lags,
    add_group_rolling_means,
    add_snap_active,
)


def test_add_cyclical_time_features_adds_float32_columns() -> None:
    df = pd.DataFrame({"wday": [1, 2, 3], "month": [1, 6, 12]})
    out = add_cyclical_time_features(df, wday_col="wday", month_col="month")

    for col in ["wday_sin", "wday_cos", "month_sin", "month_cos"]:
        assert col in out.columns
        assert out[col].dtype == np.float32
        assert out[col].between(-1.0, 1.0).all()


def test_add_snap_active_picks_state_specific_snap_column() -> None:
    df = pd.DataFrame(
        {
            "state_id": ["CA", "TX", "WI", "ZZ"],
            "snap_CA": [1, 1, 1, 1],
            "snap_TX": [0, 1, 0, 1],
            "snap_WI": [0, 0, 1, 1],
        }
    )

    out = add_snap_active(df, state_col="state_id")
    assert out["snap_active"].tolist() == [1, 1, 1, 0]


def test_add_group_lags_creates_expected_nans_per_group() -> None:
    df = pd.DataFrame(
        {
            "id": ["A", "A", "A", "B", "B"],
            "sales": [1.0, 2.0, 3.0, 10.0, 11.0],
        }
    )

    out, lag_cols = add_group_lags(df, group_col="id", value_col="sales", lags=[1, 2], dtype="float32")
    assert lag_cols == ["lag_1", "lag_2"]

    # First row of each group should have NaN for lag_1.
    first_A = out.index[out["id"] == "A"][0]
    first_B = out.index[out["id"] == "B"][0]
    assert pd.isna(out.loc[first_A, "lag_1"])
    assert pd.isna(out.loc[first_B, "lag_1"])


def test_add_group_rolling_means_matches_manual_example() -> None:
    # One group with known values.
    df = pd.DataFrame({"id": ["A"] * 6, "sales": [1, 2, 3, 4, 5, 6]})

    out, rolling_cols = add_group_rolling_means(
        df,
        group_col="id",
        value_col="sales",
        shift=2,
        windows=[2],
        dtype="float32",
    )

    assert rolling_cols == ["rolling_mean_2"]
    # For sales [1,2,3,4,5,6], shift=2 => [nan,nan,1,2,3,4], rolling(2).mean => [nan,nan,nan,1.5,2.5,3.5]
    assert out["rolling_mean_2"].iloc[-1] == np.float32(3.5)
