import numpy as np
import pandas as pd
import pytest

from walmart_demand_forecasting.evaluation.metrics import (
    compute_item_level_wrmsse,
    merge_on_id_date,
    rmse,
)


def test_rmse_non_negative_and_zero_for_perfect_prediction() -> None:
    y = np.array([0.0, 1.0, 2.0])
    assert rmse(y, y) == 0.0
    assert rmse(y, y + 1.0) >= 0.0


def test_rmse_improves_when_predictions_improve() -> None:
    y = np.array([0.0, 1.0, 2.0, 3.0])
    preds_bad = y + 10.0
    preds_better = y + 1.0
    assert rmse(y, preds_better) < rmse(y, preds_bad)


def test_merge_on_id_date_keeps_alignment() -> None:
    left = pd.DataFrame(
        {
            "unique_id": ["A", "A", "B"],
            "ds": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-01"]),
            "y": [1.0, 2.0, 3.0],
        }
    )
    right = pd.DataFrame(
        {
            "unique_id": ["A", "B"],
            "ds": pd.to_datetime(["2020-01-02", "2020-01-01"]),
            "pred": [20.0, 30.0],
        }
    )

    merged = merge_on_id_date(left, right, id_col="unique_id", ds_col="ds", how="left")
    assert len(merged) == len(left)
    # A@2020-01-02 and B@2020-01-01 should match.
    assert merged.loc[(merged["unique_id"] == "A") & (merged["ds"] == pd.Timestamp("2020-01-02")), "pred"].iloc[0] == 20.0
    assert merged.loc[(merged["unique_id"] == "B") & (merged["ds"] == pd.Timestamp("2020-01-01")), "pred"].iloc[0] == 30.0


def test_compute_item_level_wrmsse_properties() -> None:
    # Tiny synthetic history for two series; horizon kept small for the test.
    ds = pd.date_range("2020-01-01", periods=6, freq="D")

    m5_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "unique_id": "A",
                    "ds": ds,
                    "y": [1, 2, 3, 4, 5, 6],
                    "sell_price": 1.0,
                    "is_available": 1,
                }
            ),
            pd.DataFrame(
                {
                    "unique_id": "B",
                    "ds": ds,
                    "y": [2, 1, 2, 1, 2, 1],
                    "sell_price": 2.0,
                    "is_available": 1,
                }
            ),
        ],
        ignore_index=True,
    )

    score_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "unique_id": ["A", "A", "B", "B"],
                    "ds": pd.to_datetime(["2020-01-05", "2020-01-06", "2020-01-05", "2020-01-06"]),
                    "y": [5.0, 6.0, 2.0, 1.0],
                }
            )
        ],
        ignore_index=True,
    )
    score_df["pred_good"] = score_df["y"]
    score_df["pred_bad"] = score_df["y"] + 10.0

    wrmsse, weights = compute_item_level_wrmsse(
        m5_df=m5_df,
        score_df=score_df,
        pred_cols={"good": "pred_good", "bad": "pred_bad"},
        horizon=2,
        return_weights=True,
    )

    assert set(wrmsse.keys()) == {"good", "bad"}
    assert wrmsse["good"] >= 0.0
    assert wrmsse["bad"] >= 0.0
    assert np.isfinite(wrmsse["good"]) and np.isfinite(wrmsse["bad"])
    assert wrmsse["good"] < wrmsse["bad"]

    # Weights are revenue shares; should sum to ~1.
    assert float(weights.sum()) == pytest.approx(1.0, abs=1e-9)
