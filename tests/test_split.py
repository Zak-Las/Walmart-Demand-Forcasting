import pandas as pd

from walmart_demand_forecasting.evaluation.split import (
    contiguous_day_split,
    split_date_for_last_horizon,
    train_test_split_by_date,
)


def test_contiguous_day_split_partitions_without_overlap() -> None:
    df = pd.DataFrame({"d_int": list(range(1, 11))})
    split = contiguous_day_split(df, horizon=2, d_int_col="d_int")

    assert split.max_d == 10
    assert split.valid_start == 6
    assert split.test_start == 8

    train = split.train_mask
    valid = split.valid_mask
    test = split.test_mask

    assert int(train.sum()) == 6
    assert int(valid.sum()) == 2
    assert int(test.sum()) == 2

    assert not (train & valid).any()
    assert not (train & test).any()
    assert not (valid & test).any()

    assert (train | valid | test).all()


def test_split_date_for_last_horizon_returns_expected_date() -> None:
    ds = pd.date_range("2020-01-01", periods=10, freq="D")
    df = pd.DataFrame({"d_int": list(range(1, 11)), "ds": ds})

    # max_d=10, horizon=2 => split_day=8 => ds[7]
    split_date = split_date_for_last_horizon(df, horizon=2, d_int_col="d_int", ds_col="ds")
    assert split_date == pd.Timestamp("2020-01-08")


def test_train_test_split_by_date_is_inclusive_exclusive() -> None:
    ds = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    df = pd.DataFrame({"ds": ds, "value": [1, 2, 3]})

    train_df, test_df = train_test_split_by_date(df, split_date=pd.Timestamp("2020-01-02"), ds_col="ds")

    assert train_df["ds"].max() == pd.Timestamp("2020-01-02")
    assert test_df["ds"].min() == pd.Timestamp("2020-01-03")
