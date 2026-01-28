from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

from walmart_demand_forecasting.common.memory import PathLike, read_and_squeeze


REQUIRED_M5_FILES = (
    "calendar.csv",
    "sales_train_evaluation.csv",
    "sell_prices.csv",
)


def default_m5_input_dir() -> Path:
    """Default to the repo-root `data/` folder.

    Works whether called from notebooks, scripts, or an interactive session.
    """

    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent / "data"
    return Path("data")


def _missing_files(input_dir: Path, required_files: Iterable[str]) -> list[str]:
    return [name for name in required_files if not (input_dir / name).exists()]


def _require_m5_files(paths: "Paths") -> None:
    input_dir = Path(paths.input_dir)
    missing = _missing_files(input_dir, REQUIRED_M5_FILES)
    if not missing:
        return

    missing_str = ", ".join(missing)
    raise FileNotFoundError(
        "Missing M5 files required by the dataset loaders.\n"
        f"Expected under: {str(input_dir.resolve())}\n"
        f"Missing: {missing_str}\n\n"
        "Fix: run `make download_m5` (or `make verify_m5`) from the repo root, "
        "or pass `Paths(input_dir=...)` pointing to the folder containing the CSVs."
    )


@dataclass(frozen=True)
class Paths:
    input_dir: PathLike = field(default_factory=default_m5_input_dir)
    sales_file: str = "sales_train_evaluation.csv"
    calendar_file: str = "calendar.csv"
    prices_file: str = "sell_prices.csv"


def _add_d_int(df: pd.DataFrame, d_col: str = "d", out_col: str = "d_int") -> pd.DataFrame:
    df[out_col] = df[d_col].apply(lambda x: int(x.split("_")[1]))
    return df


def load_ca_foods_lgbm(start_day: int = 1000, paths: Paths = Paths()) -> tuple[pd.DataFrame, list[str]]:
    """Load the CA-FOODS subset in the same shape used by the LightGBM notebook."""

    _require_m5_files(paths)

    df_sales = read_and_squeeze(paths.sales_file, paths.input_dir)
    df_cal = read_and_squeeze(paths.calendar_file, paths.input_dir)
    df_prices = read_and_squeeze(paths.prices_file, paths.input_dir)

    df_sales = df_sales[(df_sales["state_id"] == "CA") & (df_sales["cat_id"] == "FOODS")]

    id_vars = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    df = pd.melt(df_sales, id_vars=id_vars, var_name="d", value_name="sales")

    cal_cols = [
        "d",
        "date",
        "wday",
        "month",
        "year",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
        "wm_yr_wk",
        "snap_CA",
    ]
    df = df.merge(df_cal[cal_cols], on="d", how="left")
    df = df.merge(df_prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

    event_cols = ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]
    for col in event_cols:
        df[col] = df[col].astype("object").fillna("NoEvent")

    df = _add_d_int(df, "d", "d_int")
    df = df[df["d_int"] >= start_day]

    cat_feats = ["id", "item_id", "dept_id", "store_id", "snap_CA"] + event_cols
    for col in cat_feats:
        df[col] = df[col].astype("category")

    return df, cat_feats


def load_ca_foods_nf(start_day: int = 1000, paths: Paths = Paths()) -> pd.DataFrame:
    """Load the CA-FOODS subset in the shape used by the NeuralForecast notebook.

    Returns a dataframe with columns expected by NeuralForecast:
    unique_id, ds, y plus exogenous variables.
    """

    _require_m5_files(paths)

    df_sales = read_and_squeeze(paths.sales_file, paths.input_dir)
    df_cal = read_and_squeeze(paths.calendar_file, paths.input_dir)
    df_prices = read_and_squeeze(paths.prices_file, paths.input_dir)

    df_sales = df_sales[(df_sales["state_id"] == "CA") & (df_sales["cat_id"] == "FOODS")]

    id_vars = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    df = pd.melt(df_sales, id_vars=id_vars, var_name="d", value_name="y")

    df = df.merge(df_cal[["d", "date", "wm_yr_wk", "event_name_1", "snap_CA"]], on="d", how="left")
    df = df.merge(df_prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

    df["ds"] = pd.to_datetime(df["date"])
    df = _add_d_int(df, "d", "d_int")
    df = df[df["d_int"] >= start_day]

    df = df.rename(columns={"id": "unique_id"})
    df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    df["is_available"] = (~df["sell_price"].isna()).astype(int)
    df["sell_price"] = df["sell_price"].astype(float)
    df["sell_price"] = df.groupby("unique_id")["sell_price"].transform(lambda x: x.ffill().bfill())

    df["event_name_1"] = df["event_name_1"].fillna("None")
    df["is_event"] = (df["event_name_1"] != "None").astype(int)
    df["snap_CA"] = df["snap_CA"].fillna(0).astype(int)

    return df[["unique_id", "ds", "y", "sell_price", "is_available", "snap_CA", "is_event", "d_int"]]


def load_global_lgbm(start_day: int = 1200, buffer: int = 60, paths: Paths = Paths()) -> pd.DataFrame:
    """Load the global panel for the LightGBM baseline, keeping a lag buffer."""

    _require_m5_files(paths)

    df_sales = read_and_squeeze(paths.sales_file, paths.input_dir)
    id_vars = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    df = pd.melt(df_sales, id_vars=id_vars, var_name="d", value_name="sales")

    df = _add_d_int(df, "d", "d_int")
    df = df[df["d_int"] >= (start_day - buffer)]
    df["sales"] = df["sales"].astype("float32")

    df_cal = read_and_squeeze(paths.calendar_file, paths.input_dir)
    df_cal = _add_d_int(df_cal, "d", "d_int")
    df_cal = df_cal[df_cal["d_int"] >= (start_day - buffer)]

    cal_cols = [
        "d",
        "date",
        "wday",
        "month",
        "year",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
        "wm_yr_wk",
        "snap_CA",
        "snap_TX",
        "snap_WI",
    ]
    df_cal = df_cal[cal_cols]

    for col in ["snap_CA", "snap_TX", "snap_WI"]:
        df_cal[col] = df_cal[col].fillna(0).astype("int8")

    df = df.merge(df_cal, on="d", how="left")

    df_prices = read_and_squeeze(paths.prices_file, paths.input_dir)
    df = df.merge(df_prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

    df["sell_price"] = df["sell_price"].astype("float32")
    df["sell_price"] = df.groupby("id")["sell_price"].transform(lambda x: x.ffill().bfill())

    return df


def load_global_nf(start_day: int | None = 1200, paths: Paths = Paths()) -> pd.DataFrame:
    """Load the global panel for NeuralForecast N-BEATSx (minimal exogenous set).

    If start_day is None, no day-based filtering is applied.
    """

    _require_m5_files(paths)

    df_cal = read_and_squeeze(paths.calendar_file, paths.input_dir)
    df_cal["date"] = pd.to_datetime(df_cal["date"])
    df_cal = _add_d_int(df_cal, "d", "d_int")

    df_cal = df_cal[["d", "date", "wm_yr_wk", "event_name_1", "snap_CA", "snap_TX", "snap_WI", "d_int"]]
    for col in ["snap_CA", "snap_TX", "snap_WI"]:
        df_cal[col] = df_cal[col].fillna(0).astype("int8")

    df_sales = read_and_squeeze(paths.sales_file, paths.input_dir)
    id_vars = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    df = pd.melt(df_sales, id_vars=id_vars, var_name="d", value_name="y")
    df = _add_d_int(df, "d", "d_int")

    df = df.merge(df_cal.drop(columns=["d_int"]), on="d", how="left")
    df = df.drop(columns=["d"])

    df_prices = read_and_squeeze(paths.prices_file, paths.input_dir)
    df = df.merge(df_prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

    df = df.rename(columns={"id": "unique_id", "date": "ds"})
    df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    df["is_available"] = (~df["sell_price"].isna()).astype("int8")
    df["sell_price"] = df["sell_price"].astype("float32")
    df["sell_price"] = df.groupby("unique_id")["sell_price"].transform(lambda x: x.ffill().bfill())

    df["event_name_1"] = df["event_name_1"].fillna("None")
    df["is_event"] = (df["event_name_1"] != "None").astype("int8")

    if start_day is not None:
        df = df[df["d_int"] >= start_day]

    return df


__all__ = [
    "Paths",
    "default_m5_input_dir",
    "load_ca_foods_lgbm",
    "load_ca_foods_nf",
    "load_global_lgbm",
    "load_global_nf",
]
