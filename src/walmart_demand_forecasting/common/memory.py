from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd


PathLike = Union[str, Path]


def read_and_squeeze(file_name: str, input_dir_path: PathLike = "../data/", verbose: bool = True) -> pd.DataFrame:
    """Read a CSV and downcast numeric columns to reduce memory.

    Notes:
    - This function mirrors the behavior used in the notebooks to keep results consistent.
    - It downcasts ints/floats independently per-column based on observed min/max.

    Args:
        file_name: CSV filename (e.g., "calendar.csv").
        input_dir_path: Directory containing the file.
        verbose: Print memory reduction summary.

    Returns:
        Loaded dataframe with downcast numeric dtypes.
    """

    if not isinstance(file_name, str):
        raise TypeError(f"file_name must be a string, got {type(file_name)!r}")

    input_dir = Path(input_dir_path)
    file_path = input_dir / file_name

    if not file_path.exists():
        raise FileNotFoundError(
            f"Cannot find {file_name!r} under {str(input_dir)!r}. "
            "Check input_dir_path."
        )

    df = pd.read_csv(file_path)

    numerics = {"int16", "int32", "int64", "float16", "float32", "float64", "int8", "float8"}
    start_mem_mb = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype
        if str(col_type) not in numerics:
            continue

        c_min = df[col].min()
        c_max = df[col].max()

        if str(col_type).startswith("int"):
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            else:
                df[col] = df[col].astype(np.int64)
        else:
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                df[col] = df[col].astype(np.float16)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)

    end_mem_mb = df.memory_usage().sum() / 1024**2
    if verbose:
        reduction = 100 * (start_mem_mb - end_mem_mb) / start_mem_mb if start_mem_mb else 0.0
        print(
            f"For {file_name}, mem. usage decreased from {start_mem_mb:5.2f} Mb "
            f"to {end_mem_mb:5.2f} Mb ({reduction:.1f}% reduction)"
        )

    return df
