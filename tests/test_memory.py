from __future__ import annotations

import numpy as np
import pandas as pd

from walmart_demand_forecasting.common.memory import read_and_squeeze


def test_read_and_squeeze_downcasts_numeric_columns(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "int_col": [0, 1, 2, 3, 4, 5],
            "float_col": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "text": ["a", "b", "c", "d", "e", "f"],
        }
    )

    csv_path = tmp_path / "tiny.csv"
    df.to_csv(csv_path, index=False)

    out = read_and_squeeze("tiny.csv", input_dir_path=tmp_path, verbose=False)

    assert out["int_col"].dtype == np.int8
    assert out["float_col"].dtype in (np.float16, np.float32)
    assert out["text"].dtype == object


def test_read_and_squeeze_requires_string_filename(tmp_path) -> None:
    try:
        read_and_squeeze(123, input_dir_path=tmp_path, verbose=False)  # type: ignore[arg-type]
        raise AssertionError("Expected TypeError")
    except TypeError:
        pass


def test_read_and_squeeze_raises_for_missing_file(tmp_path) -> None:
    try:
        read_and_squeeze("does_not_exist.csv", input_dir_path=tmp_path, verbose=False)
        raise AssertionError("Expected FileNotFoundError")
    except FileNotFoundError:
        pass
