"""Prepare local artifacts for Lambda inference testing.

This script rebuilds the NeuralForecast-shaped dataframes used for the global
N-BEATSx model directly from the raw M5 CSVs under `data/`.

Outputs (default under `Artifacts/deployment/`):
- train_history_<N>d.parquet: last N rows per series from the train split
- future_exog_full.parquet: future exogenous rows for the heldout horizon
- test_payload.json: a single-series example payload matching app.py handler
- payload_meta.json: small metadata for debugging/repro

Intended usage:
  python deployment/prepare_deployment.py --config configs/global.toml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        import tomllib  # py311+
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore

    with path.open("rb") as f:
        return tomllib.load(f)


def _df_for_json(df: pd.DataFrame, *, ds_col: str = "ds") -> pd.DataFrame:
    out = df.copy()
    if ds_col in out.columns:
        out[ds_col] = pd.to_datetime(out[ds_col]).dt.strftime("%Y-%m-%d")
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare deployment payload artifacts for NBEATSx Lambda")
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/global.toml"),
        help="Path to configs/global.toml (default: configs/global.toml)",
    )
    p.add_argument(
        "--m5-input-dir",
        type=Path,
        default=None,
        help="Override M5 CSV folder (default: [paths].m5_input_dir from config)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Artifacts/deployment"),
        help="Where to write artifacts (default: Artifacts/deployment)",
    )
    p.add_argument(
        "--history-len",
        type=int,
        default=370,
        help="Number of history rows per series to include in the payload (default: 370)",
    )
    p.add_argument(
        "--sample-id",
        type=str,
        default=None,
        help="Optional unique_id for the example payload (default: first series)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = _load_toml(args.config)

    params = cfg["global"]["nbeatsx"]
    horizon = int(params["horizon"])
    start_day = int(params["start_day"])
    futr_exog_list = list(params.get("futr_exog_list", ["sell_price", "is_available", "snap_active", "is_event"]))

    m5_input_dir = args.m5_input_dir or Path(cfg.get("paths", {}).get("m5_input_dir", "data"))

    from walmart_demand_forecasting.datasets.m5_loaders import Paths as M5Paths
    from walmart_demand_forecasting.datasets.m5_loaders import load_global_nf
    from walmart_demand_forecasting.evaluation.split import split_date_for_last_horizon, train_test_split_by_date
    from walmart_demand_forecasting.features.m5_features import add_snap_active

    print("Loading global NeuralForecast dataset from CSVs...")
    m5_paths = M5Paths(input_dir=m5_input_dir)
    m5_df = load_global_nf(start_day=None, paths=m5_paths)
    add_snap_active(m5_df, state_col="state_id")

    cols = [
        "unique_id",
        "ds",
        "y",
        "sell_price",
        "is_available",
        "snap_active",
        "is_event",
        "d_int",
    ]
    Y_df = m5_df[cols].copy()

    split_date = split_date_for_last_horizon(Y_df, horizon=horizon, d_int_col="d_int", ds_col="ds")
    print(f"Split date (last-horizon cutoff): {pd.to_datetime(split_date).date()}")

    print(f"Filtering history < Day {start_day}...")
    Y_trainable = Y_df[Y_df["d_int"] >= start_day].drop(columns=["d_int"])
    train_df, test_df = train_test_split_by_date(Y_trainable, split_date=split_date, ds_col="ds")

    futr_exog_df = test_df[["unique_id", "ds", *futr_exog_list]].copy()

    history_len = int(args.history_len)
    print(f"Slicing last {history_len} history rows per series...")
    train_df_trimmed = train_df.groupby("unique_id").tail(history_len).reset_index(drop=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history_path = out_dir / f"train_history_{history_len}d.parquet"
    future_path = out_dir / "future_exog_full.parquet"
    meta_path = out_dir / "payload_meta.json"
    payload_path = out_dir / "test_payload.json"

    train_df_trimmed.to_parquet(history_path, index=False)
    futr_exog_df.to_parquet(future_path, index=False)

    if args.sample_id is None:
        sample_id = str(train_df_trimmed["unique_id"].iloc[0])
    else:
        sample_id = str(args.sample_id)

    history_one = train_df_trimmed[train_df_trimmed["unique_id"] == sample_id]
    future_one = futr_exog_df[futr_exog_df["unique_id"] == sample_id]

    if history_one.empty:
        raise SystemExit(f"sample_id not found in trimmed history: {sample_id}")
    if future_one.empty:
        raise SystemExit(f"sample_id not found in future exog (test horizon): {sample_id}")

    payload = {
        "history": _df_for_json(history_one).to_dict(orient="records"),
        "future": _df_for_json(future_one).to_dict(orient="records"),
    }

    payload_path.write_text(json.dumps(payload) + "\n")
    meta = {
        "config": str(args.config),
        "m5_input_dir": str(m5_input_dir),
        "start_day": start_day,
        "horizon": horizon,
        "history_len": history_len,
        "futr_exog_list": futr_exog_list,
        "split_date": str(pd.to_datetime(split_date).date()),
        "sample_id": sample_id,
        "history_rows": int(history_one.shape[0]),
        "future_rows": int(future_one.shape[0]),
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    print("Artifacts generated:")
    print("  -", history_path)
    print("  -", future_path)
    print("  -", payload_path)
    print("  -", meta_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
