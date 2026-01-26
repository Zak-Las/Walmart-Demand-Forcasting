from __future__ import annotations

from pathlib import Path
from typing import Any


def save_lgbm_model(model: Any, model_path: str | Path) -> Path:
    """Save a LightGBM Booster to disk, creating parent dirs if needed."""

    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    return path


def load_lgbm_model(model_path: str | Path) -> Any:
    """Load a LightGBM Booster from disk."""

    import lightgbm as lgb

    path = Path(model_path)
    return lgb.Booster(model_file=str(path))
