"""Reproducibility metadata: versions, features, preprocessing audit trail."""
from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.utils.config import MODELS_DIR


def _package_versions() -> dict[str, str]:
    packages = (
        "pandas",
        "numpy",
        "scikit-learn",
        "statsmodels",
        "pmdarima",
        "prophet",
        "xgboost",
        "tensorflow",
    )
    versions = {}
    for pkg in packages:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not_installed"
    return versions


def build_run_metadata(
    *,
    best_model: str,
    feature_lists: dict[str, list[str]] | None = None,
    fold_preprocessors: list[dict] | None = None,
    evaluation_mode: str,
    extra: dict | None = None,
) -> dict:
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": _package_versions(),
        "best_model": best_model,
        "evaluation_mode": evaluation_mode,
        "feature_lists": feature_lists or {},
        "fold_preprocessing": fold_preprocessors or [],
    }
    if extra:
        meta.update(extra)
    return meta


def save_run_metadata(metadata: dict, path: Path | None = None) -> Path:
    out = path or (MODELS_DIR / "pipeline_run_metadata.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    return out
