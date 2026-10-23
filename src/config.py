from __future__ import annotations

import os
from pathlib import Path


_sm = os.environ.get("SM_MODEL_DIR")
if _sm:
    ROOT = Path(_sm)
else:
    ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
SQL_DIR = ROOT / "sql"
SCHEMA_DIR = ROOT / "schemas"


def load_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.split("#", 1)[0].strip())


load_env()


def env(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)


SF_MODE = env("SF_MODE", "daemon")
MODEL_A_TARGET_MODE = env("MODEL_A_TARGET_MODE", "log_ratio")

MODELS_DIR = ASSETS / "models"
MODEL_A_DIR = ASSETS / "model_a"
FEATURES_DIR = ASSETS / "features"
LOOKUPS_DIR = ASSETS / "lookups"

MODEL_FILES = {
    "model_a": MODELS_DIR / "model_a.txt",
    "dynamic_early": MODELS_DIR / "dynamic_model_early.txt",
    "dynamic_late": MODELS_DIR / "dynamic_model_late.txt",
    "dynamic_lists": MODELS_DIR / "dynamic_feature_lists.json",
    "flc_model": MODELS_DIR / "future_leg_correction_model.txt",
    "flc_lists": MODELS_DIR / "future_leg_correction_lists.json",
}

MODEL_A_FILES = {
    "route_patterns": MODEL_A_DIR / "route_patterns_v3.csv",
    "knn_pool": MODEL_A_DIR / "voyage_ais_atl_aggregates_clean.csv",
    "subservice_mmsi_map": MODEL_A_DIR / "subservice_mmsi_map.csv",
    "carrier_map": MODEL_A_DIR / "voyage_carrier_map.csv",
    "voyage_extras": MODEL_A_DIR / "voyage_extras.csv",
    "encoding_priors": MODEL_A_DIR / "encoding_priors.parquet",
}

LOOKUP_FILES = {
    "port_lookup": LOOKUPS_DIR / "port_lookup.parquet",
    "vessel_features": LOOKUPS_DIR / "vessel_features.parquet",
}

FEATURE_FILES = {
    "bias_correction": FEATURES_DIR / "bias_correction.json",
    "route_marker_wait": FEATURES_DIR / "route_marker_wait.csv",
    "route_stoppage": FEATURES_DIR / "route_stoppage_patterns.csv",
    "leg_context": FEATURES_DIR / "leg_context_features.parquet",
}

MYSQL_TABLE_RUNS = env("MYSQL_TABLE_RUNS", "voyage_prediction_runs")
MYSQL_TABLE_LEGS = env("MYSQL_TABLE_LEGS", "voyage_prediction_legs")
