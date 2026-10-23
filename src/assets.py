from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from . import config

log = logging.getLogger("assets")


def missing_required_assets() -> list[Path]:
    required = [
        config.MODEL_FILES["model_a"],
        config.MODEL_FILES["dynamic_early"],
        config.MODEL_FILES["dynamic_late"],
        config.MODEL_FILES["dynamic_lists"],
        config.MODEL_FILES["flc_model"],
        config.MODEL_FILES["flc_lists"],
        config.MODEL_A_FILES["route_patterns"],
        config.MODEL_A_FILES["knn_pool"],
    ]
    return [p for p in required if not p.exists()]


def validate_required_assets() -> None:
    missing = missing_required_assets()
    if missing:
        joined = "\n".join(f"- {p}" for p in missing)
        raise FileNotFoundError(f"Missing required live_predict assets:\n{joined}")


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if path.suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format: {path}")


def load_port_lookup(client=None) -> pd.DataFrame:
    path = config.LOOKUP_FILES["port_lookup"]
    if path.exists():
        return read_table(path)
    if client is None:
        raise FileNotFoundError(f"Missing port lookup and no Snowflake client: {path}")
    log.info("Port lookup not found locally; fetching from Snowflake")
    return client.query_file(config.SQL_DIR / "port_lookup.sql")


def load_vessel_features(client, leg_ids) -> pd.DataFrame:
    ids = sorted({int(i) for i in leg_ids})
    if ids and client is not None:
        # Always fetch fresh per-voyage when a client is available — avoids stale
        # parquet entries (e.g. future legs absent from date-filtered bulk snapshot).
        log.info("Fetching vessel features from Snowflake for %d legs", len(ids))
        sql = (config.SQL_DIR / "vessel_features.sql").read_text(encoding="utf-8")
        return client.query_text(sql.replace("{leg_ids}", ",".join(map(str, ids))))
    path = config.LOOKUP_FILES["vessel_features"]
    if path.exists():
        return read_table(path)
    if not ids:
        return pd.DataFrame()
    raise FileNotFoundError(f"Missing vessel features and no Snowflake client: {path}")


def read_optional(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return read_table(path)


def warn_optional(path: Path, label: str) -> None:
    if not path.exists():
        log.warning("Optional asset missing: %s (%s)", label, path)
