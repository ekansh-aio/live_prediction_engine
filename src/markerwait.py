from __future__ import annotations

import logging
import pathlib

import pandas as pd

from .config import FEATURE_FILES

log = logging.getLogger("markerwait")

COVERAGE = 0.80
ROUTE_CAP_H = 120.0


def load(path=None) -> pd.DataFrame | None:
    p = pathlib.Path(path or FEATURE_FILES["route_marker_wait"])
    if not p.exists():
        log.warning("Marker-wait table not found: %s — marker wait will be 0h", p)
        return None
    df = pd.read_csv(p) if p.suffix == ".csv" else pd.read_parquet(p)
    df.columns = [str(c).lower() for c in df.columns]
    if "expected_h" not in df.columns and "med_dwell_h" in df.columns:
        df["expected_h"] = df.get("stop_rate", 1.0) * df["med_dwell_h"]
    return df


def shortlist(map_df: pd.DataFrame, service_type, start_port_id, end_port_id,
              coverage: float = COVERAGE) -> pd.DataFrame:
    m = map_df[(pd.to_numeric(map_df["service_type"], errors="coerce") == pd.to_numeric(service_type, errors="coerce"))
               & (map_df["start_port_id"].astype(str) == str(start_port_id))
               & (map_df["end_port_id"].astype(str) == str(end_port_id))].copy()
    if not len(m):
        return m
    m = m.sort_values("expected_h", ascending=False)
    m["cum_share"] = m["expected_h"].cumsum() / m["expected_h"].sum()
    k = int((m["cum_share"] <= coverage).sum()) + 1
    return m.iloc[:k].drop(columns="cum_share")


def leg_remaining_wait(service_type, start_port_id, end_port_id,
                       remaining_h: float, elapsed_h: float,
                       map_df: pd.DataFrame | None = None,
                       coverage: float = COVERAGE,
                       route_cap_h: float = ROUTE_CAP_H) -> float:
    remaining_h = max(float(remaining_h or 0.0), 0.0)
    elapsed_h = max(float(elapsed_h or 0.0), 0.0)
    if remaining_h <= 0.0:
        return 0.0
    if map_df is None:
        map_df = load()
    if map_df is None:
        return 0.0
    s = shortlist(map_df, service_type, start_port_id, end_port_id, coverage)
    if not len(s):
        return 0.0
    total = min(float(s["expected_h"].sum()), route_cap_h)
    return total * (remaining_h / (remaining_h + elapsed_h))


def voyage_wait(legs: pd.DataFrame, map_df: pd.DataFrame | None = None,
                coverage: float = COVERAGE, route_cap_h: float = ROUTE_CAP_H) -> float:
    if map_df is None:
        map_df = load()
    if map_df is None:
        return 0.0
    legs = legs.rename(columns={c: c.lower() for c in legs.columns})
    total = 0.0
    for _, r in legs.iterrows():
        s = shortlist(map_df, r["service_type"], r["start_port_id"], r["end_port_id"], coverage)
        if len(s):
            total += min(float(s["expected_h"].sum()), route_cap_h)
    return total
