from __future__ import annotations

import pandas as pd

from .config import FEATURE_FILES


def load() -> pd.DataFrame | None:
    path = FEATURE_FILES["route_stoppage"]
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = [str(c).lower() for c in df.columns]
    return df


def estimate(patterns: pd.DataFrame | None, service_type, start_port_id,
             end_port_id, kind: str = "junction", default: float = 0.0) -> float:
    if patterns is None or len(patterns) == 0:
        return float(default)
    col = {"junction": "junc_median_h", "leg": "leg_stop_median",
           "crawl": "crawl_stop_median", "total": "total_stop_median_h"}.get(kind, "junc_median_h")
    svc = pd.to_numeric(service_type, errors="coerce")
    m = patterns[(pd.to_numeric(patterns["service_type"], errors="coerce") == svc)
                 & (patterns["start_port_id"].astype(str) == str(start_port_id))
                 & (patterns["end_port_id"].astype(str) == str(end_port_id))]
    if len(m) and pd.notna(m[col].iloc[0]):
        return float(m[col].iloc[0])
    m0 = patterns[pd.to_numeric(patterns["service_type"], errors="coerce") == svc]
    if len(m0) and pd.notna(m0[col].median()):
        return float(m0[col].median())
    return float(default)


def junction_estimate(patterns, service_type, start_port_id, end_port_id, default=0.0):
    return estimate(patterns, service_type, start_port_id, end_port_id, kind="junction", default=default)
