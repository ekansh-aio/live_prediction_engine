from __future__ import annotations

import math
import re
from datetime import date, datetime

import numpy as np
import pandas as pd


def parse_dt(value):
    return pd.to_datetime(value, format="ISO8601", utc=True, errors="coerce")


def interval_to_hours(val):
    if pd.isna(val) or val is None:
        return np.nan
    s = str(val).strip()
    try:
        return pd.to_timedelta(s).total_seconds() / 3600
    except Exception:
        pass
    m = re.match(r"(?:(\d+)\s*days?,?\s*)?(\d+):(\d+):(\d+)", s)
    if m:
        d, h, mn, sc = (int(x) if x else 0 for x in m.groups())
        return d * 24 + h + mn / 60 + sc / 3600
    try:
        return float(s)
    except Exception:
        return np.nan


def jsonable(value):
    if value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.integer, np.bool_)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if not math.isfinite(float(value)):
            return None
        return float(value)
    if pd.isna(value):
        return None
    return value


def uppercase_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).upper() for c in out.columns]
    return out
