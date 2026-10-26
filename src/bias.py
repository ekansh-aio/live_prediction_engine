from __future__ import annotations

import json
import math
import os

from .config import FEATURE_FILES

_TAB = None


def load():
    global _TAB
    if _TAB is not None:
        return _TAB
    _TAB = {}
    if os.environ.get("BIAS_CORR", "1") != "1":
        return _TAB
    path = FEATURE_FILES["bias_correction"]
    if path.exists():
        _TAB = json.loads(path.read_text(encoding="utf-8"))
    return _TAB


def legs_bucket(n_future_legs) -> str:
    try:
        return str(min(int(n_future_legs), 2))
    except Exception:
        return "0"


def apply(remaining_h: float, n_future_legs):
    tab = load()
    raw = max(float(remaining_h or 0.0), 0.0)
    if not tab:
        return raw, 0.0, None, None
    adj = float(tab.get("legs_ahead_adj_h", {}).get(legs_bucket(n_future_legs), 0.0))
    effective_adj = min(adj, raw * 0.50)
    corr = max(raw - effective_adj, 0.0)
    lo, hi = tab.get("logratio_p10"), tab.get("logratio_p90")
    p10 = max(math.exp(lo) * (corr + 1) - 1, 0.0) if lo is not None else None
    p90 = max(math.exp(hi) * (corr + 1) - 1, 0.0) if hi is not None else None
    return corr, adj, p10, p90
