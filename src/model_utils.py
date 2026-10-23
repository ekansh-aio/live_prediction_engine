from __future__ import annotations

from numbers import Number
from typing import Any

import numpy as np
import pandas as pd
import lightgbm as lgb


def _is_numeric_categories(cats: Any) -> bool:
    if cats is None:
        return False
    try:
        return all(isinstance(v, Number) and not isinstance(v, bool) for v in cats)
    except Exception:
        return False


def align_to_booster(
    df: pd.DataFrame,
    booster: lgb.Booster,
    feature_names: list[str] | None = None,
    cat_names: list[str] | None = None,
    extra_numeric_fill: dict[str, float] | None = None,
    extra_cat_fill: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if feature_names is None:
        feature_names = list(booster.feature_name())

    frame = df.copy()
    if frame.index.name is None and len(frame) == 1:
        pass

    cols = {c: frame[c] if c in frame.columns else np.nan for c in feature_names}
    out = pd.DataFrame(cols, index=frame.index)

    pandas_cats = list(getattr(booster, "pandas_categorical", None) or [])
    cat_by_name = {}
    if pandas_cats and len(pandas_cats) == len(feature_names):
        for name, cats in zip(feature_names, pandas_cats):
            if cats is not None and len(cats) > 0:
                cat_by_name[name] = list(cats)
    elif pandas_cats and len(feature_names) >= len(pandas_cats):
        tail_names = feature_names[-len(pandas_cats):]
        for name, cats in zip(tail_names, pandas_cats):
            if cats is not None and len(cats) > 0:
                cat_by_name[name] = list(cats)

    if cat_names:
        for c in cat_names:
            if c in feature_names and c not in cat_by_name:
                cat_by_name[c] = None

    numeric_fill = {"TOW_SIZE": 0.0, "TOTAL_ASSETS": 0.0}
    if extra_numeric_fill:
        numeric_fill.update(extra_numeric_fill)

    for c in feature_names:
        if c in cat_by_name:
            continue
        if c in numeric_fill and out[c].isna().all():
            out[c] = numeric_fill[c]
            continue
        if pd.api.types.is_object_dtype(out[c]) or pd.api.types.is_string_dtype(out[c]):
            try:
                out[c] = pd.to_numeric(out[c], errors="coerce")
            except Exception:
                pass
            continue
        if not pd.api.types.is_numeric_dtype(out[c]):
            try:
                out[c] = pd.to_numeric(out[c], errors="coerce")
            except Exception:
                pass

    vessel_counts = ["TOW_SIZE", "TOTAL_ASSETS"]
    for c in vessel_counts:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    vessel_cont = [
        "TUG_BHP", "TUG_SPEED_LADEN", "TUG_SPEED_UPRIVER", "TUG_SPEED_DOWNRIVER",
        "TUG_LENGTH", "TUG_BREADTH", "TUG_DRAFT", "TOTAL_BARGE_TONNAGE",
        "AVG_BARGE_DRAFT", "TOTAL_DEADWEIGHT", "TOTAL_LOADED_BARRELS",
    ]
    for c in vessel_cont:
        if c in out.columns:
            s = pd.to_numeric(out[c], errors="coerce")
            if s.notna().any():
                out[c] = s.fillna(float(s.median()))
            else:
                out[c] = s

    for c, cats in cat_by_name.items():
        if c not in out.columns:
            continue
        if cats is not None and _is_numeric_categories(cats):
            s = pd.to_numeric(out[c], errors="coerce")
            out[c] = pd.Categorical(s, categories=cats)
        else:
            s = out[c].astype("string")
            if cats is not None and len(cats) > 0 and not _is_numeric_categories(cats):
                out[c] = pd.Categorical(s, categories=list(cats))
            else:
                out[c] = s.astype("category")

    if extra_cat_fill:
        for c, val in extra_cat_fill.items():
            if c in out.columns and isinstance(out[c].dtype, pd.CategoricalDtype):
                cats = list(out[c].cat.categories)
                if val in cats:
                    out[c] = out[c].fillna(val)
                else:
                    out[c] = out[c].cat.add_categories([val]).fillna(val)

    return out
