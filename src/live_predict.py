from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import assets, config, markerwait, stoppage
from .consolidator_runtime import assemble_voyage
from .dynamic_infer import DynamicPredictor, predict_dynamic
from .flc_infer import FLCInfer
from .model_a_infer import ModelAInfer
from .prediction_store import load_latest, store_prediction
from .utils import jsonable, parse_dt, uppercase_columns

log = logging.getLogger("live_predict")

# Stop-type combos that mark a "target" subservice leg and reset the context block.
# Mirrors build_leg_context_features.py in Opentug-Phase2.
_CONTEXT_TARGETS = {(1, 1), (3, 1), (1, 2)}
_CTX_INTERCHANGE_START = 6
_CTX_INTERCHANGE_END = 6
_CTX_OFF_HIRE_END = 4
_CTX_FUEL_OPS_END = (7, 9, 10)
_CTX_HIRE_TRANSIT = (3, 3)


def classify_leg_status(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Use TIME_STARTED as the "leg has departed" signal; fall back to ETA_UPDATED
    # if TIME_STARTED is not in the result set (older schema).
    started = out["TIME_STARTED"].notna() if "TIME_STARTED" in out.columns else out["ETA_UPDATED"].notna()
    conds = [
        started & out["TIME_COMPLETED"].notna(),
        started & out["TIME_COMPLETED"].isna(),
        ~started & out["TIME_COMPLETED"].isna(),
        ~started & out["TIME_COMPLETED"].notna(),
    ]
    out["LEG_STATUS"] = pd.Series(pd.Categorical(np.select(conds, ["completed", "in-progress", "future", "not-eta"], default="unknown")))
    return out


def _read_sql(name: str) -> str:
    return (config.SQL_DIR / name).read_text(encoding="utf-8")


def _aggregate_context_spine(pending: list, anchor_e) -> dict:
    """Aggregate preceding non-target legs into context features (no AIS)."""
    n = len(pending)
    ctx_inter = sum(1 for L in pending if L["S"] == _CTX_INTERCHANGE_START or L["E6"] == _CTX_INTERCHANGE_END)
    ctx_off   = sum(1 for L in pending if L["E6"] == _CTX_OFF_HIRE_END)
    ctx_fuel  = sum(1 for L in pending if L["E6"] in _CTX_FUEL_OPS_END)
    ctx_hire  = sum(1 for L in pending if L["S"] == _CTX_HIRE_TRANSIT[0] and L["E6"] == _CTX_HIRE_TRANSIT[1])
    comp = [L for L in pending if pd.notna(L["C"]) and pd.notna(anchor_e) and L["C"] <= anchor_e]

    total_dist = sum(L["DIST"] for L in pending if pd.notna(L["DIST"]))
    spds = [L["SPD"] for L in pending if pd.notna(L["SPD"])]
    p_hrs = [
        L["DIST"] / L["SPD"] if (pd.notna(L["DIST"]) and pd.notna(L["SPD"]) and L["SPD"] > 0) else np.nan
        for L in pending
    ]
    p_hrs = [h for h in p_hrs if not np.isnan(h)]

    actual_total = np.nan
    gap_h = np.nan
    if comp and pd.notna(anchor_e):
        durations = [(L["C"] - L["E"]).total_seconds() / 3600.0 for L in comp if pd.notna(L["E"])]
        if durations:
            actual_total = sum(durations)
        last_end = max(L["C"] for L in comp)
        gap_h = max(0.0, (anchor_e - last_end).total_seconds() / 3600.0)

    return {
        "preceding_non_target_count": n,
        "had_interchange":      int(ctx_inter > 0),
        "had_off_hire":         int(ctx_off > 0),
        "had_fuel_or_ops":      int(ctx_fuel > 0),
        "had_hire_transit":     int(ctx_hire > 0),
        "ctx_interchange_count": ctx_inter,
        "ctx_completed_count":  len(comp),
        "ctx_total_planned_dist_nm": round(total_dist, 2) if total_dist else np.nan,
        "ctx_avg_planned_speed":     round(float(np.mean(spds)), 2) if spds else np.nan,
        "ctx_total_planned_h":       round(float(np.sum(p_hrs)), 2) if p_hrs else np.nan,
        "ctx_max_planned_h":         round(float(np.max(p_hrs)), 2) if p_hrs else np.nan,
        "ctx_last_ctx_planned_h":    round(float(p_hrs[-1]), 2) if p_hrs else np.nan,
        "ctx_actual_total_h":        round(actual_total, 2) if pd.notna(actual_total) else np.nan,
        "ctx_gap_h_to_target_e":     round(gap_h, 2) if pd.notna(gap_h) else np.nan,
    }


def compute_leg_context_from_spine(voyage_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-leg context features from the voyage spine.

    Uses the same aggregation logic as build_leg_context_features.py but derives
    everything from voyage_legs (already in the payload). AIS-based context stats
    (ctx_ping_count, ctx_avg_sog, etc.) are NaN — the model handles them as missing.
    """
    df = voyage_df.copy()
    df = df.sort_values(["VOYAGE_ID", "SEQUENCE_NUM"]).reset_index(drop=True)

    def _stop(val):
        try:
            return int(pd.to_numeric(val, errors="coerce"))
        except (TypeError, ValueError):
            return -1

    rows = []
    pending: list = []
    cur_vid = None

    for _, r in df.iterrows():
        vid = r.get("VOYAGE_ID")
        if vid != cur_vid:
            cur_vid = vid
            pending = []

        ae = r.get("ETA_UPDATED")
        feats = _aggregate_context_spine(pending, ae)
        feats["subservice_id"] = int(r["ID"])
        for c in ["ctx_ping_count", "ctx_avg_sog", "ctx_pct_moving", "ctx_zero_speed_share", "ctx_avg_barges"]:
            feats[c] = np.nan
        feats["has_context"] = int(feats["preceding_non_target_count"] > 0)
        rows.append(feats)

        s_type = _stop(r.get("START_STOP_TYPE"))
        e_type = _stop(r.get("END_STOP_TYPE"))
        is_target = (
            (s_type, e_type) in _CONTEXT_TARGETS
            and pd.notna(r.get("ETA_UPDATED"))
            and pd.notna(r.get("TIME_COMPLETED"))
        )
        if is_target:
            pending = []
        else:
            pending.append({
                "E": r.get("ETA_UPDATED"),
                "C": r.get("TIME_COMPLETED"),
                "S": s_type,
                "E6": e_type,
                "DIST": pd.to_numeric(r.get("ESTIMATED_DISTANCE"), errors="coerce"),
                "SPD": pd.to_numeric(r.get("ESTIMATED_SPEED"), errors="coerce"),
            })

    return pd.DataFrame(rows)


def fetch_vessel_features(client, leg_ids: list) -> pd.DataFrame:
    """Fetch vessel/barge composition features from Snowflake for the given leg IDs."""
    if not leg_ids:
        return pd.DataFrame()
    ids_str = ",".join(str(int(i)) for i in leg_ids)
    sql = _read_sql("vessel_features.sql").replace("{leg_ids}", ids_str)
    return client.query_text(sql)


def _departure_anchor_iso(leg_row: pd.Series) -> str | None:
    for col in ("TIME_STARTED", "ETA_UPDATED"):
        val = leg_row.get(col)
        if pd.notna(val):
            return val.isoformat() if hasattr(val, "isoformat") else str(val)
    return None


def fetch_live_voyage_spine(client: SnowflakeClient, voyage_id: int) -> pd.DataFrame:
    df = client.query_text(_read_sql("live_voyage_spine.sql"), {"voyage_id": voyage_id})
    df = uppercase_columns(df)
    for c in ["TIME_STARTED", "ETA_UPDATED", "TIME_COMPLETED", "INITIAL_ETA", "TIME_CREATED",
              "EARLIEST_DEPARTURE", "LATEST_DEPARTURE", "EARLIEST_ARRIVAL", "LATEST_ARRIVAL"]:
        if c in df.columns:
            df[c] = parse_dt(df[c])
    if len(df) == 0:
        raise ValueError(f"Voyage {voyage_id} not found")
    return classify_leg_status(df).sort_values(["VOYAGE_ID", "SEQUENCE_NUM"]).reset_index(drop=True)


def fetch_live_leg_ais(client: SnowflakeClient, current_leg_row: pd.Series) -> pd.DataFrame:
    return uppercase_columns(client.query_text(_read_sql("live_leg_ais.sql"), {
        "leg_id": int(current_leg_row["ID"]),
        "departure_anchor": _departure_anchor_iso(current_leg_row),
    }))


def fetch_live_leg_statuslog(client: SnowflakeClient, current_leg_row: pd.Series) -> pd.DataFrame:
    return uppercase_columns(client.query_text(_read_sql("live_leg_statuslog.sql"), {
        "leg_id": int(current_leg_row["ID"]),
        "departure_anchor": _departure_anchor_iso(current_leg_row),
    }))


def select_current_leg(voyage_df: pd.DataFrame) -> pd.Series:
    current = voyage_df[voyage_df["LEG_STATUS"] == "in-progress"]
    if len(current) == 0:
        raise ValueError("Voyage has no in-progress leg")
    if len(current) > 1:
        log.warning("Multiple in-progress legs (%d); selecting highest SEQUENCE_NUM", len(current))
    return current.sort_values("SEQUENCE_NUM").iloc[-1]


def split_future_legs(voyage_df: pd.DataFrame) -> pd.DataFrame:
    return voyage_df[voyage_df["LEG_STATUS"] == "future"].sort_values("SEQUENCE_NUM")


def _fill_missing_eta_updated(voyage_df: pd.DataFrame, chain_start: pd.Timestamp) -> pd.DataFrame:
    """Set ETA_UPDATED on future legs that are missing it by chaining from chain_start.

    chain_start is snapshot_ts + dynamic_remaining_h of the current leg, i.e. the
    predicted moment the current leg ends and the first future leg begins.
    Subsequent legs advance the chain by each leg's naive_h (dist / speed).
    """
    df = voyage_df.copy()
    ts = chain_start
    for idx in df[df["LEG_STATUS"] == "future"].sort_values("SEQUENCE_NUM").index:
        if pd.isna(df.at[idx, "ETA_UPDATED"]):
            df.at[idx, "ETA_UPDATED"] = ts
            log.info("Chained ETA_UPDATED for future leg_id=%s → %s", df.at[idx, "ID"], ts)
        dist = pd.to_numeric(df.at[idx, "ESTIMATED_DISTANCE"], errors="coerce")
        spd = pd.to_numeric(df.at[idx, "ESTIMATED_SPEED"], errors="coerce")
        if pd.notna(dist) and pd.notna(spd) and spd > 0:
            ts = ts + pd.Timedelta(hours=float(dist / spd))
    return df


def predict_voyage_live(voyage_id: int, store: bool = True) -> dict:
    from snowflake_client import SnowflakeClient  # lazy — not available in SageMaker container
    log.info("Starting prediction for voyage_id=%s (store=%s)", voyage_id, store)
    assets.validate_required_assets()
    log.info("Assets validated")
    client = SnowflakeClient()
    try:
        log.info("Fetching voyage spine")
        voyage_df = fetch_live_voyage_spine(client, voyage_id)
        current_leg = select_current_leg(voyage_df)
        future_legs = split_future_legs(voyage_df)
        log.info(
            "Voyage loaded: %d legs total, current leg_id=%s (seq=%s), %d future legs",
            len(voyage_df),
            current_leg.get("ID"),
            current_leg.get("SEQUENCE_NUM"),
            len(future_legs),
        )
        log.info("Fetching AIS and statuslog for current leg")
        ais_df = fetch_live_leg_ais(client, current_leg)
        statuslog_df = fetch_live_leg_statuslog(client, current_leg)
        log.info("AIS rows=%d, statuslog rows=%d", len(ais_df), len(statuslog_df))
        now_ts = pd.Timestamp.now("UTC")
        log.info("Loading models (Model A, Dynamic, FLC)")
        model_a = ModelAInfer(client=client, leg_ids=voyage_df["ID"].tolist())
        dynamic = DynamicPredictor()
        flc = FLCInfer()
        marker_map = markerwait.load()
        stop_patterns = stoppage.load()
        log.info("Models loaded")
        log.info("Running Dynamic prediction (pass 1 — chain anchor for future legs)")
        dyn_chain_h, _, _ = predict_dynamic(current_leg, ais_df, statuslog_df, now_ts, dynamic, voyage_df=voyage_df)
        voyage_df = _fill_missing_eta_updated(voyage_df, now_ts + pd.Timedelta(hours=dyn_chain_h))
        future_legs = split_future_legs(voyage_df)
        log.info("Building Model A features/predictions")
        ma_input = pd.concat([pd.DataFrame([current_leg]), future_legs], ignore_index=True)
        ma_scores, ma_feature_map = model_a.predict(ma_input)
        log.info("Model A done (%d predictions)", len(ma_scores))
        ma_by_id = {int(k): v for k, v in ma_scores.set_index("ID")["predicted_duration_h"].to_dict().items()}
        log.info("Running Dynamic prediction (pass 2 — with Model A fallback)")
        dynamic_remaining_h, dynamic_row, dynamic_features = predict_dynamic(current_leg, ais_df, statuslog_df, now_ts, dynamic, voyage_df=voyage_df, model_a_pred_h=ma_by_id.get(int(current_leg["ID"])))
        log.info("Dynamic done (remaining_h=%.2f)", dynamic_remaining_h)

        future_predictions = []
        legs = []
        legs.append({
            "leg_id": int(current_leg["ID"]),
            "sequence_num": int(current_leg["SEQUENCE_NUM"]),
            "role": "current",
            "leg_status": str(current_leg["LEG_STATUS"]),
            "model_a_pred_h": float(ma_by_id.get(int(current_leg["ID"]), 0.0)),
            "dynamic_pred_h": float(dynamic_remaining_h),
            "flc_residual_h": None,
            "flc_corrected_h": None,
            "model_a_features": ma_feature_map.get(int(current_leg["ID"])),
            "dynamic_features": dynamic_features,
            "flc_features": None,
        })
        if len(future_legs):
            log.info("Running FLC for %d future legs", len(future_legs))
        for _, future_leg in future_legs.iterrows():
            ma_pred_h = float(ma_by_id.get(int(future_leg["ID"]), 0.0))
            resid_h, corrected_h, flc_features = flc.predict_leg(future_leg, current_leg, dynamic_row, ma_pred_h)
            future_predictions.append({
                "leg_id": int(future_leg["ID"]),
                "model_a_pred_h": ma_pred_h,
                "flc_residual_h": resid_h,
                "corrected_duration_h": corrected_h,
            })
            legs.append({
                "leg_id": int(future_leg["ID"]),
                "sequence_num": int(future_leg["SEQUENCE_NUM"]),
                "role": "future",
                "leg_status": str(future_leg["LEG_STATUS"]),
                "model_a_pred_h": ma_pred_h,
                "dynamic_pred_h": None,
                "flc_residual_h": resid_h,
                "flc_corrected_h": corrected_h,
                "model_a_features": ma_feature_map.get(int(future_leg["ID"])),
                "dynamic_features": None,
                "flc_features": flc_features,
            })
        if len(future_legs):
            log.info("FLC done")

        log.info("Assembling voyage ETA")
        latest_incumbent_eta = None
        if "ETA" in statuslog_df.columns:
            eta_series = pd.to_datetime(statuslog_df["ETA"], utc=True, errors="coerce").dropna()
            if len(eta_series):
                latest_incumbent_eta = eta_series.iloc[-1]
        voyage_result = assemble_voyage(voyage_df, now_ts, dynamic_remaining_h, dynamic_row, future_predictions, stop_patterns, marker_map,
                                        incumbent_eta=latest_incumbent_eta)
        log.info("Voyage ETA assembled")
        result = {
            "voyage_id": int(voyage_id),
            "as_of_utc": now_ts.isoformat(),
            "voyage": voyage_result,
            "legs": legs,
            "asset_status": {
                "markerwait_loaded": marker_map is not None,
                "stoppage_loaded": stop_patterns is not None,
            },
        }
        if store:
            log.info("Storing prediction locally")
            result["run_id"] = store_prediction(result)
            log.info("Stored run_id=%s", result["run_id"])
        log.info("Prediction complete for voyage_id=%s", voyage_id)
        return jsonable(result)
    finally:
        client.close()


def predict_voyage_from_data(payload: dict, *, preloaded: dict | None = None) -> dict:
    """Run the full prediction pipeline from pre-fetched data (no Snowflake).

    Used by the SageMaker inference handler. `preloaded` holds models loaded
    once in model_fn and reused across requests.
    """
    voyage_id = payload["voyage_id"]
    raw_ts = payload.get("snapshot_ts") or pd.Timestamp.now("UTC").isoformat()
    snapshot_ts = pd.Timestamp(raw_ts)
    if snapshot_ts.tzinfo is None:
        snapshot_ts = snapshot_ts.tz_localize("UTC")

    voyage_df = uppercase_columns(pd.DataFrame(payload["voyage_legs"]))
    for c in ["TIME_STARTED", "ETA_UPDATED", "TIME_COMPLETED", "INITIAL_ETA", "TIME_CREATED",
              "EARLIEST_DEPARTURE", "LATEST_DEPARTURE", "EARLIEST_ARRIVAL", "LATEST_ARRIVAL"]:
        if c in voyage_df.columns:
            voyage_df[c] = parse_dt(voyage_df[c])
    # Only reclassify if LEG_STATUS is absent — callers that pre-fetched via
    # fetch_live_voyage_spine already have the correct status in the payload.
    if "LEG_STATUS" not in voyage_df.columns:
        voyage_df = classify_leg_status(voyage_df)
    voyage_df = voyage_df.sort_values(["VOYAGE_ID", "SEQUENCE_NUM"]).reset_index(drop=True)

    ais_df = uppercase_columns(pd.DataFrame(payload.get("current_leg_ais", [])))
    statuslog_df = uppercase_columns(pd.DataFrame(payload.get("current_leg_statuslog", [])))

    assets.validate_required_assets()

    current_leg = select_current_leg(voyage_df)
    future_legs = split_future_legs(voyage_df)

    if preloaded:
        model_a = preloaded["model_a"]
        dynamic = preloaded["dynamic"]
        flc = preloaded["flc"]
        marker_map = preloaded["marker_map"]
        stop_patterns = preloaded["stop_patterns"]
    else:
        model_a = ModelAInfer(client=None, leg_ids=voyage_df["ID"].tolist())
        dynamic = DynamicPredictor()
        flc = FLCInfer()
        marker_map = markerwait.load()
        stop_patterns = stoppage.load()

    # Fresh vessel features from payload (caller fetched them live from Snowflake)
    vessel_features_override = None
    if payload.get("vessel_features"):
        vessel_features_override = uppercase_columns(pd.DataFrame(payload["vessel_features"]))

    dyn_chain_h, _, _ = predict_dynamic(current_leg, ais_df, statuslog_df, snapshot_ts, dynamic, voyage_df=voyage_df)
    voyage_df = _fill_missing_eta_updated(voyage_df, snapshot_ts + pd.Timedelta(hours=dyn_chain_h))
    future_legs = split_future_legs(voyage_df)

    ma_input = pd.concat([pd.DataFrame([current_leg]), future_legs], ignore_index=True)
    ma_scores, ma_feature_map = model_a.predict(
        ma_input,
        vessel_features_override=vessel_features_override,
    )
    ma_by_id = {int(k): v for k, v in ma_scores.set_index("ID")["predicted_duration_h"].to_dict().items()}

    prev_dynamic_h, prev_as_of_utc = None, None
    _prev = load_latest(int(payload["voyage_id"]))
    if _prev:
        _prev_leg = next((l for l in _prev.get("legs", []) if l.get("role") == "current"), None)
        if _prev_leg and _prev_leg.get("dynamic_pred_h") is not None:
            prev_dynamic_h = float(_prev_leg["dynamic_pred_h"])
            prev_as_of_utc = pd.Timestamp(_prev["as_of_utc"])

    dynamic_remaining_h, dynamic_row, dynamic_features = predict_dynamic(
        current_leg, ais_df, statuslog_df, snapshot_ts, dynamic, voyage_df=voyage_df,
        model_a_pred_h=ma_by_id.get(int(current_leg["ID"])),
        prev_dynamic_h=prev_dynamic_h,
        prev_as_of_utc=prev_as_of_utc,
    )

    future_predictions = []
    legs = []
    legs.append({
        "leg_id": int(current_leg["ID"]),
        "sequence_num": int(current_leg["SEQUENCE_NUM"]),
        "role": "current",
        "leg_status": str(current_leg["LEG_STATUS"]),
        "model_a_pred_h": float(ma_by_id.get(int(current_leg["ID"]), 0.0)),
        "dynamic_pred_h": float(dynamic_remaining_h),
        "flc_residual_h": None,
        "flc_corrected_h": None,
        "model_a_features": ma_feature_map.get(int(current_leg["ID"])),
        "dynamic_features": dynamic_features,
        "flc_features": None,
    })
    for _, future_leg in future_legs.iterrows():
        ma_pred_h = float(ma_by_id.get(int(future_leg["ID"]), 0.0))
        resid_h, corrected_h, flc_features = flc.predict_leg(future_leg, current_leg, dynamic_row, ma_pred_h)
        future_predictions.append({
            "leg_id": int(future_leg["ID"]),
            "model_a_pred_h": ma_pred_h,
            "flc_residual_h": resid_h,
            "corrected_duration_h": corrected_h,
        })
        legs.append({
            "leg_id": int(future_leg["ID"]),
            "sequence_num": int(future_leg["SEQUENCE_NUM"]),
            "role": "future",
            "leg_status": str(future_leg["LEG_STATUS"]),
            "model_a_pred_h": ma_pred_h,
            "dynamic_pred_h": None,
            "flc_residual_h": resid_h,
            "flc_corrected_h": corrected_h,
            "model_a_features": ma_feature_map.get(int(future_leg["ID"])),
            "dynamic_features": None,
            "flc_features": flc_features,
        })

    latest_incumbent_eta = None
    if "ETA" in statuslog_df.columns:
        eta_series = pd.to_datetime(statuslog_df["ETA"], utc=True, errors="coerce").dropna()
        if len(eta_series):
            latest_incumbent_eta = eta_series.iloc[-1]

    voyage_result = assemble_voyage(
        voyage_df, snapshot_ts, dynamic_remaining_h, dynamic_row, future_predictions,
        stop_patterns, marker_map, incumbent_eta=latest_incumbent_eta,
    )
    return jsonable({
        "voyage_id": int(voyage_id),
        "as_of_utc": snapshot_ts.isoformat(),
        "voyage": voyage_result,
        "legs": legs,
        "asset_status": {
            "markerwait_loaded": marker_map is not None,
            "stoppage_loaded": stop_patterns is not None,
        },
    })
