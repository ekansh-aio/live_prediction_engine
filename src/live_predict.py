from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from snowflake_client import SnowflakeClient

from . import assets, config, markerwait, stoppage
from .consolidator_runtime import assemble_voyage
from .dynamic_infer import DynamicPredictor, predict_dynamic
from .flc_infer import FLCInfer
from .model_a_infer import ModelAInfer
from .prediction_store import store_prediction
from .utils import jsonable, parse_dt, uppercase_columns

log = logging.getLogger("live_predict")


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


def predict_voyage_live(voyage_id: int, store: bool = True) -> dict:
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
        log.info("Building Model A features/predictions")
        ma_input = pd.concat([pd.DataFrame([current_leg]), future_legs], ignore_index=True)
        ma_scores, ma_feature_map = model_a.predict(ma_input)
        log.info("Model A done (%d predictions)", len(ma_scores))
        ma_by_id = {int(k): v for k, v in ma_scores.set_index("ID")["predicted_duration_h"].to_dict().items()}

        log.info("Running Dynamic prediction")
        dynamic_remaining_h, dynamic_row, dynamic_features = predict_dynamic(current_leg, ais_df, statuslog_df, now_ts, dynamic, voyage_df=voyage_df)
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
    voyage_df = classify_leg_status(voyage_df).sort_values(["VOYAGE_ID", "SEQUENCE_NUM"]).reset_index(drop=True)

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

    ma_input = pd.concat([pd.DataFrame([current_leg]), future_legs], ignore_index=True)
    ma_scores, ma_feature_map = model_a.predict(ma_input)
    ma_by_id = {int(k): v for k, v in ma_scores.set_index("ID")["predicted_duration_h"].to_dict().items()}

    dynamic_remaining_h, dynamic_row, dynamic_features = predict_dynamic(
        current_leg, ais_df, statuslog_df, snapshot_ts, dynamic, voyage_df=voyage_df
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
