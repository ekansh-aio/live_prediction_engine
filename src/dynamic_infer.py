from __future__ import annotations

import json
import logging

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config

log = logging.getLogger("dynamic_infer")
from .model_utils import align_to_booster
from .utils import jsonable


class DynamicPredictor:
    def __init__(self, dynamic_dir=None):
        dynamic_dir = dynamic_dir or config.MODELS_DIR
        cfg = json.loads((dynamic_dir / "dynamic_feature_lists.json").read_text(encoding="utf-8"))
        self.early_model = lgb.Booster(model_file=str(dynamic_dir / "dynamic_model_early.txt"))
        self.late_model = lgb.Booster(model_file=str(dynamic_dir / "dynamic_model_late.txt"))
        self.early_features = cfg["EARLY_FEATURES"]
        self.late_features = cfg["LATE_FEATURES"]
        self.early_cat = cfg["EARLY_CAT"]
        self.late_cat = cfg["LATE_CAT"]
        self.early_best = int(cfg["EARLY_BEST_ITER"])
        self.late_best = int(cfg["LATE_BEST_ITER"])
        self.threshold = float(cfg["SPLIT_THRESHOLD"])

    def _split_key(self, row, progress_pct):
        el = row.get("elapsed_h", np.nan)
        rm = row.get("remaining_h", np.nan)
        if pd.notna(el) and pd.notna(rm) and (el + rm) > 0:
            return el / (el + rm)
        return progress_pct

    def predict_for_row(self, row, progress_pct):
        split_key = self._split_key(row, progress_pct)
        if split_key < self.threshold:
            feats, model, cats, best = self.early_features, self.early_model, self.early_cat, self.early_best
        else:
            feats, model, cats, best = self.late_features, self.late_model, self.late_cat, self.late_best
        X = align_to_booster(pd.DataFrame([row]), model, feature_names=feats, cat_names=cats)
        return max(0.0, float(model.predict(X, num_iteration=best)[0]))


def _safe_float(value, default=np.nan):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _planned_duration_h(distance, speed):
    dist = _safe_float(distance)
    spd = _safe_float(speed)
    if pd.notna(dist) and pd.notna(spd) and spd > 0:
        return dist / spd
    return np.nan


def _leg_delay_h(leg: pd.Series) -> float:
    completed = pd.to_datetime(leg.get("TIME_COMPLETED"), utc=True, errors="coerce")
    anchor = pd.to_datetime(leg.get("ETA_UPDATED"), utc=True, errors="coerce")
    if pd.isna(completed) or pd.isna(anchor):
        return np.nan
    actual_h = max((completed - anchor).total_seconds() / 3600.0, 0.0)
    planned_h = _planned_duration_h(leg.get("ESTIMATED_DISTANCE"), leg.get("ESTIMATED_SPEED"))
    if pd.isna(planned_h):
        return np.nan
    return actual_h - planned_h


def _flags(current_leg: pd.Series) -> dict:
    st = int(_safe_float(current_leg.get("START_STOP_TYPE"), 0.0) or 0)
    et = int(_safe_float(current_leg.get("END_STOP_TYPE"), 0.0) or 0)
    lg = int(_safe_float(current_leg.get("LEG_TYPE"), 0.0) or 0)
    sv = int(_safe_float(current_leg.get("SERVICE_TYPE"), 0.0) or 0)
    return {
        "is_on_hire": int(lg == 1),
        "is_mobilize": int(lg == 2),
        "is_demobilize": int(et == 4),
        "is_loaded_haul": int(st == 1 and et == 2),
        "is_hire_transit": 0,
        "is_pickup_move": int(et == 1),
        "is_off_hire_move": int(et == 4),
        "is_interchange": int(st == 6 or et == 6),
        "is_service_type_2": int(sv == 2),
        "start_is_load": int(st == 1),
        "start_is_on_hire": int(st in (3, 7)),
        "end_is_discharge": int(et == 2),
        "end_is_off_hire": int(et == 4),
        "start_stop_type": st,
        "end_stop_type": et,
        "leg_type": lg,
    }


def _voyage_context(current_leg: pd.Series, voyage_df: pd.DataFrame | None, departure_anchor: pd.Timestamp, cumulative_delay_h: float) -> dict:
    if voyage_df is None or len(voyage_df) == 0:
        seq = _safe_float(current_leg.get("SEQUENCE_NUM"), np.nan)
        return {
            "sequence_num": seq,
            "total_legs_in_voyage": np.nan,
            "is_last_leg": np.nan,
            "prior_leg_delay_h": np.nan,
            "voyage_cumulative_delay_h": cumulative_delay_h,
            "initial_eta_deviation_h": np.nan,
            "is_behind_initial_eta": np.nan,
            "predictability_score": 3.0,
            "has_context": 0.0,
            "preceding_non_target_count": 0.0,
            "had_interchange": 0.0,
            "had_off_hire": 0.0,
            "had_fuel_or_ops": 0.0,
            "had_hire_transit": 0.0,
            "ctx_interchange_count": 0.0,
            "ctx_completed_count": 0.0,
            "ctx_total_planned_dist_nm": np.nan,
            "ctx_avg_planned_speed": np.nan,
            "ctx_total_planned_h": np.nan,
            "ctx_max_planned_h": np.nan,
            "ctx_last_ctx_planned_h": np.nan,
            "ctx_actual_total_h": np.nan,
            "ctx_gap_h_to_target_e": np.nan,
        }

    ordered = voyage_df.sort_values("SEQUENCE_NUM").reset_index(drop=True)
    seq = _safe_float(current_leg.get("SEQUENCE_NUM"), np.nan)
    prior = ordered[ordered["SEQUENCE_NUM"] < seq].copy() if pd.notna(seq) else ordered.iloc[0:0].copy()
    completed_prior = prior[prior.get("LEG_STATUS").astype(str) == "completed"].copy() if "LEG_STATUS" in prior.columns else prior.iloc[0:0].copy()
    non_target = prior.copy()

    prior_delay_h = _leg_delay_h(completed_prior.iloc[-1]) if len(completed_prior) else np.nan
    completed_delays = completed_prior.apply(_leg_delay_h, axis=1) if len(completed_prior) else pd.Series(dtype=float)
    total_completed_delay_h = float(completed_delays.dropna().sum()) if len(completed_delays.dropna()) else 0.0
    voyage_cumulative_delay_h = total_completed_delay_h + (_safe_float(cumulative_delay_h, 0.0) if pd.notna(cumulative_delay_h) else 0.0)

    initial_eta = pd.to_datetime(current_leg.get("INITIAL_ETA"), utc=True, errors="coerce")
    initial_eta_deviation_h = (initial_eta - departure_anchor).total_seconds() / 3600.0 if pd.notna(initial_eta) and pd.notna(departure_anchor) else np.nan

    planned_h = prior.apply(lambda r: _planned_duration_h(r.get("ESTIMATED_DISTANCE"), r.get("ESTIMATED_SPEED")), axis=1) if len(prior) else pd.Series(dtype=float)
    completed_anchors = pd.to_datetime(completed_prior.get("TIME_COMPLETED"), utc=True, errors="coerce") if len(completed_prior) else pd.Series(dtype="datetime64[ns, UTC]")

    return {
        "sequence_num": seq,
        "total_legs_in_voyage": float(len(ordered)),
        "is_last_leg": float(int(pd.notna(seq) and seq == float(ordered["SEQUENCE_NUM"].max()))),
        "prior_leg_delay_h": prior_delay_h,
        "voyage_cumulative_delay_h": voyage_cumulative_delay_h,
        "initial_eta_deviation_h": initial_eta_deviation_h,
        "is_behind_initial_eta": float(int(pd.notna(initial_eta) and pd.notna(departure_anchor) and initial_eta < departure_anchor)),
        "predictability_score": 3.0,
        "has_context": float(int(len(non_target) > 0)),
        "preceding_non_target_count": float(len(non_target)),
        "had_interchange": float(int(((pd.to_numeric(non_target.get("START_STOP_TYPE"), errors="coerce") == 6) | (pd.to_numeric(non_target.get("END_STOP_TYPE"), errors="coerce") == 6)).any())) if len(non_target) else 0.0,
        "had_off_hire": float(int((pd.to_numeric(non_target.get("END_STOP_TYPE"), errors="coerce") == 4).any())) if len(non_target) else 0.0,
        "had_fuel_or_ops": float(int(((pd.to_numeric(non_target.get("START_STOP_TYPE"), errors="coerce") == 9) | (pd.to_numeric(non_target.get("END_STOP_TYPE"), errors="coerce") == 9)).any())) if len(non_target) else 0.0,
        "had_hire_transit": 0.0,
        "ctx_interchange_count": float((((pd.to_numeric(non_target.get("START_STOP_TYPE"), errors="coerce") == 6) | (pd.to_numeric(non_target.get("END_STOP_TYPE"), errors="coerce") == 6)).sum())) if len(non_target) else 0.0,
        "ctx_completed_count": float(len(completed_prior)),
        "ctx_total_planned_dist_nm": float(pd.to_numeric(prior.get("ESTIMATED_DISTANCE"), errors="coerce").sum()) if len(prior) else np.nan,
        "ctx_avg_planned_speed": float(pd.to_numeric(prior.get("ESTIMATED_SPEED"), errors="coerce").mean()) if len(prior) else np.nan,
        "ctx_total_planned_h": float(planned_h.dropna().sum()) if len(planned_h.dropna()) else np.nan,
        "ctx_max_planned_h": float(planned_h.dropna().max()) if len(planned_h.dropna()) else np.nan,
        "ctx_last_ctx_planned_h": float(planned_h.dropna().iloc[-1]) if len(planned_h.dropna()) else np.nan,
        "ctx_actual_total_h": float(((completed_anchors - pd.to_datetime(completed_prior.get("ETA_UPDATED"), utc=True, errors="coerce")).dt.total_seconds() / 3600.0).dropna().sum()) if len(completed_prior) else np.nan,
        "ctx_gap_h_to_target_e": (departure_anchor - completed_anchors.max()).total_seconds() / 3600.0 if len(completed_anchors.dropna()) and pd.notna(departure_anchor) else np.nan,
    }


def _latest_eta_remaining(ais_df: pd.DataFrame, col: str, now_ts: pd.Timestamp, departure_anchor: pd.Timestamp):
    if col not in ais_df.columns or ais_df.empty:
        return np.nan
    vals = pd.to_datetime(ais_df[col], utc=True, errors="coerce").dropna()
    if vals.empty:
        return np.nan
    last_eta = vals.iloc[-1]
    elapsed_h = (now_ts - departure_anchor).total_seconds() / 3600
    return (last_eta - departure_anchor).total_seconds() / 3600 - elapsed_h


def build_dynamic_live_row(current_leg: pd.Series, ais_df: pd.DataFrame, statuslog_df: pd.DataFrame, now_ts: pd.Timestamp, voyage_df: pd.DataFrame | None = None) -> dict:
    ais = ais_df.copy()
    ts_started = pd.to_datetime(current_leg.get("TIME_STARTED"), utc=True, errors="coerce")
    ts_eta_updated = pd.to_datetime(current_leg.get("ETA_UPDATED"), utc=True, errors="coerce")
    departure_anchor = ts_started if pd.notna(ts_started) else ts_eta_updated
    if pd.isna(departure_anchor) and len(ais) > 0:
        _ais_ts = pd.to_datetime(ais.get("TIMESTAMP"), utc=True, errors="coerce").dropna()
        if len(_ais_ts):
            departure_anchor = _ais_ts.min()
            log.warning(
                "leg_id=%s: departure_anchor derived from first AIS ping (%s) — TIME_STARTED and ETA_UPDATED both missing",
                current_leg.get("ID"), departure_anchor,
            )
    planned_speed = _safe_float(current_leg.get("ESTIMATED_SPEED"))
    planned_distance = _safe_float(current_leg.get("ESTIMATED_DISTANCE"))
    planned_h = _planned_duration_h(planned_distance, planned_speed)
    port_pair = f"{current_leg.get('START_PORT_ID')}_{current_leg.get('END_PORT_ID')}"
    flag_values = _flags(current_leg)

    if len(ais) == 0:
        elapsed_h = max((now_ts - departure_anchor).total_seconds() / 3600, 0.0) if pd.notna(departure_anchor) else np.nan
        progress_pct = elapsed_h / planned_h if pd.notna(elapsed_h) and pd.notna(planned_h) and planned_h > 0 else np.nan
        row = {
            "elapsed_h": elapsed_h,
            "progress_pct": progress_pct,
            "remaining_h": np.nan,
            "ping_count_2h": 0,
            "sog_avg_2h": np.nan,
            "sog_std_2h": np.nan,
            "sog_max_2h": np.nan,
            "sog_min_2h": np.nan,
            "sog_moving_avg_2h": np.nan,
            "pct_moving_2h": 0.0,
            "pct_truly_moving_2h": 0.0,
            "avg_barges_2h": np.nan,
            "sog_avg_4h": np.nan,
            "sog_avg_8h": np.nan,
            "sog_avg_16h": np.nan,
            "sog_delta_2h": 0.0,
            "sog_delta_4h": 0.0,
            "sog_delta_8h": 0.0,
            "sog_delta_16h": 0.0,
            "cumulative_delay_h": np.nan,
            "progress_rate_per_h": np.nan,
            "expected_progress_rate": np.nan,
            "progress_rate_ratio": np.nan,
            "prev_remaining_h": np.nan,
            "eta_revision_h": np.nan,
            "naive_remaining_h": np.nan,
            "ais_eta_remaining_h": np.nan,
            "opentug_eta_remaining_h": np.nan,
            "planned_speed": planned_speed,
            "planned_distance": planned_distance,
            "planned_dur_h": planned_h,
            "speed_adj_progress": progress_pct,
            "speed_vs_plan": np.nan,
            "remaining_distance_nm": planned_distance * (1.0 - progress_pct) if pd.notna(planned_distance) and pd.notna(progress_pct) else np.nan,
            "remaining_distance_naive": np.nan,
            "overall_pace_ratio": np.nan,
            "dist_per_hour_actual": np.nan,
            "cumulative_dist_nm": np.nan,
            "eta_incumbent_spread_h": np.nan,
            "eta_incumbent_count": 0.0,
            "eta_incumbent_vs_naive": np.nan,
            "eta_incumbents_agree": np.nan,
            "speed_adjusted_remaining_h": np.nan,
            "is_stopped": np.nan,
            "is_slow": np.nan,
            "consecutive_slow_windows": 0.0,
            "ais_eta_revision_h": np.nan,
            "sog_consistency": np.nan,
            "pace_trend_h": 0.0,
            "stop_rate": np.nan,
            "max_stop_streak": 0.0,
            "recent_vs_early_speed": np.nan,
            "total_stopped_h": 0.0,
            "pct_time_moving": np.nan,
            "vessel_hist_pace_ratio": np.nan,
            "vessel_hist_delay_std": np.nan,
            "vessel_hist_leg_count": np.nan,
            "vessel_route_pace_ratio": np.nan,
            "vessel_route_count": np.nan,
            "sog_vs_vessel_hist": np.nan,
            "pp_hist_remaining_median": np.nan,
            "pp_hist_remaining_mean": np.nan,
            "pp_hist_remaining_count": np.nan,
            "nav_status_mode": np.nan,
            "sl_n_mile_marker": int(statuslog_df["MILE_MARKER_ID"].notna().sum()) if len(statuslog_df) else 0,
            "sl_evt_1": int(pd.to_numeric(statuslog_df.get("EVENT_TYPE"), errors="coerce").eq(1).sum()) if len(statuslog_df) else 0,
            "sl_evt_5": int(pd.to_numeric(statuslog_df.get("EVENT_TYPE"), errors="coerce").eq(5).sum()) if len(statuslog_df) else 0,
            "sl_evt_7": int(pd.to_numeric(statuslog_df.get("EVENT_TYPE"), errors="coerce").eq(7).sum()) if len(statuslog_df) else 0,
            "sl_statuslog_present": int(len(statuslog_df) > 0),
            "ctx_ping_count": 0.0,
            "ctx_avg_sog": np.nan,
            "ctx_pct_moving": np.nan,
            "ctx_zero_speed_share": np.nan,
            "ctx_avg_barges": np.nan,
            "port_pair": port_pair,
        }
        row.update(flag_values)
        row.update(_voyage_context(current_leg, voyage_df, departure_anchor, row.get("cumulative_delay_h", np.nan)))
        return row

    ais["TIMESTAMP"] = pd.to_datetime(ais["TIMESTAMP"], utc=True, errors="coerce")
    ais = ais.dropna(subset=["TIMESTAMP"]).sort_values("TIMESTAMP")
    cutoff_2h = now_ts - pd.Timedelta(hours=2)
    cutoff_4h = now_ts - pd.Timedelta(hours=4)
    cutoff_8h = now_ts - pd.Timedelta(hours=8)
    cutoff_16h = now_ts - pd.Timedelta(hours=16)
    w2 = ais[ais["TIMESTAMP"] >= cutoff_2h]
    w4 = ais[ais["TIMESTAMP"] >= cutoff_4h]
    w8 = ais[ais["TIMESTAMP"] >= cutoff_8h]
    w16 = ais[ais["TIMESTAMP"] >= cutoff_16h]
    speed = pd.to_numeric(ais["SPEED"], errors="coerce")
    speed2 = pd.to_numeric(w2["SPEED"], errors="coerce")
    speed4 = pd.to_numeric(w4["SPEED"], errors="coerce")
    speed8 = pd.to_numeric(w8["SPEED"], errors="coerce")
    speed16 = pd.to_numeric(w16["SPEED"], errors="coerce")
    elapsed_h = max((now_ts - departure_anchor).total_seconds() / 3600, 0.0) if pd.notna(departure_anchor) else np.nan
    progress_pct = elapsed_h / planned_h if pd.notna(planned_h) and planned_h > 0 else np.nan
    sog2 = float(speed2.mean()) if len(speed2.dropna()) else np.nan
    sog4 = float(speed4.mean()) if len(speed4.dropna()) else np.nan
    sog8 = float(speed8.mean()) if len(speed8.dropna()) else np.nan
    sog16 = float(speed16.mean()) if len(speed16.dropna()) else np.nan
    moving2 = float(speed2[speed2 > 0.5].mean()) if len(speed2[speed2 > 0.5]) else np.nan
    remaining_distance_nm = max(planned_distance * (1.0 - progress_pct), 0.0) if pd.notna(planned_distance) and pd.notna(progress_pct) else np.nan
    remaining_naive = (remaining_distance_nm / max(sog8, 0.5)) if pd.notna(remaining_distance_nm) and pd.notna(sog8) else np.nan
    prev_remaining_h = np.nan
    eta_revision_h = np.nan
    if pd.notna(remaining_naive):
        prev_elapsed_h = max(elapsed_h - 2.0, 0.0)
        prev_progress_pct = prev_elapsed_h / planned_h if pd.notna(planned_h) and planned_h > 0 else np.nan
        prev_remaining_h = (max(planned_distance * (1.0 - prev_progress_pct), 0.0) / max(speed16.mean() if len(speed16.dropna()) else speed8.mean(), 0.5)) if pd.notna(prev_progress_pct) else np.nan
        if pd.notna(prev_remaining_h):
            eta_revision_h = (prev_remaining_h - 2.0) - remaining_naive

    evt = pd.to_numeric(statuslog_df.get("EVENT_TYPE"), errors="coerce") if len(statuslog_df) else pd.Series(dtype=float)
    ais_eta_remaining_h = _latest_eta_remaining(ais, "AIS_ETA", now_ts, departure_anchor)
    opentug_eta_remaining_h = _latest_eta_remaining(ais, "OPENTUG_ETA", now_ts, departure_anchor)
    cumulative_dist_nm = min(planned_distance, elapsed_h * moving2) if pd.notna(planned_distance) and pd.notna(elapsed_h) and pd.notna(moving2) else np.nan
    dist_per_hour_actual = (cumulative_dist_nm / elapsed_h) if pd.notna(cumulative_dist_nm) and elapsed_h > 0 else np.nan
    overall_pace_ratio = (dist_per_hour_actual / planned_speed) if pd.notna(dist_per_hour_actual) and pd.notna(planned_speed) and planned_speed > 0 else np.nan
    # When elapsed has overrun the original plan, rebase progress on speed-inferred
    # distance so progress_pct stays ≤ 1 and remaining_distance/naive stay positive.
    if (pd.notna(planned_h) and planned_h > 0
            and pd.notna(elapsed_h) and elapsed_h > planned_h
            and pd.notna(cumulative_dist_nm) and pd.notna(planned_distance) and planned_distance > 0):
        progress_pct = min(cumulative_dist_nm / planned_distance, 0.999)
        remaining_distance_nm = max(planned_distance - cumulative_dist_nm, 0.0)
        remaining_naive = (remaining_distance_nm / max(sog8, 0.5)) if pd.notna(sog8) else np.nan
    speed_adj_progress = (1.0 - remaining_naive / planned_h) if pd.notna(remaining_naive) and pd.notna(planned_h) and planned_h > 0 else progress_pct
    speed_adjusted_remaining_h = max(elapsed_h / progress_pct - elapsed_h, 0.0) if pd.notna(sog2) and sog2 > 0.5 and pd.notna(progress_pct) and progress_pct > 0 else np.nan
    is_stopped = float(int(pd.notna(sog2) and sog2 < 0.5)) if pd.notna(sog2) else np.nan
    is_slow = float(int(pd.notna(sog2) and pd.notna(planned_speed) and planned_speed > 0 and sog2 < planned_speed * 0.7)) if pd.notna(sog2) else np.nan
    pct_moving_2h = float((speed2 > 0.5).mean()) if len(speed2) else 0.0
    total_stopped_h = elapsed_h * (1.0 - pct_moving_2h) if pd.notna(elapsed_h) else np.nan
    initial_eta = pd.to_datetime(current_leg.get("INITIAL_ETA"), utc=True, errors="coerce")
    ais_eta_revision_h = (ais_eta_remaining_h - ((initial_eta - departure_anchor).total_seconds() / 3600.0)) if pd.notna(ais_eta_remaining_h) and pd.notna(initial_eta) and pd.notna(departure_anchor) else np.nan
    row = {
        "elapsed_h": elapsed_h,
        "progress_pct": progress_pct,
        "remaining_h": np.nan,
        "windows_remaining": max(np.ceil((planned_h - elapsed_h) / 2.0), 0.0) if pd.notna(planned_h) else np.nan,
        "ping_count_2h": int(len(w2)),
        "sog_avg_2h": sog2,
        "sog_std_2h": float(speed2.std()) if len(speed2.dropna()) else np.nan,
        "sog_max_2h": float(speed2.max()) if len(speed2.dropna()) else np.nan,
        "sog_min_2h": float(speed2.min()) if len(speed2.dropna()) else np.nan,
        "sog_moving_avg_2h": moving2,
        "pct_moving_2h": pct_moving_2h,
        "pct_truly_moving_2h": float((speed2 > 1.0).mean()) if len(speed2) else 0.0,
        "avg_barges_2h": float(pd.to_numeric(w2.get("INFERRED_NUM_BARGES"), errors="coerce").mean()) if len(w2) else np.nan,
        "nav_status_mode": pd.to_numeric(w2.get("NAV_STATUS_CODE"), errors="coerce").mode().iloc[0] if len(w2) and not pd.to_numeric(w2.get("NAV_STATUS_CODE"), errors="coerce").mode().empty else np.nan,
        "sog_avg_4h": sog4,
        "sog_avg_8h": sog8,
        "sog_avg_16h": sog16,
        "sog_delta_2h": float(sog2 - sog4) if pd.notna(sog2) and pd.notna(sog4) else 0.0,
        "sog_delta_4h": float(sog2 - sog8) if pd.notna(sog2) and pd.notna(sog8) else 0.0,
        "sog_delta_8h": float(sog2 - sog16) if pd.notna(sog2) and pd.notna(sog16) else 0.0,
        "sog_delta_16h": 0.0,
        "cumulative_delay_h": elapsed_h - (progress_pct * planned_h) if pd.notna(progress_pct) and pd.notna(planned_h) else np.nan,
        "progress_rate_per_h": progress_pct / elapsed_h if elapsed_h > 0 and pd.notna(progress_pct) else np.nan,
        "expected_progress_rate": 1.0 / planned_h if pd.notna(planned_h) and planned_h > 0 else np.nan,
        "progress_rate_ratio": ((progress_pct / elapsed_h) / (1.0 / planned_h)) if elapsed_h > 0 and pd.notna(progress_pct) and pd.notna(planned_h) and planned_h > 0 else np.nan,
        "prev_remaining_h": prev_remaining_h,
        "eta_revision_h": eta_revision_h,
        "naive_remaining_h": remaining_naive,
        "ais_eta_remaining_h": ais_eta_remaining_h,
        "opentug_eta_remaining_h": opentug_eta_remaining_h,
        "eta_incumbent_spread_h": 0.0 if pd.notna(ais_eta_remaining_h) else np.nan,
        "eta_incumbent_count": float(int(pd.notna(ais_eta_remaining_h))),
        "eta_incumbent_vs_naive": remaining_naive - ais_eta_remaining_h if pd.notna(remaining_naive) and pd.notna(ais_eta_remaining_h) else np.nan,
        "eta_incumbents_agree": 1.0 if pd.notna(ais_eta_remaining_h) else np.nan,
        "planned_speed": planned_speed,
        "planned_distance": planned_distance,
        "planned_dur_h": planned_h,
        "speed_adj_progress": np.clip(speed_adj_progress, 0.0, 1.0) if pd.notna(speed_adj_progress) else np.nan,
        "speed_vs_plan": (sog2 / planned_speed) if pd.notna(sog2) and pd.notna(planned_speed) and planned_speed > 0 else np.nan,
        "remaining_distance_nm": remaining_distance_nm,
        "remaining_distance_naive": (remaining_distance_nm / sog4) if pd.notna(remaining_distance_nm) and pd.notna(sog4) and sog4 > 0.5 else remaining_naive,
        "overall_pace_ratio": overall_pace_ratio,
        "dist_per_hour_actual": dist_per_hour_actual,
        "cumulative_dist_nm": cumulative_dist_nm,
        "speed_adjusted_remaining_h": speed_adjusted_remaining_h,
        "is_stopped": is_stopped,
        "is_slow": is_slow,
        "consecutive_slow_windows": float(int(is_slow == 1.0)),
        "ais_eta_revision_h": ais_eta_revision_h,
        "sog_consistency": moving2 / (moving2 + float(speed2.std())) if pd.notna(moving2) and pd.notna(float(speed2.std()) if len(speed2.dropna()) else np.nan) else np.nan,
        "pace_trend_h": float(sog2 - sog8) if pd.notna(sog2) and pd.notna(sog8) else 0.0,
        "stop_rate": float(1.0 - pct_moving_2h),
        "max_stop_streak": float(int(is_stopped == 1.0)),
        "recent_vs_early_speed": (sog2 / sog16) if pd.notna(sog2) and pd.notna(sog16) and sog16 > 0 else np.nan,
        "total_stopped_h": total_stopped_h,
        "pct_time_moving": 1.0 - (total_stopped_h / elapsed_h) if pd.notna(total_stopped_h) and elapsed_h > 0 else np.nan,
        "vessel_hist_pace_ratio": np.nan,
        "vessel_hist_delay_std": np.nan,
        "vessel_hist_leg_count": np.nan,
        "vessel_route_pace_ratio": np.nan,
        "vessel_route_count": np.nan,
        "sog_vs_vessel_hist": np.nan,
        "pp_hist_remaining_median": np.nan,
        "pp_hist_remaining_mean": np.nan,
        "pp_hist_remaining_count": np.nan,
        "sl_n_mile_marker": int(statuslog_df["MILE_MARKER_ID"].notna().sum()) if len(statuslog_df) else 0,
        "sl_evt_1": int(evt.eq(1).sum()) if len(evt) else 0,
        "sl_evt_5": int(evt.eq(5).sum()) if len(evt) else 0,
        "sl_evt_7": int(evt.eq(7).sum()) if len(evt) else 0,
        "sl_statuslog_present": int(len(statuslog_df) > 0),
        "ctx_ping_count": float(len(ais)),
        "ctx_avg_sog": float(speed.mean()) if len(speed.dropna()) else np.nan,
        "ctx_pct_moving": float((speed > 0.5).mean()) if len(speed) else np.nan,
        "ctx_zero_speed_share": float((speed.fillna(0) <= 0.1).mean()) if len(speed) else np.nan,
        "ctx_avg_barges": float(pd.to_numeric(ais.get("INFERRED_NUM_BARGES"), errors="coerce").mean()) if len(ais) else np.nan,
        "port_pair": port_pair,
    }
    row.update(flag_values)
    row.update(_voyage_context(current_leg, voyage_df, departure_anchor, row.get("cumulative_delay_h", np.nan)))
    return row


_SMOOTH_ALPHA = 0.7      # weight on previous prediction; higher = more stable ETA
_SMOOTH_MAX_AGE_H = 2.0  # ignore previous prediction older than this


def predict_dynamic(current_leg: pd.Series, ais_df: pd.DataFrame, statuslog_df: pd.DataFrame, now_ts: pd.Timestamp, predictor: DynamicPredictor, voyage_df: pd.DataFrame | None = None, model_a_pred_h: float | None = None, prev_dynamic_h: float | None = None, prev_as_of_utc: pd.Timestamp | None = None):
    row = build_dynamic_live_row(current_leg, ais_df, statuslog_df, now_ts, voyage_df=voyage_df)
    remaining_h = predictor.predict_for_row(row, row.get("progress_pct", np.nan))
    progress_pct = row.get("progress_pct", np.nan)

    if (remaining_h == 0.0
            and (pd.isna(progress_pct) or progress_pct <= 1.0)
            and float(row.get("ctx_ping_count", 0)) > 0
            and model_a_pred_h is not None and pd.notna(model_a_pred_h)):
        elapsed = float(row.get("elapsed_h") or 0.0)
        ma_rem = max(0.0, float(model_a_pred_h) - elapsed)
        if ma_rem > 0:
            log.warning(
                "Dynamic returned 0.0 despite %d AIS pings (progress_pct=%s). "
                "Falling back to Model A remaining (%.2f h).",
                int(row.get("ctx_ping_count", 0)),
                f"{progress_pct:.2f}" if pd.notna(progress_pct) else "NaN",
                ma_rem,
            )
            remaining_h = ma_rem
        else:
            log.warning(
                "Dynamic returned 0.0 despite %d AIS pings; elapsed (%.1fh) > model_a (%.1fh). "
                "Using model_a_pred_h directly as conservative remaining estimate.",
                int(row.get("ctx_ping_count", 0)), elapsed, float(model_a_pred_h),
            )
            remaining_h = float(model_a_pred_h)

    # When elapsed > planned the model is operating out-of-distribution.
    # Fall back to AIS_ETA (vessel's own transponder broadcast) — independent
    # of any Opentug estimate, which this system is intended to replace.
    # If AIS_ETA is also unavailable or expired, fall back to naive_remaining_h.
    if pd.notna(progress_pct) and progress_pct > 1.0:
        ais_rem = row.get("ais_eta_remaining_h", np.nan)
        if pd.notna(ais_rem) and ais_rem > 0:
            remaining_h = float(ais_rem)
        else:
            ma_rem = np.nan
            if model_a_pred_h is not None and pd.notna(model_a_pred_h):
                elapsed = float(row.get("elapsed_h") or 0.0)
                ma_rem = max(0.0, float(model_a_pred_h) - elapsed)

            if pd.notna(ma_rem) and ma_rem > 0:
                log.warning(
                    "Over-elapsed leg (progress_pct=%.1f, elapsed_h=%.1f): "
                    "AIS ETA not applicable. Falling back to Model A remaining (%.2f h).",
                    progress_pct,
                    row.get("elapsed_h", float("nan")),
                    ma_rem,
                )
                remaining_h = ma_rem
            else:
                naive = row.get("naive_remaining_h", np.nan)
                if pd.notna(naive) and float(naive) > 0:
                    log.warning(
                        "Over-elapsed leg (progress_pct=%.1f, elapsed_h=%.1f): "
                        "Model A not applicable. Falling back to naive_remaining_h (%.2fh).",
                        progress_pct, row.get("elapsed_h", float("nan")), float(naive),
                    )
                    remaining_h = float(naive)
                elif model_a_pred_h is not None and pd.notna(model_a_pred_h) and float(model_a_pred_h) > 0:
                    log.warning(
                        "Over-elapsed leg (progress_pct=%.1f, elapsed_h=%.1f): "
                        "All fallbacks give 0. Using model_a_pred_h=%.1fh directly as conservative estimate.",
                        progress_pct, row.get("elapsed_h", float("nan")), float(model_a_pred_h),
                    )
                    remaining_h = float(model_a_pred_h)
                else:
                    log.warning(
                        "Over-elapsed leg (progress_pct=%.1f, elapsed_h=%.1f): "
                        "No fallback available (model_a_pred_h=%s). Returning 0.",
                        progress_pct, row.get("elapsed_h", float("nan")), model_a_pred_h,
                    )
                    remaining_h = 0.0

    # ETA smoothing: blend new prediction with previous, decayed by elapsed time.
    if (prev_dynamic_h is not None and pd.notna(prev_dynamic_h) and float(prev_dynamic_h) > 0
            and prev_as_of_utc is not None and pd.notna(prev_as_of_utc)):
        age_h = (now_ts - prev_as_of_utc).total_seconds() / 3600.0
        if 0 < age_h <= _SMOOTH_MAX_AGE_H:
            prev_decayed = max(0.0, float(prev_dynamic_h) - age_h)
            if prev_decayed > 0:
                smoothed = (1.0 - _SMOOTH_ALPHA) * remaining_h + _SMOOTH_ALPHA * prev_decayed
                log.info(
                    "ETA smooth: raw=%.2fh prev_decayed=%.2fh -> smoothed=%.2fh (age=%.2fh)",
                    remaining_h, prev_decayed, smoothed, age_h,
                )
                remaining_h = smoothed

    if model_a_pred_h is not None and pd.notna(model_a_pred_h) and float(model_a_pred_h) > 0:
        _contextual_cap = float(model_a_pred_h) * config.DYNAMIC_REMAINING_MA_CAP
    else:
        _planned_h = row.get("planned_dur_h", np.nan)
        _contextual_cap = (float(_planned_h) * config.DYNAMIC_REMAINING_PLAN_CAP
                           if pd.notna(_planned_h) and float(_planned_h) > 0 else float("inf"))
    _cap = min(config.DYNAMIC_REMAINING_CAP_H, _contextual_cap)
    if remaining_h > _cap:
        log.warning(
            "remaining_h=%.1f exceeds contextual cap=%.1f "
            "(model_a_pred_h=%s, planned_dur_h=%s). Clamping.",
            remaining_h, _cap, model_a_pred_h, row.get("planned_dur_h"),
        )
        remaining_h = _cap

    row["dynamic_remaining_h"] = remaining_h
    return remaining_h, row, jsonable(row)
