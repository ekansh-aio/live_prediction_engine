from __future__ import annotations

import json
import logging
import math

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config

log = logging.getLogger("dynamic_infer")
from .model_utils import align_to_booster
from .utils import jsonable

_R_NM = 3440.065  # Earth radius in nautical miles

def _haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return _R_NM * 2 * math.asin(math.sqrt(max(0.0, min(1.0, a))))

_port_lookup_cache: pd.DataFrame | None = None

def _get_port_lookup() -> pd.DataFrame:
    global _port_lookup_cache
    if _port_lookup_cache is None:
        _port_lookup_cache = pd.read_parquet(config.LOOKUP_FILES["port_lookup"]).set_index("ID")[["LAT", "LON"]]
    return _port_lookup_cache

_vessel_hist_cache: pd.DataFrame | None = None
_vessel_route_cache: pd.DataFrame | None = None

_pp_hist_cache: pd.DataFrame | None = None

def _get_vessel_hist() -> pd.DataFrame:
    global _vessel_hist_cache
    if _vessel_hist_cache is None:
        p = config.LOOKUPS_DIR / "vessel_hist_lookup.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            df.index = df.index.astype(int)
            _vessel_hist_cache = df
        else:
            _vessel_hist_cache = pd.DataFrame()
    return _vessel_hist_cache

def _get_vessel_route() -> pd.DataFrame:
    global _vessel_route_cache
    if _vessel_route_cache is None:
        p = config.LOOKUPS_DIR / "vessel_route_lookup.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            df.index = df.index.set_levels(df.index.levels[0].astype(int), level=0)
            _vessel_route_cache = df
        else:
            _vessel_route_cache = pd.DataFrame()
    return _vessel_route_cache

def _get_pp_hist() -> pd.DataFrame:
    global _pp_hist_cache
    if _pp_hist_cache is None:
        p = config.LOOKUPS_DIR / "pp_hist_lookup.parquet"
        _pp_hist_cache = pd.read_parquet(p) if p.exists() else pd.DataFrame()
    return _pp_hist_cache

_route_tortuosity_cache: pd.DataFrame | None = None

def _get_route_tortuosity() -> pd.DataFrame:
    global _route_tortuosity_cache
    if _route_tortuosity_cache is None:
        p = config.LOOKUPS_DIR / "route_tortuosity_lookup.parquet"
        _route_tortuosity_cache = pd.read_parquet(p) if p.exists() else pd.DataFrame()
    return _route_tortuosity_cache


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
        _raw_pct = elapsed_h / planned_h if pd.notna(elapsed_h) and pd.notna(planned_h) and planned_h > 0 else np.nan
        # Fix 2b (0-ping path): when significantly over-elapsed, progress_pct > 1 is
        # meaningless and sends a "done" signal to the ML model. Null it out.
        if pd.notna(_raw_pct) and pd.notna(planned_h) and planned_h > 0 and _raw_pct > 1.5:
            progress_pct = np.nan
        else:
            progress_pct = _raw_pct
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
            "gps_valid": np.nan,
            "gps_block_reason": "no_ais",
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
    sog_ctx = float(speed.mean()) if len(speed.dropna()) else np.nan
    # Fix 3: prevent sog_ctx=0 (vessel stationary in 96h window) from making cumulative_dist=0.
    # Floor at 15% of planned speed so a temporarily stopped vessel doesn't appear to have
    # made zero progress across a potentially long elapsed period.
    sog_for_dist = sog_ctx
    if pd.notna(sog_ctx) and pd.notna(planned_speed) and planned_speed > 0 and sog_ctx < planned_speed * 0.15:
        sog_for_dist = planned_speed * 0.15
    cumulative_dist_nm = min(planned_distance, elapsed_h * sog_for_dist) if pd.notna(planned_distance) and pd.notna(elapsed_h) and pd.notna(sog_for_dist) else np.nan
    dist_per_hour_actual = (cumulative_dist_nm / elapsed_h) if pd.notna(cumulative_dist_nm) and elapsed_h > 0 else np.nan
    overall_pace_ratio = (dist_per_hour_actual / planned_speed) if pd.notna(dist_per_hour_actual) and pd.notna(planned_speed) and planned_speed > 0 else np.nan
    # When elapsed has overrun the original plan, rebase progress on speed-inferred
    # distance so progress_pct stays ≤ 1 and remaining_distance/naive stay positive.
    if (pd.notna(planned_h) and planned_h > 0
            and pd.notna(elapsed_h) and elapsed_h > planned_h
            and pd.notna(cumulative_dist_nm) and pd.notna(planned_distance) and planned_distance > 0):
        progress_pct = min(cumulative_dist_nm / planned_distance, 0.999)
        remaining_distance_nm = max(planned_distance - cumulative_dist_nm, 0.0)
        if remaining_distance_nm > 0:
            remaining_naive = (remaining_distance_nm / max(sog8, 0.5)) if pd.notna(sog8) else np.nan
        else:
            # Distance extrapolation saturated — don't feed the model a false "0 left" signal.
            remaining_naive = np.nan
        # Fix 2: when significantly over-elapsed AND distance has saturated, the cumulative_dist
        # formula is meaningless (elapsed×sog >> planned_dist). Null out progress_pct so the ML
        # model doesn't receive a permanent "99.9% done" signal for a vessel still days from arrival.
        if (elapsed_h > planned_h * 1.5
                and pd.notna(cumulative_dist_nm) and pd.notna(planned_distance)
                and planned_distance > 0
                and cumulative_dist_nm >= planned_distance * 0.98):
            progress_pct = np.nan
            remaining_naive = np.nan
            speed_adjusted_remaining_h = np.nan  # will be recomputed below but NaN guard is here
    # Fix 2b: same null-out for 0-ping over-elapsed case where cumulative_dist is NaN.
    # Without AIS pings the time-based progress_pct caps at 1.0, which is equally misleading.
    if (pd.notna(planned_h) and planned_h > 0
            and pd.notna(elapsed_h) and elapsed_h > planned_h * 1.5
            and not pd.notna(cumulative_dist_nm)):
        progress_pct = np.nan
        remaining_naive = np.nan

    # GPS Fix: use haversine distance from the vessel's last AIS position to the destination
    # port as a ground-truth remaining distance. This replaces the time-based formula for
    # over-delayed vessels where elapsed×sog saturates/nulls progress_pct (Fix 2/2b above).
    _vessel_lat = np.nan
    _vessel_lon = np.nan
    if len(ais) > 0 and "LAT" in ais.columns and "LON" in ais.columns:
        _last_ping = ais.iloc[-1]
        _vessel_lat = pd.to_numeric(_last_ping.get("LAT"), errors="coerce")
        _vessel_lon = pd.to_numeric(_last_ping.get("LON"), errors="coerce")
    dist_to_dest_nm = np.nan
    _end_port_id = current_leg.get("END_PORT_ID")
    if pd.notna(_vessel_lat) and pd.notna(_vessel_lon) and _end_port_id is not None:
        try:
            _port_row = _get_port_lookup().loc[int(_end_port_id)]
            dist_to_dest_nm = _haversine_nm(
                _vessel_lat, _vessel_lon,
                float(_port_row["LAT"]), float(_port_row["LON"]),
            )
        except (KeyError, ValueError, TypeError):
            pass

    # Save raw haversine before tortuosity — used for AIS plausibility checks below.
    _raw_to_dest_nm = dist_to_dest_nm

    # Distance from last AIS ping to start port — used to detect wrong-direction AIS.
    _raw_to_start_nm = np.nan
    _start_port_id = current_leg.get("START_PORT_ID")
    if pd.notna(_vessel_lat) and pd.notna(_vessel_lon) and _start_port_id is not None:
        try:
            _sp = _get_port_lookup().loc[int(_start_port_id)]
            _raw_to_start_nm = _haversine_nm(_vessel_lat, _vessel_lon, float(_sp["LAT"]), float(_sp["LON"]))
        except (KeyError, ValueError, TypeError):
            pass

    # Per-ping haversine trend to destination (last 12 pings) — detects approaching vs. retreating.
    _recent_to_dest_nm: pd.Series = pd.Series(dtype=float)
    if pd.notna(_raw_to_dest_nm) and len(ais) >= 2 and _end_port_id is not None:
        try:
            _ep = _get_port_lookup().loc[int(_end_port_id)]
            _tail = ais.tail(12)
            _recent_to_dest_nm = pd.Series([
                _haversine_nm(
                    pd.to_numeric(r.get("LAT"), errors="coerce"),
                    pd.to_numeric(r.get("LON"), errors="coerce"),
                    float(_ep["LAT"]), float(_ep["LON"]),
                )
                for _, r in _tail.iterrows()
                if pd.notna(pd.to_numeric(r.get("LAT"), errors="coerce"))
            ]).dropna()
        except (KeyError, ValueError, TypeError):
            pass

    # Apply route tortuosity: haversine is straight-line; real routes (rivers, coast)
    # are longer. Scale dist_to_dest by the route's historical planned/haversine ratio.
    # Cap at planned_distance * 0.95 — tortuosity can't push remaining > full route length.
    if pd.notna(dist_to_dest_nm):
        _rt = _get_route_tortuosity()
        if len(_rt) and port_pair in _rt.index:
            _tortuosity = float(_rt.loc[port_pair, "tortuosity"])
            if pd.notna(_tortuosity) and _tortuosity > 1.0:
                dist_to_dest_nm = dist_to_dest_nm * _tortuosity
        if pd.notna(planned_distance) and planned_distance > 0:
            dist_to_dest_nm = min(dist_to_dest_nm, planned_distance * 0.95)

    _over_delayed = (pd.notna(elapsed_h) and pd.notna(planned_h) and planned_h > 0
                     and elapsed_h > planned_h * 1.5)

    # GPS position plausibility: reject bad/mismatched AIS before applying GPS fix.
    # Each sub-condition is checked separately so we can log the blocking reason.
    # (1) raw haversine < 95% of planned route (not near start or past it)
    # (2) vessel trending toward destination in recent pings (not moving away)
    # (3) last ping near best recent position (no outlier rebound) — may reject noisy river tracks
    # (4) if over-delayed, vessel is materially closer to destination than to start
    _gps_block_reason = ""
    if not (pd.notna(_raw_to_dest_nm) and pd.notna(planned_distance) and planned_distance > 0):
        _gps_block_reason = "no_haversine_or_plan"
    elif _raw_to_dest_nm > planned_distance * 0.95:
        _gps_block_reason = f"haversine_too_large({_raw_to_dest_nm:.0f}nm>={planned_distance:.0f}nm)"
    elif len(_recent_to_dest_nm) >= 2 and _recent_to_dest_nm.iloc[-1] > _recent_to_dest_nm.iloc[0] + 40.0:
        _gps_block_reason = f"moving_away({_recent_to_dest_nm.iloc[-1]:.0f}nm_vs_{_recent_to_dest_nm.iloc[0]:.0f}nm_start)"
    elif len(_recent_to_dest_nm) >= 1 and _recent_to_dest_nm.iloc[-1] > _recent_to_dest_nm.min() + 25.0:
        _gps_block_reason = f"last_ping_outlier({_recent_to_dest_nm.iloc[-1]:.0f}nm_vs_min_{_recent_to_dest_nm.min():.0f}nm)"
    elif _over_delayed and pd.notna(_raw_to_start_nm) and _raw_to_dest_nm > _raw_to_start_nm + 75.0:
        _gps_block_reason = f"closer_to_start({_raw_to_dest_nm:.0f}nm_dest_vs_{_raw_to_start_nm:.0f}nm_start)"
    _gps_pos_valid = (_gps_block_reason == "")
    if not _gps_pos_valid and pd.notna(_raw_to_dest_nm):
        log.debug("GPS fix blocked [%s]: port_pair=%s raw_haversine=%.0fnm planned=%.0fnm",
                  _gps_block_reason, port_pair, _raw_to_dest_nm, planned_distance if pd.notna(planned_distance) else -1)

    if (_gps_pos_valid
            and pd.notna(dist_to_dest_nm)
            and pd.notna(planned_distance) and planned_distance > 0
            and (pd.isna(progress_pct) or _over_delayed)):
        progress_pct = max(0.0, min(0.999, 1.0 - dist_to_dest_nm / planned_distance))
        remaining_distance_nm = dist_to_dest_nm
        # Use voyage avg pace (cumulative_dist/elapsed) rather than sog8 for remaining_naive.
        # sog8 overestimates pace for stop-heavy tug/barge voyages, causing severe under-prediction.
        _voyage_avg_sog = (
            cumulative_dist_nm / elapsed_h
            if (pd.notna(cumulative_dist_nm) and pd.notna(elapsed_h) and elapsed_h > 0
                and cumulative_dist_nm > 0)
            else None
        )
        if _voyage_avg_sog and _voyage_avg_sog > 0.1:
            remaining_naive = dist_to_dest_nm / _voyage_avg_sog
        elif pd.notna(sog8):
            remaining_naive = dist_to_dest_nm / max(sog8, 0.5)
        else:
            remaining_naive = np.nan

    speed_adj_progress = (1.0 - remaining_naive / planned_h) if pd.notna(remaining_naive) and pd.notna(planned_h) and planned_h > 0 else progress_pct

    # Vessel history features — look up by primary MMSI from AIS data
    _mmsi = np.nan
    if len(ais) > 0 and "MMSI" in ais.columns:
        _mmsi_vals = pd.to_numeric(ais["MMSI"], errors="coerce").dropna()
        if len(_mmsi_vals):
            _mmsi = int(_mmsi_vals.mode().iloc[0])
    vessel_hist_pace_ratio = np.nan
    vessel_hist_delay_std  = np.nan
    vessel_hist_leg_count  = np.nan
    vessel_hist_avg_pace   = np.nan
    vessel_route_pace_ratio = np.nan
    vessel_route_count      = np.nan
    if pd.notna(_mmsi):
        _vh = _get_vessel_hist()
        if len(_vh) and _mmsi in _vh.index:
            _vrow = _vh.loc[_mmsi]
            vessel_hist_pace_ratio = float(_vrow.get("vessel_hist_pace_ratio", np.nan))
            vessel_hist_delay_std  = float(_vrow.get("vessel_hist_delay_std",  np.nan))
            vessel_hist_leg_count  = float(_vrow.get("vessel_hist_leg_count",  np.nan))
            vessel_hist_avg_pace   = float(_vrow.get("vessel_hist_avg_pace",   np.nan))
        _vr = _get_vessel_route()
        if len(_vr):
            _key = (_mmsi, port_pair)
            if _key in _vr.index:
                _vrrow = _vr.loc[_key]
                vessel_route_pace_ratio = float(_vrrow.get("vessel_route_pace_ratio", np.nan))
                vessel_route_count      = float(_vrrow.get("vessel_route_count",      np.nan))
    sog_vs_vessel_hist = (sog2 / vessel_hist_avg_pace) if pd.notna(sog2) and pd.notna(vessel_hist_avg_pace) and vessel_hist_avg_pace > 0 else np.nan

    # Port-pair + progress-bucket historical remaining hours
    pp_hist_remaining_median = np.nan
    pp_hist_remaining_mean   = np.nan
    pp_hist_remaining_count  = np.nan
    if pd.notna(progress_pct) and 0.0 <= progress_pct < 1.0:
        _bucket = round(math.floor(progress_pct * 10) / 10.0, 1)
        _pp = _get_pp_hist()
        if len(_pp):
            _pp_key = (port_pair, _bucket)
            if _pp_key in _pp.index:
                _pprow = _pp.loc[_pp_key]
                pp_hist_remaining_median = float(_pprow.get("pp_hist_remaining_median", np.nan))
                pp_hist_remaining_mean   = float(_pprow.get("pp_hist_remaining_mean",   np.nan))
                pp_hist_remaining_count  = float(_pprow.get("pp_hist_remaining_count",  np.nan))

    speed_adjusted_remaining_h = max(elapsed_h / progress_pct - elapsed_h, 0.0) if pd.notna(sog2) and sog2 > 0.5 and pd.notna(progress_pct) and 0 < progress_pct <= 1.0 else np.nan
    is_stopped = float(int(pd.notna(sog2) and sog2 < 0.5)) if pd.notna(sog2) else np.nan
    is_slow = float(int(pd.notna(sog2) and pd.notna(planned_speed) and planned_speed > 0 and sog2 < planned_speed * 0.7)) if pd.notna(sog2) else np.nan
    pct_moving_2h = float((speed2 > 0.5).mean()) if len(speed2) else 0.0
    pct_moving_ctx = float((speed > 0.5).mean()) if len(speed) else 0.0
    total_stopped_h = elapsed_h * (1.0 - pct_moving_ctx) if pd.notna(elapsed_h) else np.nan
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
        "dist_to_dest_nm": dist_to_dest_nm,
        "gps_valid": float(int(_gps_pos_valid)) if pd.notna(_raw_to_dest_nm) else np.nan,
        "gps_block_reason": _gps_block_reason if not _gps_pos_valid and pd.notna(_raw_to_dest_nm) else "",
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
        "vessel_hist_pace_ratio": vessel_hist_pace_ratio,
        "vessel_hist_delay_std": vessel_hist_delay_std,
        "vessel_hist_leg_count": vessel_hist_leg_count,
        "vessel_route_pace_ratio": vessel_route_pace_ratio,
        "vessel_route_count": vessel_route_count,
        "sog_vs_vessel_hist": sog_vs_vessel_hist,
        "pp_hist_remaining_median": pp_hist_remaining_median,
        "pp_hist_remaining_mean": pp_hist_remaining_mean,
        "pp_hist_remaining_count": pp_hist_remaining_count,
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

    # No AIS observations yet — dynamic model has no observational signal.
    # Model A is trained purely on planning data (distance, speed, route) and is
    # better suited for this regime than a dynamic model running on all-NaN features.
    if (float(row.get("ctx_ping_count", 0)) == 0
            and model_a_pred_h is not None and pd.notna(model_a_pred_h)
            and float(model_a_pred_h) > 0):
        elapsed = float(row.get("elapsed_h") or 0.0)
        _ma_rem = max(float(model_a_pred_h) - elapsed, 0.0)
        # Fix 1: only override with model_a when it gives a positive remaining.
        # If elapsed >= model_a (vessel over-schedule with no pings), don't lock to 0 —
        # let the ML model run on whatever features it has.
        if _ma_rem > 0:
            remaining_h = _ma_rem
            log.info(
                "0 AIS pings — using Model A remaining: %.2fh - %.2fh elapsed = %.2fh.",
                float(model_a_pred_h), elapsed, remaining_h,
            )
        else:
            log.info(
                "0 AIS pings, elapsed (%.1fh) >= model_a (%.1fh) — falling through to ML model.",
                elapsed, float(model_a_pred_h),
            )

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
                "Using naive_remaining_h as fallback.",
                int(row.get("ctx_ping_count", 0)), elapsed, float(model_a_pred_h),
            )
            naive = row.get("naive_remaining_h", np.nan)
            remaining_h = float(naive) if pd.notna(naive) and float(naive) > 0 else 0.0

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
                        "All fallbacks give 0. Voyage appears complete — returning 0.",
                        progress_pct, row.get("elapsed_h", float("nan")),
                    )
                    remaining_h = 0.0
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

    # Tighter cap for 0-ping vessels: model has no observational signal, cap at model_a × 1.5.
    if float(row.get("ctx_ping_count", 0)) == 0 and model_a_pred_h is not None and pd.notna(model_a_pred_h) and float(model_a_pred_h) > 0:
        _no_ais_cap = float(model_a_pred_h) * 1.5
        if remaining_h > _no_ais_cap:
            log.warning("0-ping vessel: clamping remaining_h %.1fh → %.1fh (model_a × 1.5)", remaining_h, _no_ais_cap)
            remaining_h = _no_ais_cap

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
