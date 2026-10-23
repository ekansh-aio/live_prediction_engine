from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config
from .model_utils import align_to_booster


class FLCInfer:
    def __init__(self):
        self.model = lgb.Booster(model_file=str(config.MODEL_FILES["flc_model"]))
        cfg = json.loads(config.MODEL_FILES["flc_lists"].read_text(encoding="utf-8"))
        self.numeric = cfg.get("NUMERIC_USE", cfg["FEATURES"])
        self.cat = cfg.get("CATEGORICAL_USE", [])
        self.best = int(cfg.get("BEST_ITER", -1))

    def build_row(self, future_leg: pd.Series, current_leg: pd.Series, current_live_row: dict, model_a_pred_h: float) -> dict:
        f_dist = float(future_leg.get("ESTIMATED_DISTANCE") or np.nan)
        f_speed = float(future_leg.get("ESTIMATED_SPEED") or np.nan)
        f_planned = f_dist / f_speed if pd.notna(f_dist) and pd.notna(f_speed) and f_speed > 0 else np.nan
        cur_pp = f"{current_leg['START_PORT_ID']}_{current_leg['END_PORT_ID']}"
        fut_pp = f"{future_leg['START_PORT_ID']}_{future_leg['END_PORT_ID']}"
        s4 = current_live_row.get("sog_avg_4h", np.nan)
        s8 = current_live_row.get("sog_avg_8h", np.nan)
        return {
            "prior_leg_delay_h": current_live_row.get("cumulative_delay_h", np.nan),
            "prior_leg_pace_ratio": current_live_row.get("progress_rate_ratio", 1.0),
            "voyage_avg_pace_ratio": current_live_row.get("overall_pace_ratio", 1.0),
            "voyage_total_delay_h": current_live_row.get("cumulative_delay_h", np.nan),
            "legs_completed": 1.0,
            "future_planned_dist": f_dist,
            "future_planned_speed": f_speed,
            "future_planned_dur_h": f_planned,
            "future_num_locks": float(future_leg.get("NUM_LOCKS") or 0.0),
            "same_port_pair": int(cur_pp == fut_pp),
            "mid_sog_avg_4h": s4,
            "mid_sog_avg_16h": current_live_row.get("sog_avg_16h", np.nan),
            "mid_sog_delta_8h": current_live_row.get("sog_delta_8h", 0.0),
            "mid_cumulative_delay_h": current_live_row.get("cumulative_delay_h", np.nan),
            "mid_progress_rate_ratio": current_live_row.get("progress_rate_ratio", np.nan),
            "mid_pct_moving_4h": current_live_row.get("pct_moving_2h", np.nan),
            "mid_speed_ratio": (s8 / f_speed) if pd.notna(s8) and pd.notna(f_speed) and f_speed > 0 else np.nan,
            "future_service_type": str(future_leg.get("SERVICE_TYPE", "")),
            "future_port_pair": fut_pp,
            "prior_start_port": str(current_leg["START_PORT_ID"]),
            "prior_end_port": str(current_leg["END_PORT_ID"]),
            "future_start_port": str(future_leg["START_PORT_ID"]),
            "future_end_port": str(future_leg["END_PORT_ID"]),
            "future_ma_predicted_h": float(model_a_pred_h),
        }

    def predict_leg(self, future_leg: pd.Series, current_leg: pd.Series, current_live_row: dict, model_a_pred_h: float):
        row = self.build_row(future_leg, current_leg, current_live_row, model_a_pred_h)
        X = align_to_booster(pd.DataFrame([row]), self.model, feature_names=self.numeric + self.cat, cat_names=self.cat)
        resid = float(self.model.predict(X, num_iteration=self.best)[0])
        return resid, max(0.0, float(model_a_pred_h) + resid), row
