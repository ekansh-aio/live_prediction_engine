from __future__ import annotations

import os

import pandas as pd

from . import bias, markerwait, stoppage


def assemble_voyage(voyage_legs: pd.DataFrame, snapshot_ts: pd.Timestamp, current_remaining_h: float,
                    current_live_row: dict, future_predictions: list[dict], stop_patterns=None, marker_map=None,
                    incumbent_eta=None) -> dict:
    voyage_legs = voyage_legs.sort_values("SEQUENCE_NUM").reset_index(drop=True)
    in_progress = voyage_legs[voyage_legs["LEG_STATUS"] == "in-progress"]
    if len(in_progress) == 0:
        raise ValueError(f"assemble_voyage: no in-progress leg found among {len(voyage_legs)} legs")
    current_leg = in_progress.iloc[-1]
    future_legs = voyage_legs[voyage_legs["LEG_STATUS"] == "future"]
    future_duration_h = sum(max(float(r["corrected_duration_h"]), 0.0) for r in future_predictions)
    future_stop_h = 0.0
    future_marker_h = 0.0
    current_marker_h = 0.0
    marker_mode = os.environ.get("MARKER_MODE", "current")

    if marker_map is not None:
        if marker_mode == "future" and len(future_legs):
            future_marker_h = float(markerwait.voyage_wait(future_legs, marker_map))
        elif marker_mode == "current":
            current_marker_h = float(markerwait.leg_remaining_wait(
                current_leg.get("SERVICE_TYPE"), current_leg.get("START_PORT_ID"), current_leg.get("END_PORT_ID"),
                current_remaining_h, float(current_live_row.get("elapsed_h") or 0.0), marker_map,
            ))

    if len(future_legs):
        future_stop_h += float(stoppage.junction_estimate(
            stop_patterns, current_leg.get("SERVICE_TYPE"), current_leg.get("START_PORT_ID"), current_leg.get("END_PORT_ID"), 0.0
        ))
    last_idx = voyage_legs.index[-1]
    for idx, fl in future_legs.iterrows():
        if idx == last_idx:
            continue
        future_stop_h += float(stoppage.junction_estimate(
            stop_patterns, fl.get("SERVICE_TYPE"), fl.get("START_PORT_ID"), fl.get("END_PORT_ID"), 0.0
        ))

    raw_total_h = current_remaining_h + current_marker_h + future_duration_h + future_stop_h + future_marker_h
    total_remaining_h, bias_adj_h, p10_h, p90_h = bias.apply(raw_total_h, len(future_legs))
    consolidated_eta = snapshot_ts + pd.Timedelta(hours=total_remaining_h)
    if incumbent_eta is None:
        last_leg = voyage_legs.iloc[-1]
        incumbent_eta = last_leg["INITIAL_ETA"] if pd.notna(last_leg.get("INITIAL_ETA")) else None
    eta_diff = None
    if incumbent_eta is not None:
        inc_ts = incumbent_eta.tz_localize("UTC") if getattr(incumbent_eta, "tzinfo", None) is None else incumbent_eta
        eta_diff = round((consolidated_eta - inc_ts).total_seconds() / 3600, 2)

    return {
        "current_subservice_id": int(current_leg["ID"]),
        "current_leg_seq": int(current_leg["SEQUENCE_NUM"]),
        "total_legs": int(len(voyage_legs)),
        "future_legs": int(len(future_legs)),
        "current_remaining_h": round(float(current_remaining_h), 2),
        "future_legs_duration_h": round(float(future_duration_h), 2),
        "future_dwell_h": round(float(future_stop_h), 2),
        "future_marker_wait_h": round(float(future_marker_h), 2),
        "current_marker_wait_h": round(float(current_marker_h), 2),
        "raw_total_remaining_h": round(float(raw_total_h), 2),
        "bias_adj_h": round(float(bias_adj_h), 2),
        "total_remaining_h": round(float(total_remaining_h), 2),
        "consolidated_eta": consolidated_eta.isoformat(),
        "eta_p10": (snapshot_ts + pd.Timedelta(hours=p10_h)).isoformat() if p10_h is not None else None,
        "eta_p90": (snapshot_ts + pd.Timedelta(hours=p90_h)).isoformat() if p90_h is not None else None,
        "incumbent_eta": incumbent_eta.isoformat() if incumbent_eta is not None else None,
        "eta_diff_vs_incumbent_h": eta_diff,
    }
