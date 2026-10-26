"""Batch payload audit and endpoint comparison for 43 flagged voyage IDs.

Steps:
  1. Fetch payload from Snowflake for each voyage
  2. Audit payload quality (nulls, bad dtypes, missing features)
  3. Call SageMaker endpoint and compare input features vs returned features

Usage:
    python sagemaker/batch_audit.py [--endpoint opentug-live-predict-eta-py312] [--dry-run]
    python sagemaker/batch_audit.py --only-audit      # skip endpoint call
    python sagemaker/batch_audit.py --voyage-ids 125574 123199   # subset
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import boto3
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from snowflake_client import SnowflakeClient  # noqa: E402
from src.live_predict import (  # noqa: E402
    classify_leg_status,
    fetch_live_leg_ais,
    fetch_live_leg_statuslog,
    fetch_live_voyage_spine,
    fetch_vessel_features,
    select_current_leg,
)
from src.utils import uppercase_columns  # noqa: E402

# ── The 43 flagged voyage IDs (from prediction_discrepancy_findings.pdf) ──────
FLAGGED_IDS = [
    125362, 125295, 124312, 124258, 124256, 124204,
    123818, 123813, 123812, 123809, 123802, 123798, 123794, 123792, 123790,
    125574, 123199,
    122534, 122528, 122527, 122520, 122511, 122510, 122507, 122501, 122495,
    122486, 122467,
    120736, 118171, 111720, 111652, 110518, 110076, 101262, 100045,
    96543, 95620, 51403, 23149, 22939, 22933, 22932,
]

# Key columns expected in voyage legs
LEG_COLS = [
    "ID", "VOYAGE_ID", "SEQUENCE_NUM", "LEG_STATUS",
    "TIME_STARTED", "TIME_COMPLETED", "ETA_UPDATED",
    "START_PORT", "END_PORT", "PLANNED_DURATION_H", "PLANNED_DIST_NM",
    "VESSEL_ID", "SELECTED_ROUTE",
]
AIS_COLS = ["LAT", "LON", "SPEED", "TIMESTAMP"]
VESSEL_COLS = ["ID", "VESSEL_ID", "LOA", "BEAM", "DRAFT"]


def audit_dataframe(df: pd.DataFrame, expected_cols: list[str], label: str) -> dict:
    issues = []
    present = set(df.columns)

    missing = [c for c in expected_cols if c not in present]
    if missing:
        issues.append(f"missing columns: {missing}")

    null_summary = {}
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        if null_count > 0:
            null_summary[col] = f"{null_count}/{len(df)}"

    all_null = [c for c, v in null_summary.items() if v.split("/")[0] == v.split("/")[1]]

    return {
        "label": label,
        "rows": len(df),
        "issues": issues,
        "null_summary": null_summary,
        "all_null_cols": all_null,
    }


def audit_payload(voyage_id: int, payload: dict) -> dict:
    legs_df = uppercase_columns(pd.DataFrame(payload["voyage_legs"]))
    ais_df = uppercase_columns(pd.DataFrame(payload.get("current_leg_ais", [])))
    statuslog_df = uppercase_columns(pd.DataFrame(payload.get("current_leg_statuslog", [])))
    vessel_df = uppercase_columns(pd.DataFrame(payload.get("vessel_features", [])))

    audit = {
        "voyage_id": voyage_id,
        "legs": audit_dataframe(legs_df, LEG_COLS, "voyage_legs"),
        "ais": audit_dataframe(ais_df, AIS_COLS, "current_leg_ais"),
        "statuslog": {"rows": len(statuslog_df)},
        "vessel_features": audit_dataframe(vessel_df, VESSEL_COLS, "vessel_features"),
        "flags": [],
    }

    # Flag specific payload-level issues
    if "LEG_STATUS" in legs_df.columns and legs_df["LEG_STATUS"].isna().all():
        audit["flags"].append("LEG_STATUS column present but all-null")

    in_progress = legs_df[legs_df.get("LEG_STATUS", pd.Series(dtype=str)) == "in-progress"] if "LEG_STATUS" in legs_df.columns else pd.DataFrame()
    if len(in_progress) == 0:
        audit["flags"].append("no in-progress leg found in payload")
    elif len(in_progress) > 1:
        audit["flags"].append(f"multiple in-progress legs: {len(in_progress)}")

    if "PLANNED_DIST_NM" in legs_df.columns:
        zero_dist = int((legs_df["PLANNED_DIST_NM"].fillna(0) == 0).sum())
        if zero_dist > 0:
            audit["flags"].append(f"zero planned_dist_nm on {zero_dist}/{len(legs_df)} legs")

    if "SELECTED_ROUTE" in legs_df.columns:
        non_null_routes = int(legs_df["SELECTED_ROUTE"].notna().sum())
        audit["flags"].append(f"SELECTED_ROUTE non-null: {non_null_routes}/{len(legs_df)}")

    if len(ais_df) == 0:
        audit["flags"].append("WARN: no AIS rows — dynamic model will use fallback")
    if len(vessel_df) == 0:
        audit["flags"].append("WARN: no vessel feature rows")

    return audit


def call_endpoint(payload: dict, endpoint_name: str, profile: str) -> dict | None:
    try:
        rt = boto3.Session(profile_name=profile, region_name="us-east-1").client("sagemaker-runtime")
        body = json.dumps(payload, default=str)
        resp = rt.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Accept="application/json",
            Body=body,
        )
        return json.loads(resp["Body"].read())
    except Exception as e:
        return {"error": str(e)}


def compare_features(payload: dict, response: dict) -> dict:
    """Compare key features from input payload vs what endpoint returned."""
    legs_df = uppercase_columns(pd.DataFrame(payload["voyage_legs"]))

    in_progress = legs_df[legs_df["LEG_STATUS"] == "in-progress"] if "LEG_STATUS" in legs_df.columns else pd.DataFrame()
    current_leg_in = in_progress.iloc[0].to_dict() if len(in_progress) > 0 else {}

    legs_out = response.get("legs", [])
    current_leg_out = next((l for l in legs_out if l.get("role") == "current"), {})

    voyage = response.get("voyage", {})

    comparison = {
        "endpoint_error": response.get("error"),
        "total_remaining_h": voyage.get("total_remaining_h"),
        "consolidated_eta": voyage.get("consolidated_eta"),
        "bias_adj_h": voyage.get("bias_adj_h"),
        "dynamic_pred_h": current_leg_out.get("dynamic_pred_h"),
        "model_a_pred_h": current_leg_out.get("model_a_pred_h"),
        "raw_remaining_h": current_leg_out.get("raw_remaining_h"),
        "input_planned_dist_nm": current_leg_in.get("PLANNED_DIST_NM"),
        "input_planned_dur_h": current_leg_in.get("PLANNED_DURATION_H"),
        "input_time_started": str(current_leg_in.get("TIME_STARTED", "")),
        "input_time_completed": str(current_leg_in.get("TIME_COMPLETED", "")),
        "route_features_in_response": bool(current_leg_out.get("route_dist_nm") is not None),
    }
    return comparison


def build_payload(voyage_id: int, client: SnowflakeClient) -> dict:
    voyage_df = fetch_live_voyage_spine(client, voyage_id)
    current_leg = select_current_leg(voyage_df)
    ais_df = fetch_live_leg_ais(client, current_leg)
    statuslog_df = fetch_live_leg_statuslog(client, current_leg)
    leg_ids = voyage_df["ID"].tolist()
    vessel_features_df = fetch_vessel_features(client, leg_ids)
    snapshot_ts = pd.Timestamp.now("UTC")
    return {
        "voyage_id": int(voyage_id),
        "snapshot_ts": snapshot_ts.isoformat(),
        "voyage_legs": voyage_df.to_dict(orient="records"),
        "current_leg_ais": ais_df.to_dict(orient="records"),
        "current_leg_statuslog": statuslog_df.to_dict(orient="records"),
        "vessel_features": vessel_features_df.to_dict(orient="records"),
    }


def print_audit(a: dict):
    vid = a["voyage_id"]
    leg_rows = a["legs"]["rows"]
    ais_rows = a["ais"]["rows"]
    flags = a["flags"]
    leg_nulls = a["legs"]["null_summary"]
    all_null = a["legs"]["all_null_cols"]

    print(f"\n{'='*60}")
    print(f"Voyage {vid:>7} | legs={leg_rows} ais={ais_rows}")
    if flags:
        for f in flags:
            print(f"  FLAG: {f}")
    if leg_nulls:
        print(f"  Leg nulls  : {leg_nulls}")
    if all_null:
        print(f"  All-null   : {all_null}")
    if a["legs"]["issues"]:
        print(f"  Issues     : {a['legs']['issues']}")


def print_comparison(vid: int, comp: dict):
    if comp.get("endpoint_error"):
        print(f"  ENDPOINT ERROR: {comp['endpoint_error']}")
        return
    print(f"  remaining_h={comp['total_remaining_h']}  bias_adj={comp['bias_adj_h']}  "
          f"dynamic={comp['dynamic_pred_h']}  model_a={comp['model_a_pred_h']}")
    print(f"  consolidated_eta={comp['consolidated_eta']}")
    print(f"  input: dist={comp['input_planned_dist_nm']} dur={comp['input_planned_dur_h']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="opentug-live-predict-eta-py312")
    parser.add_argument("--profile", default="opentug")
    parser.add_argument("--only-audit", action="store_true")
    parser.add_argument("--voyage-ids", type=int, nargs="+", default=None)
    parser.add_argument("--output", default="sagemaker/batch_audit_results.json")
    args = parser.parse_args()

    voyage_ids = args.voyage_ids or FLAGGED_IDS
    print(f"Running audit for {len(voyage_ids)} voyage IDs")
    print(f"Endpoint: {args.endpoint}")

    client = SnowflakeClient()
    results = []

    try:
        for i, vid in enumerate(voyage_ids, 1):
            print(f"\n[{i}/{len(voyage_ids)}] Voyage {vid}", end="", flush=True)
            row = {"voyage_id": vid, "payload_ok": False, "audit": None, "comparison": None, "error": None}

            try:
                payload = build_payload(vid, client)
                row["payload_ok"] = True
                print(f" — payload built ({len(payload['voyage_legs'])} legs, {len(payload['current_leg_ais'])} AIS)", end="", flush=True)

                audit = audit_payload(vid, payload)
                row["audit"] = audit
                print_audit(audit)

                if not args.only_audit:
                    resp = call_endpoint(payload, args.endpoint, args.profile)
                    comp = compare_features(payload, resp)
                    row["comparison"] = comp
                    print_comparison(vid, comp)

            except Exception as e:
                row["error"] = traceback.format_exc()
                print(f"\n  ERROR: {e}")

            results.append(row)

    finally:
        client.close()

    # Save full results
    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n\nResults saved to {out_path}")

    # Summary
    ok = sum(1 for r in results if r["payload_ok"])
    errors = sum(1 for r in results if r["error"])
    zero_remaining = sum(
        1 for r in results
        if r["comparison"] and r["comparison"].get("total_remaining_h") == 0.0
    )
    print(f"\n{'='*60}")
    print(f"SUMMARY: {ok}/{len(results)} payloads built | {errors} errors | {zero_remaining} still zero remaining")


if __name__ == "__main__":
    main()
