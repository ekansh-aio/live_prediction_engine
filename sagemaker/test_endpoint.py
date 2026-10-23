"""Test the deployed SageMaker inference endpoint end-to-end.

Fetches live voyage data from Snowflake (same as predict.py does),
builds the payload, calls the endpoint, and prints the result.

Usage:
    python sagemaker/test_endpoint.py --voyage-id 125584
    python sagemaker/test_endpoint.py --voyage-id 125584 --endpoint opentug-live-predict-eta
    python sagemaker/test_endpoint.py --voyage-id 125584 --dry-run   # build payload only, skip endpoint call
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import boto3
import pandas as pd

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


def build_payload(voyage_id: int, client: SnowflakeClient) -> dict:
    print(f"Fetching voyage spine for voyage_id={voyage_id}...")
    voyage_df = fetch_live_voyage_spine(client, voyage_id)
    current_leg = select_current_leg(voyage_df)
    print(f"  {len(voyage_df)} legs total | current leg_id={int(current_leg['ID'])} seq={int(current_leg['SEQUENCE_NUM'])}")

    print("Fetching AIS pings...")
    ais_df = fetch_live_leg_ais(client, current_leg)
    print(f"  {len(ais_df)} AIS rows")

    print("Fetching statuslog...")
    statuslog_df = fetch_live_leg_statuslog(client, current_leg)
    print(f"  {len(statuslog_df)} statuslog rows")

    print("Fetching vessel features (fresh)...")
    leg_ids = voyage_df["ID"].tolist()
    vessel_features_df = fetch_vessel_features(client, leg_ids)
    print(f"  {len(vessel_features_df)} vessel feature rows")

    snapshot_ts = pd.Timestamp.now("UTC")

    return {
        "voyage_id": int(voyage_id),
        "snapshot_ts": snapshot_ts.isoformat(),
        "voyage_legs": voyage_df.to_dict(orient="records"),
        "current_leg_ais": ais_df.to_dict(orient="records"),
        "current_leg_statuslog": statuslog_df.to_dict(orient="records"),
        "vessel_features": vessel_features_df.to_dict(orient="records"),
    }


def call_endpoint(payload: dict, endpoint_name: str, profile: str) -> dict:
    rt = boto3.Session(profile_name=profile, region_name="us-east-1").client("sagemaker-runtime")
    body = json.dumps(payload, default=str)
    print(f"\nCalling endpoint '{endpoint_name}' (payload {len(body)/1024:.1f} KB)...")
    resp = rt.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Accept="application/json",
        Body=body,
    )
    return json.loads(resp["Body"].read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--voyage-id", type=int, required=True)
    parser.add_argument("--endpoint", default="opentug-live-predict-eta")
    parser.add_argument("--profile", default="opentug")
    parser.add_argument("--dry-run", action="store_true", help="Build payload but skip endpoint call")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print full JSON result")
    args = parser.parse_args()

    client = SnowflakeClient()
    try:
        payload = build_payload(args.voyage_id, client)
    finally:
        client.close()

    if args.dry_run:
        print("\n--- Payload preview (voyage_legs only) ---")
        print(json.dumps(payload["voyage_legs"][:2], default=str, indent=2))
        print(f"\nTotal legs: {len(payload['voyage_legs'])}, AIS rows: {len(payload['current_leg_ais'])}, statuslog rows: {len(payload['current_leg_statuslog'])}")
        print("\nDry run complete. Skipping endpoint call.")
        return

    result = call_endpoint(payload, args.endpoint, args.profile)

    voyage = result.get("voyage", {})
    print("\n--- Result ---")
    print(f"  voyage_id         : {result.get('voyage_id')}")
    print(f"  as_of_utc         : {result.get('as_of_utc')}")
    print(f"  consolidated_eta  : {voyage.get('consolidated_eta')}")
    print(f"  total_remaining_h : {voyage.get('total_remaining_h')}")
    print(f"  eta_p10           : {voyage.get('eta_p10')}")
    print(f"  eta_p90           : {voyage.get('eta_p90')}")
    print(f"  bias_adj_h        : {voyage.get('bias_adj_h')}")
    print(f"  future_legs       : {voyage.get('future_legs')}")
    print(f"  markerwait_loaded : {result.get('asset_status', {}).get('markerwait_loaded')}")
    print(f"  stoppage_loaded   : {result.get('asset_status', {}).get('stoppage_loaded')}")

    if args.pretty:
        print("\n--- Full JSON ---")
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
