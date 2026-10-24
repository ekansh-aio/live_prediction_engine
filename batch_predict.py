from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

import boto3
import pandas as pd

VOYAGE_IDS = [
    125584, 125482, 125436, 125405, 125403, 125380, 125378, 125377,
    125328, 125323, 125275, 125209, 125172, 124116, 122984, 122570,
    121483, 121314, 120795, 120598, 120330, 119907, 117409, 117209,
    115400, 114466, 113654, 111196, 108918, 104809, 103008,
]

ENDPOINT_NAME = "opentug-live-predict-eta"
AWS_PROFILE = "opentug"
AWS_REGION = "us-east-1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("batch_predict.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("batch_predict")


def _build_payload(voyage_id: int, client) -> dict:
    from src.live_predict import (
        fetch_live_leg_ais,
        fetch_live_leg_statuslog,
        fetch_live_voyage_spine,
        fetch_vessel_features,
        select_current_leg,
    )
    voyage_df = fetch_live_voyage_spine(client, voyage_id)
    current_leg = select_current_leg(voyage_df)
    ais_df = fetch_live_leg_ais(client, current_leg)
    statuslog_df = fetch_live_leg_statuslog(client, current_leg)
    vessel_df = fetch_vessel_features(client, voyage_df["ID"].tolist())
    return {
        "voyage_id": int(voyage_id),
        "snapshot_ts": pd.Timestamp.now("UTC").isoformat(),
        "voyage_legs": voyage_df.to_dict(orient="records"),
        "current_leg_ais": ais_df.to_dict(orient="records"),
        "current_leg_statuslog": statuslog_df.to_dict(orient="records"),
        "vessel_features": vessel_df.to_dict(orient="records"),
    }


def _call_endpoint(payload: dict) -> dict:
    rt = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION).client("sagemaker-runtime")
    body = json.dumps(payload, default=str)
    resp = rt.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=body,
    )
    return json.loads(resp["Body"].read())


def run_batch() -> None:
    from snowflake_client import SnowflakeClient
    from src.sqlite_store import store_to_sqlite

    started = datetime.now(timezone.utc).isoformat()
    log.info("Batch started at %s, %d voyages, endpoint=%s", started, len(VOYAGE_IDS), ENDPOINT_NAME)
    ok, failed = 0, []

    client = SnowflakeClient()
    try:
        for vid in VOYAGE_IDS:
            try:
                payload = _build_payload(vid, client)
                result = _call_endpoint(payload)
                sqlite_run_id = store_to_sqlite(result)
                voyage = result.get("voyage", {})
                log.info(
                    "voyage %s OK  eta=%s  remaining_h=%s  sqlite_id=%s",
                    vid,
                    voyage.get("consolidated_eta"),
                    voyage.get("total_remaining_h"),
                    sqlite_run_id,
                )
                ok += 1
            except Exception as exc:
                log.warning("voyage %s FAILED: %s", vid, exc, exc_info=True)
                failed.append(vid)
    finally:
        client.close()

    log.info(
        "Batch done: %d OK, %d failed%s",
        ok, len(failed),
        f"  failed={failed}" if failed else "",
    )


if __name__ == "__main__":
    run_batch()
