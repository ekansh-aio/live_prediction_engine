"""Populate assets/lookups/ from Snowflake before packaging for SageMaker.

Usage (requires sf_daemon.py to be running, or SF_MODE=direct):
    python sagemaker/refresh_lookups.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import os
# Extend daemon socket timeout to 5 min — bulk vessel_features query can be slow.
os.environ["SF_DAEMON_TIMEOUT"] = "300"

from snowflake_client import SnowflakeClient  # noqa: E402  (imports src.config which loads .env)

LOOKUPS_DIR = ROOT / "assets" / "lookups"


def main():
    LOOKUPS_DIR.mkdir(parents=True, exist_ok=True)
    client = SnowflakeClient()
    try:
        print("Fetching port_lookup...")
        port_df = client.query_file(ROOT / "sql" / "port_lookup.sql")
        out = LOOKUPS_DIR / "port_lookup.parquet"
        port_df.to_parquet(out, index=False)
        print(f"  Saved {len(port_df):,} rows → {out.relative_to(ROOT)}")

        print("Fetching vessel_features (3-year window)...")
        vessel_df = client.query_file(ROOT / "sql" / "vessel_features_all.sql")
        out = LOOKUPS_DIR / "vessel_features.parquet"
        vessel_df.to_parquet(out, index=False)
        print(f"  Saved {len(vessel_df):,} rows → {out.relative_to(ROOT)}")

        print("\nLookups refreshed. Ready to run: python sagemaker/package_model.py")
    finally:
        client.close()


if __name__ == "__main__":
    main()
