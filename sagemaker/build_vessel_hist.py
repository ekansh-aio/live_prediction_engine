"""Build vessel/route/port-pair history lookup parquets for dynamic model features,
and Model A encoding priors.

Output:
  assets/lookups/vessel_hist_lookup.parquet   — per-MMSI aggregates
  assets/lookups/vessel_route_lookup.parquet  — per-MMSI + port_pair aggregates
  assets/lookups/pp_hist_lookup.parquet       — per port_pair + progress_bucket
  assets/model_a/encoding_priors.parquet      — per port_pair log-ratio residual stats

Usage:
    python sagemaker/build_vessel_hist.py
"""
from __future__ import annotations
import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from snowflake_client import SnowflakeClient

VESSEL_SQL          = (ROOT / "sql" / "vessel_hist_lookup.sql").read_text(encoding="utf-8")
PP_SQL              = (ROOT / "sql" / "pp_hist_lookup.sql").read_text(encoding="utf-8")
TORTUOSITY_SQL      = (ROOT / "sql" / "route_tortuosity_lookup.sql").read_text(encoding="utf-8")
ENCODING_PRIORS_SQL = (ROOT / "sql" / "model_a_encoding_priors.sql").read_text(encoding="utf-8")
LOOKUPS             = ROOT / "assets" / "lookups"


def build_vessel_route(client: SnowflakeClient) -> None:
    print("Fetching vessel + route history...")
    df = client.query_text(VESSEL_SQL, {})
    print(f"  {len(df)} rows, {df['MMSI'].nunique()} vessels")

    df.columns = [c.lower() for c in df.columns]

    # vessel-level lookup (one row per MMSI, int index)
    vessel_df = (
        df[["mmsi", "vessel_hist_leg_count", "vessel_hist_pace_ratio",
            "vessel_hist_delay_std", "vessel_hist_avg_pace"]]
        .drop_duplicates(subset=["mmsi"])
        .copy()
    )
    vessel_df["mmsi"] = vessel_df["mmsi"].astype("Int64")
    vessel_df = vessel_df.set_index("mmsi")
    out_v = LOOKUPS / "vessel_hist_lookup.parquet"
    vessel_df.to_parquet(out_v)
    print(f"  vessel_hist_lookup: {len(vessel_df)} vessels -> {out_v}")

    # route-level lookup (one row per MMSI + port_pair)
    route_df = (
        df[["mmsi", "port_pair", "vessel_route_count", "vessel_route_pace_ratio"]]
        .dropna(subset=["port_pair"])
        .drop_duplicates(subset=["mmsi", "port_pair"])
        .copy()
    )
    route_df["mmsi"] = route_df["mmsi"].astype("Int64")
    route_df = route_df.set_index(["mmsi", "port_pair"])
    out_r = LOOKUPS / "vessel_route_lookup.parquet"
    route_df.to_parquet(out_r)
    print(f"  vessel_route_lookup: {len(route_df)} vessel-route pairs -> {out_r}")


def build_route_tortuosity(client: SnowflakeClient) -> None:
    print("Fetching route planned distances for tortuosity...")
    df = client.query_text(TORTUOSITY_SQL, {})
    print(f"  {len(df)} routes")

    df.columns = [c.lower() for c in df.columns]
    df["start_port_id"] = pd.to_numeric(df["start_port_id"], errors="coerce")
    df["end_port_id"]   = pd.to_numeric(df["end_port_id"],   errors="coerce")
    df = df.dropna(subset=["start_port_id", "end_port_id"])
    df["start_port_id"] = df["start_port_id"].astype(int)
    df["end_port_id"]   = df["end_port_id"].astype(int)

    # Load port coordinates
    ports = pd.read_parquet(LOOKUPS / "port_lookup.parquet").set_index("ID")[["LAT", "LON"]]

    import math
    def haversine_nm(lat1, lon1, lat2, lon2):
        R = 3440.065
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
        return R * 2 * math.asin(math.sqrt(max(0.0, min(1.0, a))))

    def get_tortuosity(row):
        try:
            sp = ports.loc[row["start_port_id"]]
            ep = ports.loc[row["end_port_id"]]
            h = haversine_nm(sp["LAT"], sp["LON"], ep["LAT"], ep["LON"])
            if h < 1.0:  # same-port or trivially close
                return float("nan")
            t = row["route_planned_dist_nm"] / h
            return max(1.0, min(5.0, t))  # cap: can't be < straight line, or > 5× (data issue)
        except (KeyError, ZeroDivisionError, ValueError):
            return float("nan")

    df["haversine_nm"] = df.apply(lambda r: haversine_nm(
        ports.loc[r["start_port_id"], "LAT"] if r["start_port_id"] in ports.index else float("nan"),
        ports.loc[r["start_port_id"], "LON"] if r["start_port_id"] in ports.index else float("nan"),
        ports.loc[r["end_port_id"],   "LAT"] if r["end_port_id"]   in ports.index else float("nan"),
        ports.loc[r["end_port_id"],   "LON"] if r["end_port_id"]   in ports.index else float("nan"),
    ) if (r["start_port_id"] in ports.index and r["end_port_id"] in ports.index) else float("nan"), axis=1)

    df["tortuosity"] = df.apply(get_tortuosity, axis=1)
    df = df.dropna(subset=["tortuosity"])

    out_df = df[["port_pair", "route_planned_dist_nm", "haversine_nm", "tortuosity", "route_obs_count"]]
    out_df = out_df.set_index("port_pair")
    out = LOOKUPS / "route_tortuosity_lookup.parquet"
    out_df.to_parquet(out)
    print(f"  route_tortuosity_lookup: {len(out_df)} routes, tortuosity p50={out_df['tortuosity'].median():.2f}, max={out_df['tortuosity'].max():.2f} -> {out}")


def build_pp_hist(client: SnowflakeClient) -> None:
    print("Fetching port-pair progress-bucket history...")
    df = client.query_text(PP_SQL, {})
    print(f"  {len(df)} rows, {df['PORT_PAIR'].nunique()} port pairs")

    df.columns = [c.lower() for c in df.columns]
    df["progress_bucket"] = df["progress_bucket"].round(1)
    df = df.set_index(["port_pair", "progress_bucket"])
    out = LOOKUPS / "pp_hist_lookup.parquet"
    df.to_parquet(out)
    print(f"  pp_hist_lookup: {len(df)} rows -> {out}")


def build_encoding_priors(client: SnowflakeClient) -> None:
    print("Fetching Model A encoding priors (port-pair log-ratio stats)...")
    df = client.query_text(ENCODING_PRIORS_SQL, {})
    print(f"  {len(df)} port-pair rows")
    df.columns = [c.lower() for c in df.columns]
    df["pair_count"] = df["pair_count"].astype(float)
    out = ROOT / "assets" / "model_a" / "encoding_priors.parquet"
    df.to_parquet(out, index=False)
    gm = df["global_mean"].iloc[0] if len(df) else float("nan")
    gs = df["global_std"].iloc[0] if len(df) else float("nan")
    print(f"  encoding_priors: {len(df)} routes, global_mean={gm:.4f}, global_std={gs:.4f} -> {out}")


def main():
    print("Connecting to Snowflake...")
    client = SnowflakeClient()
    build_vessel_route(client)
    build_pp_hist(client)
    build_route_tortuosity(client)
    build_encoding_priors(client)
    print("\nDone.")


if __name__ == "__main__":
    main()
