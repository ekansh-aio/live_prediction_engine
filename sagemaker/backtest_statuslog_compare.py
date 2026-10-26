"""Compare model ETA vs statuslog (dispatcher-entered) ETA for backtest snapshots.

For each snapshot in backtest.db, finds the most recent statuslog ETA that was
visible at that time T (TIME_CREATED <= snapshot_ts), then computes error vs
actual completion and produces a side-by-side comparison by time-remaining bucket.

Usage:
    python sagemaker/backtest_statuslog_compare.py --db backtest.db
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from snowflake_client import SnowflakeClient
from src.live_predict import fetch_live_voyage_spine
from src.utils import uppercase_columns, parse_dt


def fetch_full_statuslog(client: SnowflakeClient, leg_ids: list[int]) -> pd.DataFrame:
    """Fetch complete statuslog history for a list of leg IDs (no time cap)."""
    if not leg_ids:
        return pd.DataFrame()
    ids_csv = ",".join(str(i) for i in leg_ids)
    sql = f"""
        SELECT sl.SAILING_ID, sl.TIME_CREATED, sl.ETA
        FROM EDPPROD.EDP_SOURCE.WEBATDS_STATUSLOG sl
        WHERE sl.SAILING_ID IN ({ids_csv})
          AND sl.ETA IS NOT NULL
        ORDER BY sl.TIME_CREATED
    """
    df = uppercase_columns(client.query_text(sql))
    if df.empty:
        return df
    df["TIME_CREATED"] = parse_dt(df["TIME_CREATED"])
    df["ETA"] = parse_dt(df["ETA"])
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="backtest.db")
    args = parser.parse_args()

    import sqlite3
    conn = sqlite3.connect(args.db)
    df = pd.read_sql("""
        SELECT voyage_id, snapshot_ts, actual_completion_ts,
               error_h, total_remaining_h, current_leg_seq
        FROM backtest
        WHERE endpoint_error IS NULL AND error_h IS NOT NULL
    """, conn)
    conn.close()

    if df.empty:
        print("No completed rows in backtest.db yet.")
        return

    df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"], utc=True)
    df["actual_completion_ts"] = pd.to_datetime(df["actual_completion_ts"], utc=True)
    df["true_remaining_h"] = (df["actual_completion_ts"] - df["snapshot_ts"]).dt.total_seconds() / 3600

    voyage_ids = df["voyage_id"].unique().tolist()
    print(f"Fetching statuslog history for {len(voyage_ids)} voyages...")

    client = SnowflakeClient()
    sl_rows = []
    spine_cache: dict[int, pd.DataFrame] = {}

    try:
        for vid in voyage_ids:
            try:
                spine = fetch_live_voyage_spine(client, vid)
                spine_cache[vid] = spine
                seq_to_legid = spine.set_index("SEQUENCE_NUM")["ID"].to_dict()

                # Collect all leg IDs for this voyage
                leg_ids = [int(v) for v in spine["ID"].tolist()]
                sl = fetch_full_statuslog(client, leg_ids)
                if sl.empty:
                    continue

                # Map leg_id back to seq_num for easier lookup
                legid_to_seq = {int(v): k for k, v in seq_to_legid.items()}
                sl["SEQUENCE_NUM"] = sl["SAILING_ID"].map(legid_to_seq)
                sl["voyage_id"] = vid
                sl_rows.append(sl)
            except Exception as e:
                print(f"  Voyage {vid}: {e}")
    finally:
        client.close()

    if not sl_rows:
        print("No statuslog ETA data found for any voyage.")
        return

    sl_all = pd.concat(sl_rows, ignore_index=True)
    print(f"Fetched {len(sl_all)} statuslog ETA entries across {len(voyage_ids)} voyages.")

    # For each backtest row, find the latest statuslog ETA visible at snapshot_ts
    # on the current leg (matching current_leg_seq)
    results = []
    for _, row in df.iterrows():
        vid = row["voyage_id"]
        T = row["snapshot_ts"]
        seq = row["current_leg_seq"]

        # Statuslog entries for this leg, visible at T
        mask = (
            (sl_all["voyage_id"] == vid) &
            (sl_all["SEQUENCE_NUM"] == seq) &
            (sl_all["TIME_CREATED"] <= T)
        )
        leg_sl = sl_all[mask]

        if leg_sl.empty:
            sl_eta = None
            sl_error_h = None
        else:
            # Most recent statuslog ETA before T
            latest = leg_sl.sort_values("TIME_CREATED").iloc[-1]
            sl_eta = latest["ETA"]
            if pd.notna(sl_eta):
                actual = row["actual_completion_ts"]
                if sl_eta.tzinfo is None:
                    sl_eta = sl_eta.tz_localize("UTC")
                if actual.tzinfo is None:
                    actual = actual.tz_localize("UTC")
                sl_error_h = (sl_eta - actual).total_seconds() / 3600
            else:
                sl_error_h = None

        results.append({
            "voyage_id":        vid,
            "snapshot_ts":      T,
            "true_remaining_h": row["true_remaining_h"],
            "model_error_h":    row["error_h"],
            "sl_error_h":       sl_error_h,
            "has_sl_eta":       sl_error_h is not None,
        })

    res = pd.DataFrame(results)
    res_with_sl = res[res["has_sl_eta"]].copy()
    coverage = len(res_with_sl) / len(res) * 100

    print(f"\nStatuslog ETA coverage: {len(res_with_sl)}/{len(res)} snapshots ({coverage:.1f}%)")

    # Bucket by true remaining time
    bins   = [0, 24, 72, 168, 336, 9999]
    labels = ["<1d left", "1-3d left", "3-7d left", "7-14d left", ">14d left"]
    res["bucket"] = pd.cut(res["true_remaining_h"], bins=bins, labels=labels)
    res_with_sl["bucket"] = pd.cut(res_with_sl["true_remaining_h"], bins=bins, labels=labels)

    print("\n-- Error comparison by time-remaining bucket --")
    print(f"{'Bucket':<12} {'N':>5}  {'Model MAE':>10} {'SL MAE':>10}  {'Model mean':>11} {'SL mean':>10}  {'SL coverage':>12}")
    print("-" * 80)

    for label in labels:
        grp_all = res[res["bucket"] == label]
        grp_sl  = res_with_sl[res_with_sl["bucket"] == label]
        if grp_all.empty:
            continue
        n = len(grp_all)
        model_mae  = grp_all["model_error_h"].abs().mean()
        model_mean = grp_all["model_error_h"].mean()
        sl_mae     = grp_sl["sl_error_h"].abs().mean() if not grp_sl.empty else float("nan")
        sl_mean    = grp_sl["sl_error_h"].mean()       if not grp_sl.empty else float("nan")
        sl_cov     = len(grp_sl) / n * 100             if n else 0
        print(f"{label:<12} {n:>5}  {model_mae:>10.1f} {sl_mae:>10.1f}  {model_mean:>+11.1f} {sl_mean:>+10.1f}  {sl_cov:>11.1f}%")

    print()
    print("Positive error = predicted too late  |  Negative = predicted too early")


if __name__ == "__main__":
    main()
