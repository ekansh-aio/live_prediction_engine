"""Backtest the ETA model against historical voyages.

For every voyage that completed between --window-start and --window-end, simulate
the model at T_initial, T+1h, T+2h, ... until actual completion time.
At each time T we reconstruct what was knowable then:
  - Future legs (ETA_UPDATED > T or null): planned fields kept, timing nulled
  - In-progress leg (ETA_UPDATED <= T, TIME_COMPLETED > T): TIME_COMPLETED nulled
  - Completed legs (TIME_COMPLETED <= T): both timestamps kept

Results are stored in backtest.db (SQLite).

Usage:
    python sagemaker/backtest.py --window-start 2026-10-10 --window-end 2026-10-20
    python sagemaker/backtest.py --voyage-ids 122984 121107 --dry-run
    python sagemaker/backtest.py --voyage-ids 122984 121107 --workers 4
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from snowflake_client import SnowflakeClient  # noqa: E402
from src.live_predict import (  # noqa: E402
    classify_leg_status,
    fetch_live_voyage_spine,
    fetch_vessel_features,
    select_current_leg,
)
from src.utils import uppercase_columns, parse_dt  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(threadName)s] %(levelname)s %(message)s")
log = logging.getLogger("backtest")

SQL_DIR = ROOT / "sql"

# SQLite requires one connection per thread; use a thread-local connection.
_db_local = threading.local()
_db_path: Path | None = None
_db_lock = threading.Lock()


def _read_sql(name: str) -> str:
    return (SQL_DIR / name).read_text(encoding="utf-8")


# ── SQLite ────────────────────────────────────────────────────────────────────

def init_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS backtest (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            voyage_id            INTEGER,
            snapshot_ts          TEXT,
            actual_completion_ts TEXT,
            predicted_eta        TEXT,
            error_h              REAL,
            total_remaining_h    REAL,
            dynamic_pred_h       REAL,
            model_a_pred_h       REAL,
            bias_adj_h           REAL,
            current_leg_seq      INTEGER,
            n_legs_total         INTEGER,
            n_legs_completed     INTEGER,
            endpoint_error       TEXT,
            gps_valid            REAL,
            gps_block_reason     TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_voyage ON backtest(voyage_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_voyage_snapshot ON backtest(voyage_id, snapshot_ts)")
    conn.commit()
    return conn


def _get_thread_conn() -> sqlite3.Connection:
    """Return a per-thread SQLite connection, creating it if needed."""
    if not hasattr(_db_local, "conn") or _db_local.conn is None:
        assert _db_path is not None, "_db_path not initialised"
        _db_local.conn = sqlite3.connect(_db_path, check_same_thread=False)
    return _db_local.conn


def _as_utc_timestamp(value: str | pd.Timestamp | None) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts


def get_last_snapshot_ts(voyage_id: int) -> pd.Timestamp | None:
    conn = _get_thread_conn()
    row = conn.execute(
        "SELECT MAX(snapshot_ts) FROM backtest WHERE voyage_id = ?",
        (voyage_id,),
    ).fetchone()
    return _as_utc_timestamp(row[0] if row else None)


def resolve_sim_start(
    voyage_id: int,
    sim_start: pd.Timestamp,
    step: pd.Timedelta,
    resume: bool,
) -> tuple[pd.Timestamp, pd.Timestamp | None]:
    if not resume:
        return sim_start, None
    last_snapshot_ts = get_last_snapshot_ts(voyage_id)
    if last_snapshot_ts is None:
        return sim_start, None
    return max(sim_start, last_snapshot_ts + step), last_snapshot_ts


def store_result(
    voyage_id: int,
    snapshot_ts: pd.Timestamp,
    actual_completion_ts: pd.Timestamp,
    response: dict | None,
    current_leg_seq: int,
    n_legs_total: int,
    n_legs_completed: int,
    endpoint_error: str | None,
) -> None:
    voyage = (response or {}).get("voyage", {})
    predicted_eta_str = voyage.get("consolidated_eta")
    error_h = None
    if predicted_eta_str and not endpoint_error:
        try:
            eta_ts = pd.Timestamp(predicted_eta_str)
            if eta_ts.tzinfo is None:
                eta_ts = eta_ts.tz_localize("UTC")
            act_ts = actual_completion_ts
            if act_ts.tzinfo is None:
                act_ts = act_ts.tz_localize("UTC")
            error_h = (eta_ts - act_ts).total_seconds() / 3600.0
        except Exception:
            pass

    legs_info = (response or {}).get("legs", [])
    current_leg_data = next((l for l in legs_info if l.get("sequence_num") == current_leg_seq), {})
    dynamic_feats = current_leg_data.get("dynamic_features") or {}
    gps_valid = dynamic_feats.get("gps_valid")
    gps_block_reason = dynamic_feats.get("gps_block_reason") or ""

    conn = _get_thread_conn()
    with _db_lock:
        conn.execute(
            """INSERT INTO backtest
               (voyage_id, snapshot_ts, actual_completion_ts, predicted_eta, error_h,
                total_remaining_h, dynamic_pred_h, model_a_pred_h, bias_adj_h,
                current_leg_seq, n_legs_total, n_legs_completed, endpoint_error,
                gps_valid, gps_block_reason)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                voyage_id,
                snapshot_ts.isoformat(),
                actual_completion_ts.isoformat(),
                predicted_eta_str,
                error_h,
                voyage.get("total_remaining_h"),
                current_leg_data.get("dynamic_pred_h"),
                current_leg_data.get("model_a_pred_h"),
                voyage.get("bias_adj_h"),
                current_leg_seq,
                n_legs_total,
                n_legs_completed,
                endpoint_error,
                gps_valid,
                gps_block_reason,
            ),
        )
        conn.commit()


# ── Snowflake helpers ─────────────────────────────────────────────────────────

def fetch_completed_voyages(client: SnowflakeClient, window_start: str, window_end: str) -> pd.DataFrame:
    df = client.query_text(_read_sql("backtest_completed_voyages.sql"), {
        "window_start": window_start,
        "window_end": window_end,
    })
    df = uppercase_columns(df)
    for c in ["VOYAGE_STARTED_AT", "VOYAGE_COMPLETED_AT"]:
        if c in df.columns:
            df[c] = parse_dt(df[c])
    return df


def fetch_backtest_ais(client: SnowflakeClient, leg_id: int, departure_anchor_iso: str | None, as_of_ts: pd.Timestamp) -> pd.DataFrame:
    return uppercase_columns(client.query_text(_read_sql("backtest_leg_ais.sql"), {
        "leg_id": leg_id,
        "departure_anchor": departure_anchor_iso,
        "as_of_ts": as_of_ts.isoformat(),
    }))


def fetch_backtest_statuslog(client: SnowflakeClient, leg_id: int, departure_anchor_iso: str | None, as_of_ts: pd.Timestamp) -> pd.DataFrame:
    return uppercase_columns(client.query_text(_read_sql("backtest_leg_statuslog.sql"), {
        "leg_id": leg_id,
        "departure_anchor": departure_anchor_iso,
        "as_of_ts": as_of_ts.isoformat(),
    }))


# ── Temporal masking ──────────────────────────────────────────────────────────

def mask_spine_at_T(full_spine: pd.DataFrame, T: pd.Timestamp) -> pd.DataFrame:
    """Return a copy of the voyage spine as it would have appeared at time T."""
    df = full_spine.copy()
    for idx, row in df.iterrows():
        ts = row.get("ETA_UPDATED")
        tc = row.get("TIME_COMPLETED")
        ts_notna = pd.notna(ts)
        tc_notna = pd.notna(tc)

        if not ts_notna or (ts_notna and ts > T):
            df.at[idx, "ETA_UPDATED"]    = pd.NaT
            df.at[idx, "TIME_COMPLETED"] = pd.NaT
        elif ts_notna and ts <= T and tc_notna and tc > T:
            df.at[idx, "TIME_COMPLETED"] = pd.NaT
    return df


def _departure_anchor_iso(leg_row: pd.Series) -> str | None:
    for col in ("ETA_UPDATED",):
        val = leg_row.get(col)
        if pd.notna(val):
            return val.isoformat() if hasattr(val, "isoformat") else str(val)
    return None


# ── Endpoint call ─────────────────────────────────────────────────────────────

def call_endpoint(payload: dict, endpoint_name: str, runtime_client) -> dict:
    body = json.dumps(payload, default=str)
    resp = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Accept="application/json",
        Body=body,
    )
    return json.loads(resp["Body"].read())


# ── Snapshot worker ───────────────────────────────────────────────────────────

def _run_snapshot(
    voyage_id: int,
    T: pd.Timestamp,
    masked_df: pd.DataFrame,
    current_leg: pd.Series,
    vessel_features_df: pd.DataFrame,
    actual_completion_ts: pd.Timestamp,
    n_legs_total: int,
    n_legs_completed: int,
    client: SnowflakeClient,
    runtime_client,
    endpoint_name: str,
    dry_run: bool,
) -> None:
    """Execute one time-step snapshot: fetch data, call endpoint, store result."""
    current_leg_seq = int(current_leg["SEQUENCE_NUM"])
    departure_anchor = _departure_anchor_iso(current_leg)
    leg_id = int(current_leg["ID"])

    # Fetch AIS and statuslog in parallel using two threads
    _fetched: dict[str, pd.DataFrame] = {"ais": pd.DataFrame(), "statuslog": pd.DataFrame()}
    fetch_errors: list[str] = []

    def _fetch_ais() -> None:
        try:
            _fetched["ais"] = fetch_backtest_ais(client, leg_id, departure_anchor, T)
        except Exception as e:
            fetch_errors.append(f"ais: {e}")

    def _fetch_statuslog() -> None:
        try:
            _fetched["statuslog"] = fetch_backtest_statuslog(client, leg_id, departure_anchor, T)
        except Exception as e:
            fetch_errors.append(f"statuslog: {e}")

    t1 = threading.Thread(target=_fetch_ais, name=f"ais-{voyage_id}-{T.hour}h")
    t2 = threading.Thread(target=_fetch_statuslog, name=f"sl-{voyage_id}-{T.hour}h")
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    if fetch_errors:
        # Daemon doesn't support concurrent access — fall back to sequential
        if _fetched["ais"].empty:
            try:
                _fetched["ais"] = fetch_backtest_ais(client, leg_id, departure_anchor, T)
            except Exception:
                pass
        if _fetched["statuslog"].empty:
            try:
                _fetched["statuslog"] = fetch_backtest_statuslog(client, leg_id, departure_anchor, T)
            except Exception:
                pass

    ais_df = _fetched["ais"]
    statuslog_df = _fetched["statuslog"]

    payload = {
        "voyage_id":             int(voyage_id),
        "snapshot_ts":           T.isoformat(),
        "voyage_legs":           masked_df.to_dict(orient="records"),
        "current_leg_ais":       ais_df.to_dict(orient="records"),
        "current_leg_statuslog": statuslog_df.to_dict(orient="records"),
        "vessel_features":       vessel_features_df.to_dict(orient="records"),
        "disable_smoothing":     True,
    }

    if dry_run:
        log.info(
            "[dry-run] voyage=%d T=%s leg_seq=%d ais=%d sl=%d completed=%d/%d",
            voyage_id, T.isoformat(), current_leg_seq,
            len(ais_df), len(statuslog_df), n_legs_completed, n_legs_total,
        )
        store_result(voyage_id, T, actual_completion_ts, None,
                     current_leg_seq, n_legs_total, n_legs_completed, "dry-run")
        return

    response = None
    endpoint_error = None
    try:
        response = call_endpoint(payload, endpoint_name, runtime_client)
    except Exception as exc:
        endpoint_error = str(exc)
        log.warning("voyage=%d T=%s endpoint error: %s", voyage_id, T.isoformat(), exc)

    store_result(voyage_id, T, actual_completion_ts, response,
                 current_leg_seq, n_legs_total, n_legs_completed, endpoint_error)


# ── Per-voyage simulation ─────────────────────────────────────────────────────

def run_voyage(
    voyage_id: int,
    voyage_started_at: pd.Timestamp,
    voyage_completed_at: pd.Timestamp,
    endpoint_name: str,
    profile: str,
    step: pd.Timedelta,
    dry_run: bool,
    resume: bool,
    max_lookback: pd.Timedelta | None = None,
) -> int:
    """Run the full time-step simulation for one voyage. Opens its own Snowflake client."""
    sim_start = voyage_started_at
    if max_lookback is not None:
        sim_start = max(voyage_started_at, voyage_completed_at - max_lookback)

    sim_start, last_snapshot_ts = resolve_sim_start(voyage_id, sim_start, step, resume)
    if last_snapshot_ts is not None:
        log.info(
            "Voyage %d: resuming from %s (last snapshot %s)",
            voyage_id,
            sim_start.isoformat(),
            last_snapshot_ts.isoformat(),
        )
    if sim_start >= voyage_completed_at:
        log.info("Voyage %d: already complete in DB for step=%s", voyage_id, step)
        return 0

    duration_h = (voyage_completed_at - voyage_started_at).total_seconds() / 3600
    sim_h = (voyage_completed_at - sim_start).total_seconds() / 3600
    log.info("Voyage %d: duration=%.1fh, simulating last %.1fh", voyage_id, duration_h, sim_h)

    client = SnowflakeClient()
    runtime_client = None if dry_run else boto3.Session(profile_name=profile, region_name="us-east-1").client("sagemaker-runtime")
    try:
        try:
            full_spine = fetch_live_voyage_spine(client, voyage_id)
        except ValueError as e:
            log.warning("Voyage %d: spine fetch failed — %s", voyage_id, e)
            return 0

        n_legs_total = len(full_spine)
        leg_ids = full_spine["ID"].tolist()
        vessel_features_df = fetch_vessel_features(client, leg_ids)

        T = sim_start
        snapshots = 0

        while T < voyage_completed_at:
            masked = mask_spine_at_T(full_spine, T)
            masked = classify_leg_status(masked)

            try:
                current_leg = select_current_leg(masked)
            except ValueError:
                next_starts = [
                    row["ETA_UPDATED"]
                    for _, row in full_spine.iterrows()
                    if pd.notna(row.get("ETA_UPDATED")) and row["ETA_UPDATED"] > T
                ]
                T = min(next_starts) if next_starts else T + step
                continue

            n_legs_completed = int((masked["LEG_STATUS"] == "completed").sum())

            _run_snapshot(
                voyage_id=voyage_id,
                T=T,
                masked_df=masked,
                current_leg=current_leg,
                vessel_features_df=vessel_features_df,
                actual_completion_ts=voyage_completed_at,
                n_legs_total=n_legs_total,
                n_legs_completed=n_legs_completed,
                client=client,
                runtime_client=runtime_client,
                endpoint_name=endpoint_name,
                dry_run=dry_run,
            )
            snapshots += 1
            T += step

    finally:
        client.close()
        if runtime_client is not None and hasattr(runtime_client, "close"):
            runtime_client.close()

    log.info("Voyage %d: done, %d snapshots", voyage_id, snapshots)
    return snapshots


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    global _db_path

    parser = argparse.ArgumentParser(description="Backtest ETA model on historical voyages")
    parser.add_argument("--window-start", default="2026-10-10",
                        help="Start of completion window (inclusive), YYYY-MM-DD")
    parser.add_argument("--window-end", default="2026-10-20",
                        help="End of completion window (exclusive), YYYY-MM-DD")
    parser.add_argument("--step-minutes", type=int, default=60,
                        help="Snapshot interval in minutes (default: 60)")
    parser.add_argument("--endpoint", default="opentug-live-predict-eta-py312")
    parser.add_argument("--profile", default="opentug")
    parser.add_argument("--db", default="backtest.db",
                        help="Path to output SQLite database")
    parser.add_argument("--voyage-ids", type=int, nargs="+",
                        help="Run a specific subset of voyage IDs instead of the full window")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build payloads and store rows but skip endpoint calls")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the latest snapshot already stored in --db for each voyage")
    parser.add_argument("--workers", type=int, default=24,
                        help="Number of voyages to process in parallel (default: 24)")
    parser.add_argument("--max-lookback-days", type=float, default=None,
                        help="Cap simulation start to N days before completion (default: no cap)")
    parser.add_argument("--min-duration-h", type=float, default=None,
                        help="Only include voyages with duration >= this many hours")
    parser.add_argument("--max-duration-h", type=float, default=None,
                        help="Only include voyages with duration <= this many hours")
    args = parser.parse_args()

    _db_path = Path(args.db)
    main_conn = init_db(_db_path)
    main_conn.close()

    step = pd.Timedelta(minutes=args.step_minutes)
    max_lookback = pd.Timedelta(days=args.max_lookback_days) if args.max_lookback_days else None
    window_end_str = (pd.Timestamp(args.window_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00+00:00")
    window_start_str = pd.Timestamp(args.window_start).strftime("%Y-%m-%dT00:00:00+00:00")

    # Build the voyage list
    if args.voyage_ids:
        log.info("Fetching spine schedule for %d specified voyage IDs", len(args.voyage_ids))
        discovery_client = SnowflakeClient()
        voyage_rows = []
        try:
            for vid in args.voyage_ids:
                try:
                    from src.live_predict import fetch_live_voyage_spine
                    spine = fetch_live_voyage_spine(discovery_client, vid)
                    voyage_rows.append({
                        "VOYAGE_ID": vid,
                        "VOYAGE_STARTED_AT": spine["ETA_UPDATED"].dropna().min(),
                        "VOYAGE_COMPLETED_AT": spine["TIME_COMPLETED"].dropna().max(),
                        "N_LEGS": len(spine),
                    })
                except Exception as e:
                    log.warning("Voyage %d: discovery error — %s", vid, e)
        finally:
            discovery_client.close()
        voyages_df = pd.DataFrame(voyage_rows)
    else:
        log.info("Fetching completed voyages: %s → %s", window_start_str, window_end_str)
        discovery_client = SnowflakeClient()
        try:
            voyages_df = fetch_completed_voyages(discovery_client, window_start_str, window_end_str)
            voyages_df = voyages_df.rename(columns=str.upper)
        finally:
            discovery_client.close()

    voyages_df = voyages_df.dropna(subset=["VOYAGE_STARTED_AT", "VOYAGE_COMPLETED_AT"])

    # Ensure UTC-aware timestamps before any duration arithmetic
    for col in ["VOYAGE_STARTED_AT", "VOYAGE_COMPLETED_AT"]:
        voyages_df[col] = pd.to_datetime(voyages_df[col], utc=True, errors="coerce")
    voyages_df = voyages_df.dropna(subset=["VOYAGE_STARTED_AT", "VOYAGE_COMPLETED_AT"])

    # Duration filter
    voyages_df["_duration_h"] = (
        voyages_df["VOYAGE_COMPLETED_AT"] - voyages_df["VOYAGE_STARTED_AT"]
    ).dt.total_seconds() / 3600
    if args.min_duration_h is not None:
        before = len(voyages_df)
        voyages_df = voyages_df[voyages_df["_duration_h"] >= args.min_duration_h]
        log.info("Duration filter >= %.1fh: %d -> %d voyages", args.min_duration_h, before, len(voyages_df))
    if args.max_duration_h is not None:
        before = len(voyages_df)
        voyages_df = voyages_df[voyages_df["_duration_h"] <= args.max_duration_h]
        log.info("Duration filter <= %.1fh: %d -> %d voyages", args.max_duration_h, before, len(voyages_df))

    log.info("Voyages to backtest: %d (total duration %.0fh -> ~%d snapshots at %dmin steps)",
             len(voyages_df), voyages_df["_duration_h"].sum(),
             int(voyages_df["_duration_h"].sum() / (args.step_minutes / 60)),
             args.step_minutes)

    # Dispatch voyages — parallel if workers > 1
    total_snapshots = 0

    if args.workers == 1:
        for _, row in voyages_df.iterrows():
            n = run_voyage(
                int(row["VOYAGE_ID"]),
                row["VOYAGE_STARTED_AT"], row["VOYAGE_COMPLETED_AT"],
                args.endpoint, args.profile, step, args.dry_run, args.resume, max_lookback,
            )
            total_snapshots += n
    else:
        log.info("Running %d voyages with %d parallel workers", len(voyages_df), args.workers)
        futures = {}
        with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="voyage") as pool:
            for _, row in voyages_df.iterrows():
                f = pool.submit(
                    run_voyage,
                    int(row["VOYAGE_ID"]),
                    row["VOYAGE_STARTED_AT"], row["VOYAGE_COMPLETED_AT"],
                    args.endpoint, args.profile, step, args.dry_run, args.resume, max_lookback,
                )
                futures[f] = int(row["VOYAGE_ID"])
            for f in as_completed(futures):
                vid = futures[f]
                try:
                    total_snapshots += f.result()
                except Exception as exc:
                    log.error("Voyage %d failed: %s", vid, exc)

    log.info("Done. Total snapshots: %d → %s", total_snapshots, _db_path)
    print(f"\nBacktest complete: {total_snapshots} rows written to {_db_path}")


if __name__ == "__main__":
    main()
