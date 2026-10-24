from __future__ import annotations

import json
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "predictions" / "predictions.db"


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS voyage_prediction_runs (
            run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
            voyage_id       INTEGER NOT NULL,
            as_of_utc       TEXT    NOT NULL,
            current_leg_id  INTEGER NOT NULL,
            consolidated_eta_utc TEXT,
            eta_p10_utc          TEXT,
            eta_p90_utc          TEXT,
            total_remaining_h    REAL,
            raw_total_remaining_h REAL,
            bias_adj_h           REAL,
            current_remaining_h  REAL,
            future_legs_duration_h REAL,
            future_dwell_h       REAL,
            current_marker_wait_h REAL,
            future_marker_wait_h  REAL,
            result_json          TEXT NOT NULL,
            created_at           TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS voyage_prediction_legs (
            row_id           INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id           INTEGER NOT NULL,
            voyage_id        INTEGER NOT NULL,
            leg_id           INTEGER NOT NULL,
            sequence_num     INTEGER NOT NULL,
            role             TEXT    NOT NULL,
            leg_status       TEXT    NOT NULL,
            model_a_pred_h        REAL,
            dynamic_pred_h        REAL,
            flc_residual_h        REAL,
            flc_corrected_h       REAL,
            model_a_features_json TEXT,
            dynamic_features_json TEXT,
            flc_features_json     TEXT,
            created_at            TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_run_id ON voyage_prediction_legs(run_id);
    """)
    conn.commit()


def store_to_sqlite(result: dict) -> int:
    """Store a prediction result in SQLite. Returns the SQLite run_id."""
    voyage = result.get("voyage", {})
    conn = _connect()
    try:
        _init_db(conn)
        cur = conn.execute(
            """
            INSERT INTO voyage_prediction_runs (
                voyage_id, as_of_utc, current_leg_id,
                consolidated_eta_utc, eta_p10_utc, eta_p90_utc,
                total_remaining_h, raw_total_remaining_h, bias_adj_h,
                current_remaining_h, future_legs_duration_h, future_dwell_h,
                current_marker_wait_h, future_marker_wait_h, result_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                result["voyage_id"],
                result["as_of_utc"],
                voyage.get("current_subservice_id"),
                voyage.get("consolidated_eta"),
                voyage.get("eta_p10"),
                voyage.get("eta_p90"),
                voyage.get("total_remaining_h"),
                voyage.get("raw_total_remaining_h"),
                voyage.get("bias_adj_h"),
                voyage.get("current_remaining_h"),
                voyage.get("future_legs_duration_h"),
                voyage.get("future_dwell_h"),
                voyage.get("current_marker_wait_h"),
                voyage.get("future_marker_wait_h"),
                json.dumps(result),
            ),
        )
        run_id: int = cur.lastrowid  # type: ignore[assignment]
        for leg in result.get("legs", []):
            conn.execute(
                """
                INSERT INTO voyage_prediction_legs (
                    run_id, voyage_id, leg_id, sequence_num, role, leg_status,
                    model_a_pred_h, dynamic_pred_h, flc_residual_h, flc_corrected_h,
                    model_a_features_json, dynamic_features_json, flc_features_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    run_id,
                    result["voyage_id"],
                    leg["leg_id"],
                    leg["sequence_num"],
                    leg["role"],
                    leg["leg_status"],
                    leg.get("model_a_pred_h"),
                    leg.get("dynamic_pred_h"),
                    leg.get("flc_residual_h"),
                    leg.get("flc_corrected_h"),
                    json.dumps(leg["model_a_features"]) if leg.get("model_a_features") else None,
                    json.dumps(leg["dynamic_features"]) if leg.get("dynamic_features") else None,
                    json.dumps(leg["flc_features"]) if leg.get("flc_features") else None,
                ),
            )
        conn.commit()
        return run_id
    finally:
        conn.close()
