import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sagemaker import backtest


class BacktestResumeTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "resume.db"
        conn = backtest.init_db(self.db_path)
        conn.close()
        backtest._db_path = self.db_path
        if hasattr(backtest._db_local, "conn"):
            try:
                backtest._db_local.conn.close()
            except Exception:
                pass
            backtest._db_local.conn = None

    def tearDown(self):
        if hasattr(backtest._db_local, "conn") and backtest._db_local.conn is not None:
            backtest._db_local.conn.close()
            backtest._db_local.conn = None
        self.tempdir.cleanup()

    def test_resolve_sim_start_uses_last_snapshot_plus_step(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO backtest (voyage_id, snapshot_ts, actual_completion_ts) VALUES (?,?,?)",
            (123, "2026-10-10T12:00:00+00:00", "2026-10-15T00:00:00+00:00"),
        )
        conn.execute(
            "INSERT INTO backtest (voyage_id, snapshot_ts, actual_completion_ts) VALUES (?,?,?)",
            (123, "2026-10-10T14:00:00+00:00", "2026-10-15T00:00:00+00:00"),
        )
        conn.commit()
        conn.close()

        sim_start = pd.Timestamp("2026-10-10T00:00:00+00:00")
        step = pd.Timedelta(minutes=120)
        resumed_start, last_snapshot = backtest.resolve_sim_start(123, sim_start, step, resume=True)

        self.assertEqual(last_snapshot, pd.Timestamp("2026-10-10T14:00:00+00:00"))
        self.assertEqual(resumed_start, pd.Timestamp("2026-10-10T16:00:00+00:00"))

    def test_resolve_sim_start_uses_last_snapshot_plus_new_step(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO backtest (voyage_id, snapshot_ts, actual_completion_ts) VALUES (?,?,?)",
            (456, "2026-09-30T19:30:00+00:00", "2026-10-11T19:06:41+00:00"),
        )
        conn.commit()
        conn.close()

        sim_start = pd.Timestamp("2026-09-30T20:30:00+00:00")
        step = pd.Timedelta(minutes=120)
        resumed_start, last_snapshot = backtest.resolve_sim_start(456, sim_start, step, resume=True)

        self.assertEqual(last_snapshot, pd.Timestamp("2026-09-30T19:30:00+00:00"))
        self.assertEqual(resumed_start, pd.Timestamp("2026-09-30T21:30:00+00:00"))

    def test_resolve_sim_start_no_resume_keeps_original_start(self):
        sim_start = pd.Timestamp("2026-10-10T00:00:00+00:00")
        step = pd.Timedelta(minutes=120)
        resumed_start, last_snapshot = backtest.resolve_sim_start(999, sim_start, step, resume=False)

        self.assertEqual(resumed_start, sim_start)
        self.assertIsNone(last_snapshot)


if __name__ == "__main__":
    unittest.main()
