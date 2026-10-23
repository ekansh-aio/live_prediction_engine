from __future__ import annotations

import logging
import time
import json
import socket
import pandas as pd

import src.config as config

log = logging.getLogger("snowflake")


class SnowflakeClient:
    def __init__(self):
        self.mode = config.SF_MODE or "daemon"
        self._conn = None

    def _connect_direct(self):
        if self._conn is not None:
            return self._conn
        import snowflake.connector

        kwargs = {
            "account": config.env("SF_ACCOUNT"),
            "user": config.env("SF_USER"),
            "password": config.env("SF_PASSWORD"),
            "warehouse": config.env("SF_WAREHOUSE", "COMPUTE_WH"),
            "database": config.env("SF_DATABASE", "EDPPROD"),
            "schema": config.env("SF_SCHEMA", "EDP_SOURCE"),
            "login_timeout": 60,
        }
        if config.env("SF_ROLE"):
            kwargs["role"] = config.env("SF_ROLE")
        self._conn = snowflake.connector.connect(**kwargs)
        return self._conn

    def _query_daemon(self, sql: str, params: dict | None = None) -> pd.DataFrame:
        host = config.env("SF_DAEMON_HOST", "127.0.0.1")
        port = int(config.env("SF_DAEMON_PORT", "17432"))
        req = json.dumps({"sql": sql, "params": params or {}}).encode() + b"\n"
        timeout = int(config.env("SF_DAEMON_TIMEOUT") or "30")
        sock = socket.create_connection((host, port), timeout=timeout)
        try:
            sock.sendall(req)
            fh = sock.makefile("r", encoding="utf-8")
            header = json.loads(fh.readline())
            if not header.get("ok"):
                raise RuntimeError(header.get("error", "Snowflake daemon error"))
            cols = header.get("columns", [])
            rows = []
            for line in fh:
                msg = json.loads(line)
                if msg.get("done"):
                    break
                rows.extend(msg.get("rows", []))
            return pd.DataFrame(rows, columns=cols)
        finally:
            sock.close()

    def query_text(self, sql: str, params: dict | None = None) -> pd.DataFrame:
        log.debug("Querying Snowflake (mode=%s) sql_len=%d params=%s", self.mode, len(sql), params)
        t0 = time.time()
        if self.mode == "daemon":
            df = self._query_daemon(sql, params)
        else:
            conn = self._connect_direct()
            cur = conn.cursor()
            try:
                cur.execute(sql, params or {})
                if not cur.description:
                    return pd.DataFrame()
                cols = [d[0] for d in cur.description]
                df = pd.DataFrame(cur.fetchall(), columns=cols)
            finally:
                cur.close()
        log.info("Snowflake returned %d rows in %.2fs (mode=%s)", len(df), time.time() - t0, self.mode)
        return df

    def query_file(self, path: Path, params: dict | None = None) -> pd.DataFrame:
        log.debug("Querying Snowflake from file %s", path)
        return self.query_text(path.read_text(encoding="utf-8"), params)

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None
