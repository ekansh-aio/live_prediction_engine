"""
Snowflake session daemon — start once, share across all script runs.

Usage:
    python sf_daemon.py                          # uses .env / SF_* env vars
    python sf_daemon.py --account x --user y ...

The daemon authenticates with MFA once, then holds the Snowflake session open
and listens on localhost:17432.  Any process (including snowflake_explore.py)
can send SQL over that socket without needing credentials or a TOTP code.

Stop the daemon:  Ctrl-C in its terminal, or close VS Code.
"""

import datetime
import decimal
import json
import os
import signal
import socket
import sys
import threading
from pathlib import Path
from typing import Optional


class _Encoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, decimal.Decimal):
            return float(o)
        if isinstance(o, (datetime.datetime, datetime.date)):
            return o.isoformat()
        return super().default(o)

# ── load .env ────────────────────────────────────────────────────────────────
def _load_dotenv():
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.split("#")[0].strip())

_load_dotenv()

try:
    import snowflake.connector
    import pandas as pd
except ImportError:
    print("Missing deps — run: pip install snowflake-connector-python pandas")
    sys.exit(1)

import argparse

DAEMON_HOST = "127.0.0.1"
DAEMON_PORT = 17432
SOCKET_FILE = Path(__file__).parent / ".sf_daemon.port"


def _quote_ident(value):
    return '"' + str(value).replace('"', '""') + '"'


def _connect(account, user, password, warehouse, database, schema, role, passcode):
    kwargs = dict(
        account=account, user=user, password=password,
        passcode=passcode, authenticator="username_password_mfa",
        warehouse=warehouse, database=database, schema=schema,
        login_timeout=60,
    )
    if role:
        kwargs["role"] = role
    conn = snowflake.connector.connect(**kwargs)
    print(f"[daemon] Connected to {account} | {database}.{schema}" + (f" | role={role}" if role else ""))
    return conn


def _apply_session_context(conn, warehouse=None, database=None, schema=None, role=None):
    with conn.cursor() as cur:
        if role:
            cur.execute(f"USE ROLE {_quote_ident(role)}")
        if warehouse:
            cur.execute(f"USE WAREHOUSE {_quote_ident(warehouse)}")
        if database:
            cur.execute(f"USE DATABASE {_quote_ident(database)}")
        if schema:
            cur.execute(f"USE SCHEMA {_quote_ident(schema)}")
        cur.execute("select current_warehouse(), current_database(), current_schema(), current_role()")
        current_warehouse, current_database, current_schema, current_role = cur.fetchone()
    if warehouse and current_warehouse != warehouse:
        raise RuntimeError(
            f"Failed to activate warehouse {warehouse!r}; current session has {current_warehouse!r}"
        )
    return {
        "warehouse": current_warehouse,
        "database": current_database,
        "schema": current_schema,
        "role": current_role,
    }


def _keep_alive(conn, stop_event):
    """Ping Snowflake every 4 minutes so the session doesn't idle out (default 5 min)."""
    while not stop_event.wait(240):
        try:
            conn.cursor().execute("SELECT 1")
        except Exception as e:
            print(f"[daemon] keep-alive failed: {e}")


_FETCH_BATCH = 5000  # rows per NDJSON batch sent to client


def _handle_client(client_sock: socket.socket, conn: snowflake.connector.SnowflakeConnection, session_ctx: dict):
    """Handle one client connection: read a JSON request, stream NDJSON response.

    Protocol (newline-delimited JSON, one object per line):
      Line 1:  {"ok": true, "columns": [...]}         -- SELECT header
               {"ok": false, "error": "..."}           -- error (any query)
               {"ok": true, "columns": [], "rowcount": N} -- non-SELECT
      Lines 2+: {"rows": [[...], ...]}                 -- batches of rows
      Last:    {"done": true}                          -- end of stream
    """
    try:
        req_buf = b""
        client_sock.settimeout(600)
        while True:
            chunk = client_sock.recv(65536)
            if not chunk:
                break
            req_buf += chunk
            if req_buf.endswith(b"\n"):
                break

        req = json.loads(req_buf.decode())
        sql = req.get("sql", "").strip()
        params = req.get("params") or None

        _apply_session_context(conn, **session_ctx)

        with conn.cursor() as cur:
            cur.execute(sql, params)
            if cur.description:
                cols = [d[0] for d in cur.description]
                header = json.dumps({"ok": True, "columns": cols}, cls=_Encoder) + "\n"
                client_sock.sendall(header.encode())
                while True:
                    batch = cur.fetchmany(_FETCH_BATCH)
                    if not batch:
                        break
                    line = json.dumps({"rows": [list(r) for r in batch]}, cls=_Encoder) + "\n"
                    client_sock.sendall(line.encode())
                client_sock.sendall(b'{"done":true}\n')
            else:
                resp = {"ok": True, "columns": [], "rows": [], "rowcount": cur.rowcount}
                client_sock.sendall((json.dumps(resp, cls=_Encoder) + "\n").encode())
                client_sock.sendall(b'{"done":true}\n')

    except Exception as e:
        try:
            client_sock.sendall((json.dumps({"ok": False, "error": str(e)}) + "\n").encode())
        except Exception:
            pass
    finally:
        client_sock.close()


def run_daemon(account, user, password, warehouse, database, schema, role):
    conn = None
    for attempt in range(1, 4):
        # ponytail: prompt fresh each try — TOTP expires in 30s, so read the
        # code and hit Enter immediately; a stale code = "TOTP Invalid".
        passcode = input(f"Enter fresh MFA/TOTP code (attempt {attempt}/3, type fast): ").strip()
        try:
            conn = _connect(account, user, password, warehouse, database, schema, role, passcode)
            break
        except Exception as e:
            msg = str(e)
            if "TOTP" in msg or "passcode" in msg.lower() or "MFA" in msg:
                print(f"[daemon] TOTP rejected — wait for a NEW code, then retry. ({e})")
                continue
            raise
    if conn is None:
        print("[daemon] 3 bad TOTPs — stopped to avoid lockout. Check phone clock / authenticator.")
        sys.exit(1)
    session_ctx = {
        "warehouse": warehouse,
        "database": database,
        "schema": schema,
        "role": role,
    }
    active_ctx = _apply_session_context(conn, **session_ctx)
    print(
        "[daemon] Session context "
        f"warehouse={active_ctx['warehouse']} database={active_ctx['database']} "
        f"schema={active_ctx['schema']} role={active_ctx['role']}"
    )

    stop_event = threading.Event()
    ka_thread = threading.Thread(target=_keep_alive, args=(conn, stop_event), daemon=True)
    ka_thread.start()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((DAEMON_HOST, DAEMON_PORT))
    server.listen(8)
    SOCKET_FILE.write_text(str(DAEMON_PORT))

    print(f"[daemon] Listening on {DAEMON_HOST}:{DAEMON_PORT}  (Ctrl-C to stop)")

    def _shutdown(sig, frame):
        print("\n[daemon] Shutting down …")
        stop_event.set()
        server.close()
        SOCKET_FILE.unlink(missing_ok=True)
        try:
            conn.close()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    while True:
        try:
            client_sock, _ = server.accept()
        except OSError:
            break
        t = threading.Thread(target=_handle_client, args=(client_sock, conn, session_ctx), daemon=True)
        t.start()


def main():
    p = argparse.ArgumentParser(description="Snowflake session daemon")
    p.add_argument("--account",   default=os.getenv("SF_ACCOUNT"))
    p.add_argument("--user",      default=os.getenv("SF_USER"))
    p.add_argument("--password",  default=os.getenv("SF_PASSWORD"))
    p.add_argument("--warehouse", default=os.getenv("SF_WAREHOUSE", "WH_ETLDEV_XS"))
    p.add_argument("--database",  default=os.getenv("SF_DATABASE",  "EDPPROD"))
    p.add_argument("--schema",    default=os.getenv("SF_SCHEMA",    "EDP_ANALYTICS"))
    p.add_argument("--role",      default=os.getenv("SF_ROLE"))
    args = p.parse_args()

    missing = [n for n, v in [("--account", args.account), ("--user", args.user), ("--password", args.password)] if not v]
    if missing:
        print(f"Missing credentials: {', '.join(missing)}")
        print("Set SF_ACCOUNT / SF_USER / SF_PASSWORD env vars or pass as flags.")
        sys.exit(1)

    run_daemon(args.account, args.user, args.password,
               args.warehouse, args.database, args.schema, args.role)


if __name__ == "__main__":
    main()
