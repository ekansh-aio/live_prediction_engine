import unittest

from sf_daemon import _apply_session_context


class _FakeCursor:
    def __init__(self, executed):
        self.executed = executed

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql):
        self.executed.append(sql)


class _FakeConn:
    def __init__(self):
        self.executed = []

    def cursor(self):
        return _FakeCursor(self.executed)


class ApplySessionContextTests(unittest.TestCase):
    def test_applies_configured_session_context(self):
        conn = _FakeConn()

        _apply_session_context(
            conn,
            warehouse="COMPUTE_WH",
            database="EDPPROD",
            schema="EDP_SOURCE",
            role="ANALYST",
        )

        self.assertEqual(
            conn.executed,
            [
                'USE ROLE "ANALYST"',
                'USE WAREHOUSE "COMPUTE_WH"',
                'USE DATABASE "EDPPROD"',
                'USE SCHEMA "EDP_SOURCE"',
            ],
        )


if __name__ == "__main__":
    unittest.main()
