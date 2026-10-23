
from snowflake_client import SnowflakeClient
from pathlib import Path

c = SnowflakeClient()
c.query_file(Path('sql/port_lookup.sql')).to_parquet('assets/lookups/port_lookup.parquet', index=False)
  