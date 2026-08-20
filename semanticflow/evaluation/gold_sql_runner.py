from __future__ import annotations

from typing import Any

import duckdb


def run_gold_sql(sql: str, duckdb_path: str | None = None) -> tuple[bool, list[dict[str, Any]], str | None]:
    if not sql:
        return False, [], "gold_sql is empty"
    conn = None
    try:
        conn = duckdb.connect(duckdb_path, read_only=True) if duckdb_path else duckdb.connect()
        cursor = conn.execute(sql)
        columns = [col[0] for col in cursor.description] if cursor.description else []
        rows = [dict(zip(columns, record)) for record in cursor.fetchall()]
        return True, rows, None
    except Exception as exc:  # pragma: no cover - runtime dependent
        return False, [], str(exc)
    finally:
        if conn is not None:
            conn.close()
