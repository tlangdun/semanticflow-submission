from __future__ import annotations

import csv
import json
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

from semanticflow.config import Settings
from semanticflow.dbt_integration._subprocess import (
    DEFAULT_COMMAND_TIMEOUT,
    RC_NOT_FOUND,
    RC_TIMEOUT,
    run_command,
)


def _resolve_mf_path(mf_cli_path: str) -> str:
    """Resolve the MetricFlow CLI path, preferring the active venv over PATH.

    The bare name ``mf`` collides with TeX Live's Metafont binary, which often
    sits ahead of the venv on PATH; ``shutil.which`` would then resolve to the
    wrong executable and every MetricFlow call would fail. So we look in the
    venv's bin/Scripts directory *before* consulting PATH.
    """
    # If it's already an absolute path, use it
    if Path(mf_cli_path).is_absolute() and Path(mf_cli_path).exists():
        return mf_cli_path

    # Prefer the venv's own MetricFlow so a same-named binary earlier on PATH
    # (e.g. TeX Live's `mf`/Metafont) cannot shadow it.
    venv_bin = Path(sys.executable).parent
    if sys.platform == "win32":
        for cand in (venv_bin / f"{mf_cli_path}.exe", venv_bin / mf_cli_path):
            if cand.exists():
                return str(cand)
    else:
        cand = venv_bin / mf_cli_path
        if cand.exists():
            return str(cand)

    # Fall back to PATH
    found = shutil.which(mf_cli_path)
    if found:
        return found

    # Fallback to original (will likely fail but with a clear error)
    return mf_cli_path


@dataclass
class MfCommandResult:
    command: list[str]
    stdout: str
    stderr: str
    returncode: int
    issues: list[dict[str, Any]]

    @property
    def success(self) -> bool:
        return self.returncode == 0


def _extract_issues(output: str) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for line in output.splitlines():
        lowered = line.lower()
        if "error" in lowered or "warning" in lowered:
            issues.append({"message": line.strip()})
    return issues


MF_COMMAND_TIMEOUT = DEFAULT_COMMAND_TIMEOUT  # a hung mf invocation otherwise blocks the run

_SAFE_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _is_safe_ident(name: Any) -> bool:
    """A bare SQL identifier (column/dimension/alias) — no quoting/dots needed."""
    return bool(_SAFE_IDENT.match(str(name)))


def _is_safe_fragment(value: Any) -> bool:
    """A relation ref or expression that may contain dots/quotes but must not break
    out of the statement (no terminators or comment markers)."""
    text = str(value)
    return not any(tok in text for tok in (";", "--", "/*", "*/"))


def _run(
    command: list[str],
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    timeout: float | None = MF_COMMAND_TIMEOUT,
) -> MfCommandResult:
    # mf needs shell=True on Windows (WinError 10106); run_command applies it only there.
    raw = run_command(command, env=env, cwd=cwd, timeout=timeout, use_shell=True)
    if raw.returncode in (RC_TIMEOUT, RC_NOT_FOUND):
        issues = [{"message": raw.stderr}]
    else:
        issues = _extract_issues(f"{raw.stdout}\n{raw.stderr}".strip())
    return MfCommandResult(
        command=raw.command,
        stdout=raw.stdout,
        stderr=raw.stderr,
        returncode=raw.returncode,
        issues=issues,
    )


def _swap_flag(command: list[str], old: str, new: str) -> list[str]:
    return [new if item == old else item for item in command]


def _strip_flags(command: list[str], flags: set[str]) -> list[str]:
    stripped: list[str] = []
    skip_next = False
    for item in command:
        if skip_next:
            skip_next = False
            continue
        if item in flags:
            skip_next = True
            continue
        stripped.append(item)
    return stripped


def _retry_with_alt_flags(
    result: MfCommandResult,
    command: list[str],
    env: dict[str, str] | None,
    cwd: str | None = None,
) -> tuple[MfCommandResult, list[str]]:
    updated = command
    if "No such option: --project-dir" in result.stderr:
        updated = _swap_flag(updated, "--project-dir", "--dbt-project-dir")
    if "No such option: --profiles-dir" in result.stderr:
        updated = _swap_flag(updated, "--profiles-dir", "--dbt-profiles-dir")
    if updated != command:
        return _run(updated, env=env, cwd=cwd), updated
    return result, command


def _retry_without_project_flags(
    result: MfCommandResult,
    command: list[str],
    project_dir: str,
    env: dict[str, str] | None,
) -> tuple[MfCommandResult, list[str]]:
    stderr = result.stderr or ""
    stdout = result.stdout or ""
    combined = f"{stderr}\n{stdout}"
    missing_project = (
        "No such option: --project-dir" in stderr
        or "No such option: --dbt-project-dir" in stderr
        or "No such option: --profiles-dir" in stderr
        or "No such option: --dbt-profiles-dir" in stderr
        or "No such option: --profile" in stderr
        or "Missing:" in combined
        or "dbt_project.yml" in combined
    )
    if not missing_project:
        return result, command

    # Resolve to absolute path to ensure mf CLI can find the project
    resolved_project_dir = str(Path(project_dir).resolve())
    fallback_env = dict(env or {})
    fallback_env.setdefault("DBT_PROFILES_DIR", resolved_project_dir)
    fallback_env.setdefault("DBT_PROJECT_DIR", resolved_project_dir)
    fallback_command = _strip_flags(
        command,
        {"--project-dir", "--dbt-project-dir", "--profiles-dir", "--dbt-profiles-dir"},
    )
    if "No such option: --profile" in stderr:
        fallback_command = _strip_flags(fallback_command, {"--profile"})
    
    return _run(fallback_command, env=fallback_env, cwd=resolved_project_dir), fallback_command


def run_mf_validate(project_dir: str, settings: Settings, env: dict[str, str] | None = None) -> MfCommandResult:
    resolved_project_dir = str(Path(project_dir).resolve())
    env = dict(env or {})
    # Always use absolute paths - override any relative paths passed by caller
    env["DBT_PROFILES_DIR"] = resolved_project_dir
    env["DBT_PROJECT_DIR"] = resolved_project_dir
    if settings.mf_cli_mode == "dbt_sl":
        command = [
            "dbt",
            "sl",
            "validate",
            "--project-dir",
            resolved_project_dir,
            "--profiles-dir",
            resolved_project_dir,
        ]
        if settings.dbt_profile_name:
            command.extend(["--profile", settings.dbt_profile_name])
        result = _run(command, env=env, cwd=resolved_project_dir)
        result, command_used = _retry_with_alt_flags(result, command, env, cwd=resolved_project_dir)
        result, _ = _retry_without_project_flags(result, command_used, resolved_project_dir, env)
        return result

    mf_path = _resolve_mf_path(settings.mf_cli_path)
    command = [
        mf_path,
        "validate-configs",
        "--project-dir",
        resolved_project_dir,
        "--profiles-dir",
        resolved_project_dir,
    ]

    if settings.mf_validate_warehouse:
        command.append("--warehouse")
    result = _run(command, env=env, cwd=resolved_project_dir)
    result, command_used = _retry_with_alt_flags(result, command, env, cwd=resolved_project_dir)
    result, _ = _retry_without_project_flags(result, command_used, resolved_project_dir, env)
    return result


def _parse_mf_query_output(output: str) -> list[dict[str, Any]]:
    cleaned = output.strip()
    if not cleaned:
        return []
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            rows = payload.get("rows")
            if isinstance(rows, list):
                return [item for item in rows if isinstance(item, dict)]
            return [payload]
    except json.JSONDecodeError:
        pass

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if len(lines) < 2:
        return []
    if "," in lines[0]:
        headers = [item.strip() for item in lines[0].split(",")]
        results = []
        for line in lines[1:]:
            values = [item.strip() for item in line.split(",")]
            if len(values) != len(headers):
                continue
            results.append(dict(zip(headers, values)))
        return results
    return []


def _parse_csv_file(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [row for row in reader]


def _run_duckdb_fallback(
    project_dir: str,
    metric_name: str,
    dimensions: list[str] | None = None,
    time_granularity: str | None = None,
    limit: int | None = None,
    order_by: list[dict[str, str]] | None = None,
) -> tuple[MfCommandResult, list[dict[str, Any]]]:
    """Direct DuckDB query fallback when mf CLI fails (Windows workaround)."""
    if not HAS_DUCKDB:
        return MfCommandResult(
            command=["duckdb-fallback"],
            stdout="",
            stderr="DuckDB not installed",
            returncode=1,
            issues=[{"message": "DuckDB not installed"}],
        ), []

    project_path = Path(project_dir).resolve()
    manifest_path = project_path / "target" / "semantic_manifest.json"
    
    if not manifest_path.exists():
        return MfCommandResult(
            command=["duckdb-fallback"],
            stdout="",
            stderr=f"Semantic manifest not found: {manifest_path}",
            returncode=1,
            issues=[{"message": "Semantic manifest not found"}],
        ), []

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        return MfCommandResult(
            command=["duckdb-fallback"],
            stdout="",
            stderr=f"Failed to parse semantic manifest: {e}",
            returncode=1,
            issues=[{"message": str(e)}],
        ), []

    # Find the metric definition
    metric_def = None
    for m in manifest.get("metrics", []):
        if m.get("name") == metric_name:
            metric_def = m
            break
    
    if not metric_def:
        return MfCommandResult(
            command=["duckdb-fallback"],
            stdout="",
            stderr=f"Metric '{metric_name}' not found in manifest",
            returncode=1,
            issues=[{"message": f"Metric not found: {metric_name}"}],
        ), []

    # Find the measure and semantic model
    measure_name = metric_def.get("type_params", {}).get("measure", {}).get("name")
    if not measure_name:
        measures = metric_def.get("type_params", {}).get("input_measures", [])
        if measures:
            measure_name = measures[0].get("name")
    
    # Find the semantic model with this measure
    semantic_model = None
    measure_def = None
    for sm in manifest.get("semantic_models", []):
        for m in sm.get("measures", []):
            if m.get("name") == measure_name:
                semantic_model = sm
                measure_def = m
                break
        if semantic_model:
            break

    if not semantic_model or not measure_def:
        return MfCommandResult(
            command=["duckdb-fallback"],
            stdout="",
            stderr=f"Could not find semantic model for measure '{measure_name}'",
            returncode=1,
            issues=[{"message": f"Semantic model not found for measure: {measure_name}"}],
        ), []

    # Build SQL query
    table_ref = semantic_model.get("node_relation", {}).get("relation_name", "")
    if not table_ref:
        table_ref = f'"{semantic_model.get("name", "unknown")}"'
    
    # Reject identifiers/expressions that could break out of the statement; this
    # fallback interpolates manifest- and spec-derived strings directly into SQL.
    if not _is_safe_fragment(table_ref):
        return MfCommandResult(
            command=["duckdb-fallback"], stdout="",
            stderr=f"Unsafe table reference rejected: {table_ref!r}",
            returncode=1, issues=[{"message": "unsafe table reference"}],
        ), []

    agg = measure_def.get("agg", "count")
    expr = measure_def.get("expr", "*")
    if not _is_safe_fragment(expr):
        expr = "*"
    safe_measure_alias = measure_name if _is_safe_ident(measure_name) else "measure_value"

    # Build aggregation expression
    if agg == "count":
        agg_expr = f"COUNT({expr})"
    elif agg == "sum":
        agg_expr = f"SUM({expr})"
    elif agg == "avg":
        agg_expr = f"AVG({expr})"
    elif agg == "min":
        agg_expr = f"MIN({expr})"
    elif agg == "max":
        agg_expr = f"MAX({expr})"
    else:
        agg_expr = f"COUNT({expr})"

    # Build group by
    group_cols = []
    select_cols = []
    
    # Handle dimensions
    dims = dimensions or []
    if not dims and time_granularity:
        dims = [f"metric_time__{time_granularity}"]
    
    for dim in dims:
        # Handle metric_time__day format
        if dim.startswith("metric_time__"):
            # Find the time dimension from semantic model
            time_dim = None
            for d in semantic_model.get("dimensions", []):
                if d.get("type") == "time":
                    time_dim = d.get("expr", d.get("name"))
                    break
            if time_dim and _is_safe_ident(time_dim) and _is_safe_ident(dim):
                select_cols.append(f"{time_dim} AS {dim}")
                group_cols.append(time_dim)
        elif _is_safe_ident(dim):
            select_cols.append(dim)
            group_cols.append(dim)

    select_cols.append(f"{agg_expr} AS {safe_measure_alias}")
    
    sql = f"SELECT {', '.join(select_cols)} FROM {table_ref}"
    if group_cols:
        sql += f" GROUP BY {', '.join(group_cols)}"
    
    # Order by
    if order_by:
        order_parts = []
        for item in order_by:
            if not isinstance(item, dict):
                continue
            field = item.get("field") or item.get("name")
            if not field:
                continue
            # Map field to actual column
            if field.startswith("metric_time") or field == "order_date":
                for d in semantic_model.get("dimensions", []):
                    if d.get("type") == "time":
                        field = d.get("expr", d.get("name"))
                        break
            if not _is_safe_ident(field):
                continue
            direction = "DESC" if str(item.get("direction", "asc")).lower() == "desc" else "ASC"
            order_parts.append(f"{field} {direction}")
        if order_parts:
            sql += f" ORDER BY {', '.join(order_parts)}"
    
    if limit:
        try:
            sql += f" LIMIT {int(limit)}"
        except (TypeError, ValueError):
            pass

    # Find DuckDB database path
    db_path = project_path / "jaffle_shop.duckdb"
    if not db_path.exists():
        db_path = project_path / "dev.duckdb"
    if not db_path.exists():
        # Try to find any .duckdb file
        duckdb_files = list(project_path.glob("*.duckdb"))
        if duckdb_files:
            db_path = duckdb_files[0]
        else:
            return MfCommandResult(
                command=["duckdb-fallback", sql],
                stdout="",
                stderr="No DuckDB database found",
                returncode=1,
                issues=[{"message": "No DuckDB database found"}],
            ), []

    try:
        conn = duckdb.connect(str(db_path), read_only=True)
        cursor = conn.execute(sql)
        columns = [desc[0] for desc in cursor.description]
        raw_rows = cursor.fetchall()
        conn.close()
        
        rows = []
        for raw_row in raw_rows:
            row = {}
            for i, col in enumerate(columns):
                val = raw_row[i]
                # Convert date/datetime to string
                if hasattr(val, "isoformat"):
                    row[col] = val.isoformat()
                elif hasattr(val, "item"):  # numpy types
                    row[col] = val.item()
                else:
                    row[col] = val
            rows.append(row)
        
        return MfCommandResult(
            command=["duckdb-fallback", sql],
            stdout=f"DuckDB fallback query executed: {sql}\nRows: {len(rows)}",
            stderr="",
            returncode=0,
            issues=[],
        ), rows
    except Exception as e:
        return MfCommandResult(
            command=["duckdb-fallback", sql],
            stdout="",
            stderr=f"DuckDB query failed: {e}",
            returncode=1,
            issues=[{"message": str(e)}],
        ), []


def run_mf_query(
    project_dir: str,
    settings: Settings,
    metric_name: str,
    dimensions: list[str] | None = None,
    time_granularity: str | None = None,
    where: str | None = None,
    limit: int | None = None,
    order_by: list[dict[str, str]] | None = None,
    env: dict[str, str] | None = None,
) -> tuple[MfCommandResult, list[dict[str, Any]]]:
    resolved_project_dir = str(Path(project_dir).resolve())
    env = dict(env or {})
    # Always use absolute paths - override any relative paths passed by caller
    env["DBT_PROFILES_DIR"] = resolved_project_dir
    env["DBT_PROJECT_DIR"] = resolved_project_dir
    mf_path = _resolve_mf_path(settings.mf_cli_path)
    command = [
        mf_path,
        "query",
        "--project-dir",
        resolved_project_dir,
        "--profiles-dir",
        resolved_project_dir,
        "--metrics",
        metric_name,
    ]

    group_by = list(dimensions or [])
    if time_granularity and not group_by:
        group_by = [f"metric_time__{time_granularity}"]
    if group_by:
        command.extend(["--group-by", ",".join(group_by)])
    if where:
        command.extend(["--where", where])
    if limit:
        command.extend(["--limit", str(limit)])
    if order_by:
        order_fields: list[str] = []
        for item in order_by:
            if not isinstance(item, dict):
                continue
            field = item.get("field") or item.get("name")
            if not field:
                continue
            direction = str(item.get("direction", "asc")).lower()
            if direction.startswith("desc"):
                order_fields.append(f"-{field}")
            else:
                order_fields.append(field)
        if order_fields:
            command.extend(["--order", ",".join(order_fields)])
    csv_path: Path | None = None
    if settings.mf_query_output:
        csv_dir = Path(resolved_project_dir) / "target"
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_path = csv_dir / f"mf_query_{metric_name}_{int(time.time())}.csv"
        command.extend(["--csv", str(csv_path)])

    result = _run(command, env=env, cwd=resolved_project_dir)
    result, command_used = _retry_with_alt_flags(result, command, env, cwd=resolved_project_dir)
    result, command_used = _retry_without_project_flags(result, command_used, resolved_project_dir, env)
    if "No such option: --group-by" in result.stderr:
        command_used = _swap_flag(command_used, "--group-by", "--dimensions")
        result = _run(command_used, env=env, cwd=resolved_project_dir)
    if "No such option: --dimensions" in result.stderr:
        command_used = _swap_flag(command_used, "--dimensions", "--group-by")
        result = _run(command_used, env=env, cwd=resolved_project_dir)

    rows = _parse_csv_file(csv_path) if csv_path else _parse_mf_query_output(result.stdout)

    if csv_path and csv_path.exists():
        try:
            csv_path.unlink()
        except OSError:
            pass

    # If mf CLI failed, optionally try the DuckDB fallback. OFF by default: the fallback
    # ignores WHERE filters and re-implements MF semantics, so it can return silently-wrong
    # results — an honest failure is preferable for evaluation.
    if not result.success and HAS_DUCKDB and getattr(settings, "enable_duckdb_fallback", False):
        fallback_result, fallback_rows = _run_duckdb_fallback(
            project_dir=resolved_project_dir,
            metric_name=metric_name,
            dimensions=dimensions,
            time_granularity=time_granularity,
            limit=limit,
            order_by=order_by,
        )
        if fallback_result.success:
            return fallback_result, fallback_rows

    return result, rows
