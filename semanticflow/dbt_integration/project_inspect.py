from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml


def load_manifest(project_dir: str) -> dict[str, Any] | None:
    manifest_path = Path(project_dir) / "target" / "manifest.json"
    if not manifest_path.exists():
        return None
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def list_models_and_columns(project_dir: str) -> dict[str, list[str]]:
    manifest = load_manifest(project_dir)
    if not manifest:
        return {}
    models: dict[str, list[str]] = {}
    for node in manifest.get("nodes", {}).values():
        if node.get("resource_type") != "model":
            continue
        name = node.get("name")
        columns = node.get("columns", {})
        if not name:
            continue
        models[name] = list(columns.keys())
    return models


def list_models_and_types(project_dir: str) -> dict[str, dict[str, str | None]]:
    manifest = load_manifest(project_dir)
    if not manifest:
        return {}
    models: dict[str, dict[str, str | None]] = {}
    for node in manifest.get("nodes", {}).values():
        if node.get("resource_type") != "model":
            continue
        name = node.get("name")
        columns = node.get("columns", {})
        if not name:
            continue
        models[name] = {col: info.get("data_type") for col, info in columns.items()}
    return models


def _iter_yaml_files(root: Path, subdirs: Iterable[str]) -> Iterable[Path]:
    for subdir in subdirs:
        base = root / subdir
        if not base.exists():
            continue
        for path in base.rglob("*.yml"):
            yield path
        for path in base.rglob("*.yaml"):
            yield path


def _safe_load_yaml(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError:
        return None
    if not isinstance(data, dict):
        return None
    return data


def semantic_model_exists(project_dir: str, name: str) -> bool:
    root = Path(project_dir)
    for path in _iter_yaml_files(root, ["models"]):
        data = _safe_load_yaml(path)
        if not data:
            continue
        for spec in data.get("semantic_models", []) or []:
            if isinstance(spec, dict) and spec.get("name") == name:
                return True
    return False


def metric_exists(project_dir: str, name: str) -> bool:
    root = Path(project_dir)
    # Check both models and metrics directories (metrics.yml can be in models/semantic/)
    for path in _iter_yaml_files(root, ["models", "metrics"]):
        data = _safe_load_yaml(path)
        if not data:
            continue
        for spec in data.get("metrics", []) or []:
            if isinstance(spec, dict) and spec.get("name") == name:
                return True
    return False


def extract_schema_context(project_dir: str) -> dict[str, Any]:
    models = list_models_and_columns(project_dir)
    data_types = list_models_and_types(project_dir)
    return {
        "models": sorted(models.keys()),
        "columns": models,
        "data_types": data_types,
    }


_DUCKDB_TYPES_CACHE: dict[str, dict[str, dict[str, str]]] = {}


def duckdb_data_types(project_dir: str) -> dict[str, dict[str, str]]:
    """Column data types per table, read from the project's DuckDB warehouse.

    The dbt manifest carries ``data_type`` only when a catalog was generated, and the
    benchmark tasks' static ``schema_context`` predates type capture — so name-based
    time-column detection misses DATE columns like ``first_order`` and codegen declares
    them categorical (no grain truncation possible at query time). The warehouse itself
    is the authoritative type source. Returns {} when no database file is found or the
    introspection fails; cached per resolved project dir."""
    resolved = str(Path(project_dir).resolve())
    if resolved in _DUCKDB_TYPES_CACHE:
        return _DUCKDB_TYPES_CACHE[resolved]
    root = Path(resolved)
    db_path = root / "jaffle_shop.duckdb"
    if not db_path.exists():
        db_path = root / "dev.duckdb"
    if not db_path.exists():
        candidates = list(root.glob("*.duckdb"))
        db_path = candidates[0] if candidates else None
    types: dict[str, dict[str, str]] = {}
    if db_path is not None:
        try:
            import duckdb

            conn = duckdb.connect(str(db_path), read_only=True)
            try:
                rows = conn.execute(
                    "SELECT table_name, column_name, data_type "
                    "FROM information_schema.columns"
                ).fetchall()
            finally:
                conn.close()
            for table, column, data_type in rows:
                types.setdefault(table, {})[column] = data_type
        except Exception:
            types = {}
    _DUCKDB_TYPES_CACHE[resolved] = types
    return types
