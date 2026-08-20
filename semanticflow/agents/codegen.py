from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from semanticflow.agents.types import Agent, AgentMessage
from semanticflow.dbt_integration import (
    MetricSpec,
    SemanticModelSpec,
    metric_to_dict,
    semantic_model_to_dict,
)


def _is_valid_name(name: str) -> bool:
    """Check if a name is valid for MetricFlow/dbt.
    
    Valid names must:
    - Only contain lowercase letters, numbers, underscores
    - Start with a lowercase letter
    - Not end with underscore
    - Not contain double underscores
    - Be at least 2 characters
    """
    if not name or len(name) < 2:
        return False
    if not re.match(r'^[a-z][a-z0-9_]*$', name):
        return False
    if name.endswith('_'):
        return False
    if '__' in name:
        return False
    return True


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data if isinstance(data, dict) else {}


def _dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _merge_named_list(existing: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Filter out existing items with invalid names (e.g., SQL syntax like COUNT(x) or table prefixes like orders.x)
    valid_existing = [item for item in existing if "name" in item and _is_valid_name(item.get("name", ""))]
    merged: dict[str, dict[str, Any]] = {item.get("name"): item for item in valid_existing}
    for item in incoming:
        name = item.get("name")
        if not _is_valid_name(name):
            continue  # Skip incoming items with invalid names too
        if name in merged:
            merged[name].update(item)
        else:
            merged[name] = item
    return list(merged.values())


def _merge_semantic_model(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing)
    for key in ["entities", "measures", "dimensions"]:
        if key in incoming:
            merged[key] = _merge_named_list(existing.get(key, []) or [], incoming.get(key, []) or [])
    for key, value in incoming.items():
        if key in {"entities", "measures", "dimensions"}:
            continue
        if key not in merged or merged[key] in (None, ""):
            merged[key] = value
    entities = merged.get("entities") or []
    if any(entity.get("type") == "primary" for entity in entities if isinstance(entity, dict)):
        merged.pop("primary_entity", None)
    
    # Sanitize defaults.agg_time_dimension - use incoming value if existing is invalid
    defaults = merged.get("defaults")
    if isinstance(defaults, dict):
        agg_dim = defaults.get("agg_time_dimension")
        if agg_dim and not _is_valid_name(agg_dim):
            # Try to use incoming value, or find first valid time dimension
            incoming_defaults = incoming.get("defaults", {})
            if isinstance(incoming_defaults, dict) and _is_valid_name(incoming_defaults.get("agg_time_dimension", "")):
                defaults["agg_time_dimension"] = incoming_defaults["agg_time_dimension"]
            else:
                # Find first valid time dimension from merged dimensions
                for dim in merged.get("dimensions", []):
                    if isinstance(dim, dict) and dim.get("type") == "time" and _is_valid_name(dim.get("name", "")):
                        defaults["agg_time_dimension"] = dim["name"]
                        break
    
    return merged


def merge_yaml_safely(path: Path, section: str, spec_dict: dict[str, Any]) -> tuple[str, str, str | None]:
    backup_path = None
    if path.exists():
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        backup_path = str(path.with_suffix(f"{path.suffix}.bak.{timestamp}"))
        content = path.read_text(encoding="utf-8")
        Path(backup_path).write_text(content, encoding="utf-8")

    payload = _load_yaml(path)
    payload.setdefault("version", 2)
    existing_items = payload.get(section, []) or []

    # For metrics, also filter out items with duplicate labels/descriptions
    new_label = spec_dict.get("label") or spec_dict.get("description")
    
    updated_items = []
    merged = False
    for item in existing_items:
        if not isinstance(item, dict):
            updated_items.append(item)
            continue
        
        # Skip items with same name (will be replaced)
        if item.get("name") == spec_dict.get("name"):
            if section == "semantic_models":
                updated_items.append(_merge_semantic_model(item, spec_dict))
            else:
                updated_items.append(spec_dict)
            merged = True
            continue
        
        # For metrics, skip items with same label to avoid dbt error
        if section == "metrics" and new_label:
            item_label = item.get("label") or item.get("description")
            if item_label == new_label:
                continue  # Skip - would cause duplicate label error
        
        updated_items.append(item)

    if not merged:
        updated_items.append(spec_dict)

    payload[section] = updated_items
    _dump_yaml(path, payload)
    return ("merged" if merged else "created"), str(path), backup_path


def write_semantic_model_file(spec: SemanticModelSpec, project_dir: str) -> tuple[str, str, str | None]:
    """Write semantic model to consolidated semantic_models.yml file (dbt best practice)."""
    project_path = Path(project_dir)
    target_path = project_path / "models" / "semantic" / "semantic_models.yml"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    spec_dict = semantic_model_to_dict(spec)
    return merge_yaml_safely(target_path, "semantic_models", spec_dict)


def write_metric_file(spec: MetricSpec, project_dir: str) -> tuple[str, str, str | None]:
    """Write metric to consolidated metrics.yml file (dbt best practice)."""
    project_path = Path(project_dir)
    target_path = project_path / "models" / "semantic" / "metrics.yml"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    spec_dict = metric_to_dict(spec)
    return merge_yaml_safely(target_path, "metrics", spec_dict)


def reset_generated_semantics(project_dir: str) -> None:
    """Clear the tool-generated semantic layer so each task is evaluated in isolation.

    The consolidated ``semantic_models.yml`` / ``metrics.yml`` are entirely tool-authored
    (the jaffle_shop project's real models are the SQL models, not these). Without this
    reset they ACCUMULATE across tasks: a prior task's broken semantic model fails the
    current task's ``dbt build`` / ``mf validate``, so per-task results are not
    independent — fatal for an execution-equivalence benchmark.
    """
    semantic_dir = Path(project_dir) / "models" / "semantic"
    for filename, section in (("semantic_models.yml", "semantic_models"), ("metrics.yml", "metrics")):
        path = semantic_dir / filename
        if path.exists():
            _dump_yaml(path, {"version": 2, section: []})


class CodegenAgent(Agent):
    name: str = "codegen"

    async def run(self, msg: AgentMessage) -> AgentMessage:
        design = msg.state.get("design_proposals", {})
        raw_semantic_models = design.get("semantic_models", [])
        raw_metrics = design.get("metrics", [])
        sql_models = design.get("sql_models", []) or []
        time_spine_info = design.get("time_spine")

        semantic_models: list[SemanticModelSpec] = []
        for item in raw_semantic_models:
            if isinstance(item, SemanticModelSpec):
                semantic_models.append(item)
            else:
                semantic_models.append(SemanticModelSpec.model_validate(item))

        metrics: list[MetricSpec] = []
        for item in raw_metrics:
            if isinstance(item, MetricSpec):
                metrics.append(item)
            else:
                metrics.append(MetricSpec.model_validate(item))

        generated_files: list[str] = []
        merge_operations: list[dict[str, str]] = []
        project_dir = msg.task.dbt_project or "third_party/jaffle_shop_duckdb"
        # Start from a clean generated semantic layer so prior tasks' models cannot
        # contaminate this task's dbt build / MetricFlow validation (per-task isolation).
        reset_generated_semantics(project_dir)

        for spec in semantic_models:
            if not spec.description:
                spec = spec.model_copy(update={"description": f"Generated for task {msg.task.task_id}."})
            operation, path, backup = write_semantic_model_file(spec, project_dir)
            generated_files.append(path)
            merge_entry = {"path": path, "operation": operation}
            if backup:
                merge_entry["backup_path"] = backup
            merge_operations.append(merge_entry)

        for spec in metrics:
            if not spec.description:
                spec = spec.model_copy(update={"description": f"Generated for task {msg.task.task_id}."})
            operation, path, backup = write_metric_file(spec, project_dir)
            generated_files.append(path)
            merge_entry = {"path": path, "operation": operation}
            if backup:
                merge_entry["backup_path"] = backup
            merge_operations.append(merge_entry)

        # NOTE: auto-generated dbt schema tests (not_null/unique/relationships) are
        # intentionally NOT written. For an execution-equivalence eval they add no signal
        # and actively cause false failures — e.g. a generated entry collided with the
        # project's own models/staging/schema.yml (duplicate stg_payments) and broke
        # `dbt parse` for every task, and not_null tests on synthetic columns fail builds
        # for otherwise-correct queries. MetricFlow only needs semantic_models.yml +
        # metrics.yml, written above.

        created_spines: set[str] = set()
        for sql_model in sql_models:
            path_value = sql_model.get("path")
            sql = sql_model.get("sql")
            if not path_value or not sql:
                continue
            target_path = Path(project_dir) / "models" / path_value
            if target_path.exists():
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(sql, encoding="utf-8")
            generated_files.append(str(target_path))
            merge_operations.append({"path": str(target_path), "operation": "created"})
            if sql_model.get("name") == "metricflow_time_spine":
                created_spines.add(str(target_path))

        if time_spine_info:
            spine_path = Path(project_dir) / time_spine_info["path"]
            if str(spine_path) in created_spines:
                operation = "created"
            elif spine_path.exists():
                operation = "exists"
            else:
                operation = "missing"
            state = dict(msg.state)
            state["time_spine"] = {
                **time_spine_info,
                "operation": operation,
            }
        else:
            state = dict(msg.state)

        state["generated_files"] = generated_files
        state["merge_operations"] = merge_operations

        # Free deterministic pre-flight on the design that was just written: catches
        # metric-with-no-measure and unknown-column references before the dbt/mf run,
        # so a degraded design is diagnosed here instead of via an opaque parse error.
        from semanticflow.evaluation.preflight import preflight_design

        state["preflight_violations"] = preflight_design(design, msg.task.schema_context)
        return AgentMessage(task=msg.task, state=state)
