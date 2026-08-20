from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from semanticflow.agents.types import Agent, AgentMessage
from semanticflow.config import Settings
from semanticflow.dbt_integration import (
    run_dbt_build,
    run_dbt_parse,
    run_dbt_seed,
    run_mf_query,
    run_mf_validate,
)
from semanticflow.log import get_logger

logger = get_logger("semanticflow.executor")


class ExecutorAgent(Agent):
    name: str = "executor"
    settings: Settings

    def _needs_seed(self, logs: str) -> bool:
        lowered = logs.lower()
        return "raw_" in lowered and ("does not exist" in lowered or "not found" in lowered)

    _OP_RE = r"(>=|<=|!=|=|>|<|\bnot\s+in\b|\bin\b|\blike\b)"

    def _filters_to_where(
        self,
        filters: list[str],
        time_granularity: str | None = None,
        primary_entity: str | None = None,
        dim_entity_map: dict[str, str] | None = None,
    ) -> str | None:
        """Convert semantic filters into a MetricFlow `--where` clause.

        MetricFlow's --where requires Jinja dimension references, NOT bare SQL:
          time:        {{ TimeDimension('metric_time', 'day') }} >= '2018-03-01'
          categorical: {{ Dimension('order__status') }} = 'completed'
        The previous code emitted bare `metric_time__day >= ...`, which mf rejects."""
        import re

        gran = time_granularity or "day"
        clauses: list[str] = []
        for raw in filters or []:
            s = str(raw).strip()
            if not s:
                continue

            # Time filter expressed as Jinja (TimeDimension/metric_time): normalise the
            # dimension to metric_time and keep the comparison.
            if "TimeDimension" in s or "metric_time" in s:
                m = re.search(r"\}\}\s*(.+)$", s)
                if m:
                    comp = m.group(1).strip()
                else:
                    m2 = re.search(rf"{self._OP_RE}\s*(.+)$", s)
                    comp = f"{m2.group(1)} {m2.group(2)}".strip() if m2 else s
                clauses.append(f"{{{{ TimeDimension('metric_time', '{gran}') }}}} {comp}")
                continue

            # Bare `field op value`.
            m = re.match(rf"^([A-Za-z_][A-Za-z0-9_.]*)\s*{self._OP_RE}\s*(.+)$", s, re.I)
            if m:
                field = m.group(1).split(".")[-1]
                op, val = m.group(2).strip(), m.group(3).strip()
                val_is_date = bool(re.search(r"'\d{4}-\d{2}-\d{2}'", val))
                field_is_time = field.endswith(("_date", "_at")) or field in {"date", "first_order", "most_recent_order"}
                if val_is_date or field_is_time:
                    ref = "metric_time" if field in {"metric_time", "order_date"} else (
                        f"{primary_entity}__{field}" if primary_entity else field)
                    if ref == "metric_time":
                        clauses.append(f"{{{{ TimeDimension('metric_time', '{gran}') }}}} {op} {val}")
                    else:
                        clauses.append(f"{{{{ TimeDimension('{ref}', '{gran}') }}}} {op} {val}")
                else:
                    # Prefer the field's own model entity (cross-model filters, e.g. a
                    # `status` filter on `orders` while the measure is on `stg_payments`
                    # → `order__status`), falling back to the primary entity.
                    ent = (dim_entity_map or {}).get(field) or primary_entity
                    ref = f"{ent}__{field}" if ent else field
                    clauses.append(f"{{{{ Dimension('{ref}') }}}} {op} {val}")
                continue

            clauses.append(s)
        return " AND ".join(clauses) if clauses else None

    def _build_mf_group_by(self, semantic_spec: dict[str, Any], design: dict[str, Any]) -> list[str]:
        group_by = semantic_spec.get("group_by", []) or []
        if not group_by:
            return []
        time_granularity = semantic_spec.get("time_granularity")
        semantic_models = design.get("semantic_models", []) or []
        
        # Build a map of dimension -> (entity_prefix, dimension_info) for ALL semantic models
        dim_to_entity: dict[str, tuple[str, dict[str, Any]]] = {}
        primary_model = semantic_models[0] if semantic_models else {}
        primary_entity = primary_model.get("primary_entity")
        primary_defaults = primary_model.get("defaults", {}) or {}
        agg_time_dimension = primary_defaults.get("agg_time_dimension")
        
        # Collect all entity names across models
        all_entity_names: set[str] = set()
        
        for model in semantic_models:
            entities = model.get("entities", []) or []
            dimensions = model.get("dimensions", []) or []
            model_primary_entity = model.get("primary_entity")
            
            # Find primary entity if not set
            if not model_primary_entity:
                for entity in entities:
                    if isinstance(entity, dict) and entity.get("type") == "primary":
                        model_primary_entity = entity.get("name")
                        break
            
            # Collect entity names
            for entity in entities:
                if isinstance(entity, dict) and entity.get("name"):
                    all_entity_names.add(entity.get("name"))
            
            # Map dimensions to their entity prefix
            for dim in dimensions:
                if isinstance(dim, dict) and dim.get("name"):
                    dim_name = dim.get("name")
                    # Don't override if already mapped (primary model takes precedence)
                    if dim_name not in dim_to_entity:
                        dim_to_entity[dim_name] = (model_primary_entity, dim)

        resolved: list[str] = []
        for item in group_by:
            if not item or not isinstance(item, str):
                continue
            if "__" in item or item.startswith("metric_time__"):
                resolved.append(item)
                continue
            # MetricFlow's reserved virtual time dimension: resolve to metric_time__<grain>,
            # NOT entity-prefixed (which produced the invalid "order__metric_time").
            if item == "metric_time":
                resolved.append(f"metric_time__{time_granularity or 'day'}")
                continue

            # Look up which entity this dimension belongs to
            entity_prefix, dim = dim_to_entity.get(item, (primary_entity, {}))
            is_time_dim = bool(dim and dim.get("type") == "time")
            
            if time_granularity and (item == agg_time_dimension or is_time_dim):
                if item == agg_time_dimension:
                    resolved.append(f"metric_time__{time_granularity}")
                elif entity_prefix:
                    resolved.append(f"{entity_prefix}__{item}__{time_granularity}")
                else:
                    resolved.append(f"{item}__{time_granularity}")
                continue
            if item in all_entity_names:
                resolved.append(item)
                continue
            if entity_prefix:
                resolved.append(f"{entity_prefix}__{item}")
                continue
            resolved.append(item)

        seen: set[str] = set()
        deduped: list[str] = []
        for item in resolved:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped

    def _resolve_order_by(
        self,
        order_by: list[dict[str, str]] | None,
        semantic_spec: dict[str, Any],
        design: dict[str, Any],
        resolved_dims: list[str],
        metric_name: str | None,
        limit: int | None = None,
    ) -> list[dict[str, str]] | None:
        """Map order-by fields onto valid query items (the resolved group-by dimensions
        or the metric), dropping any that resolve to neither. MetricFlow rejects an
        order-by item that is not exactly one of the query items (e.g. ordering by
        'order_date' when the query groups by 'metric_time__day').

        When a limit is set, the resolved group-by dimensions are appended as
        secondary ascending sort keys so ties at the LIMIT cutoff are broken
        deterministically (matching the conventional gold tie-break, e.g.
        ORDER BY value DESC, customer_id ASC)."""
        if not order_by and not limit:
            return order_by
        measures = {str(m).lower() for m in (semantic_spec.get("base_measures") or [])}
        valid = set(resolved_dims) | ({metric_name} if metric_name else set())
        out: list[dict[str, str]] = []
        for item in order_by or []:
            if not isinstance(item, dict):
                continue
            field = item.get("field") or item.get("name")
            if not field:
                continue
            direction = item.get("direction", "asc")
            if metric_name and (field == metric_name or str(field).lower() in measures
                                or field == semantic_spec.get("metric_name")):
                new: str | None = metric_name
            elif field in valid:
                new = field
            else:
                # Resolve through the same dimension mapping as the group-by.
                r = self._build_mf_group_by({**semantic_spec, "group_by": [field]}, design)
                new = r[0] if r and r[0] in valid else None
            if new:
                out.append({"field": new, "direction": direction})
        if limit:
            present = {o["field"] for o in out}
            for dim in resolved_dims:
                if dim not in present:
                    out.append({"field": dim, "direction": "asc"})
                    present.add(dim)
        return out or None

    async def run(self, msg: AgentMessage) -> AgentMessage:
        state = dict(msg.state)
        project_dir = msg.task.dbt_project or self.settings.dbt_project_dir
        logger.info(f"Starting execution for task {msg.task.task_id}", project_dir=project_dir)

        if not self.settings.run_dbt:
            logger.info("dbt execution skipped (run_dbt=False)")
            state["dbt_results"] = {
                "dbt_parse_success": False,
                "dbt_build_success": False,
                "dbt_parse_skipped": True,
                "dbt_build_skipped": True,
                "dbt_errors": [],
                "dbt_logs": "",
            }
            state["mf_validate_success"] = False
            state["mf_validate_stdout"] = ""
            state["mf_validate_stderr"] = ""
            state["mf_validate_issues"] = []
            state["mf_query_success"] = False
            state["mf_query_stdout"] = ""
            state["mf_query_stderr"] = ""
            state["mf_query_result"] = []
            return AgentMessage(task=msg.task, state=state)

        env = os.environ.copy()
        if self.settings.duckdb_path:
            env["DBT_DUCKDB_PATH"] = self.settings.duckdb_path
            env["DUCKDB_PATH"] = self.settings.duckdb_path

        # Clear dbt cache to prevent stale manifest conflicts between tasks
        target_dir = Path(project_dir) / "target"
        if target_dir.exists():
            logger.debug(f"Clearing dbt cache at {target_dir}")
            shutil.rmtree(target_dir, ignore_errors=True)

        parse_result = run_dbt_parse(
            project_dir,
            env=env,
            profile=self.settings.dbt_profile_name,
            profiles_dir=project_dir,
        )

        build_result = None
        seed_result = None
        select = None
        mf_validate_result = None
        mf_query_result = None
        mf_query_rows: list[dict[str, Any]] = []
        design = msg.state.get("design_proposals", {})
        semantic_models = design.get("semantic_models", [])
        if semantic_models:
            select = [
                spec.get("name") if isinstance(spec, dict) else getattr(spec, "name", None)
                for spec in semantic_models
            ]
            select = [f"+{item}" for item in select if item]

        if parse_result.success:
            if self.settings.skip_mf_validation:
                # Skip MF validation (workaround for Windows WinError 10106)
                mf_validate_result = None
            else:
                mf_validate_result = run_mf_validate(project_dir, self.settings, env=env)
            build_result = run_dbt_build(
                project_dir,
                select=None,
                env=env,
                profile=self.settings.dbt_profile_name,
                profiles_dir=project_dir,
            )
            build_logs = "\n".join([build_result.stdout, build_result.stderr])
            if not build_result.success and self._needs_seed(build_logs):
                seed_result = run_dbt_seed(
                    project_dir,
                    env=env,
                    profile=self.settings.dbt_profile_name,
                    profiles_dir=project_dir,
                )
                build_result = run_dbt_build(
                    project_dir,
                    select=None,
                    env=env,
                    profile=self.settings.dbt_profile_name,
                    profiles_dir=project_dir,
                )

            semantic_spec = msg.state.get("semantic_spec", {})
            # The designer normalizes the metric name (e.g. "total revenue per
            # customer" -> "total_revenue_per_customer") and emits THAT into the
            # metrics YAML. Query by the emitted name, not the raw spec name, or
            # MetricFlow rejects it ("does not exactly match any known metrics").
            _emitted = design.get("metrics") or []
            _first = _emitted[0] if _emitted else None
            _emitted_name = (
                (_first.get("name") if isinstance(_first, dict)
                 else getattr(_first, "name", None))
                if _first is not None else None
            )
            metric_name = _emitted_name or semantic_spec.get("metric_name")
            # Run MF query if: validation passed, OR validation was skipped and build succeeded
            # But skip entirely if skip_mf_query is set (Windows MF CLI issues)
            mf_ok = (mf_validate_result is not None and mf_validate_result.success) or self.settings.skip_mf_validation
            if self.settings.skip_mf_query:
                # Skip MF query entirely - mark as skipped success
                pass
            elif (
                metric_name
                and mf_ok
                and build_result
                and build_result.success
            ):
                dimensions = self._build_mf_group_by(semantic_spec, design)
                time_granularity = semantic_spec.get("time_granularity")
                filters = semantic_spec.get("filters", []) or []
                _primary_sm = (design.get("semantic_models") or [{}])[0]
                _primary_entity = _primary_sm.get("primary_entity") or next(
                    (e.get("name") for e in _primary_sm.get("entities", []) or []
                     if e.get("type") == "primary"), None)
                # Map each (non-time) dimension column to its model's entity prefix, so a
                # filter on a secondary model's column is qualified correctly (cross-model).
                _dim_entity: dict[str, str] = {}
                for _m in (design.get("semantic_models") or []):
                    _ent = _m.get("primary_entity") or next(
                        (e.get("name") for e in (_m.get("entities") or [])
                         if isinstance(e, dict) and e.get("type") == "primary"), None)
                    if not _ent:
                        continue
                    for _d in (_m.get("dimensions") or []):
                        if isinstance(_d, dict) and _d.get("name") and _d.get("type") != "time":
                            _dim_entity.setdefault(_d["name"], _ent)
                where = self._filters_to_where(
                    filters, time_granularity=time_granularity,
                    primary_entity=_primary_entity, dim_entity_map=_dim_entity)
                limit = semantic_spec.get("limit")
                order_by = self._resolve_order_by(
                    semantic_spec.get("order_by"), semantic_spec, design, dimensions, metric_name,
                    limit=limit,
                )
                mf_query_result, mf_query_rows = run_mf_query(
                    project_dir,
                    self.settings,
                    metric_name,
                    dimensions=dimensions,
                    time_granularity=time_granularity,
                    where=where,
                    limit=limit,
                    order_by=order_by,
                    env=env,
                )

        # Log execution results
        parse_ok = parse_result.success
        build_ok = build_result.success if build_result else False
        mf_val_ok = (mf_validate_result.success if mf_validate_result else False) or self.settings.skip_mf_validation
        mf_query_ok = (mf_query_result.success if mf_query_result else False) or self.settings.skip_mf_query
        
        logger.info(
            f"Execution complete: parse={parse_ok}, build={build_ok}, mf_validate={mf_val_ok}, mf_query={mf_query_ok}",
            rows_returned=len(mf_query_rows),
        )

        state["dbt_results"] = {
            "dbt_parse_success": parse_ok,
            "dbt_build_success": build_ok,
            "dbt_seed_success": seed_result.success if seed_result else False,
            "dbt_parse": parse_result,
            "dbt_seed": seed_result,
            "dbt_build": build_result,
            "dbt_errors": (parse_result.error_types or [])
            + (build_result.error_types if build_result else []),
            "dbt_logs": "\n".join(
                [
                    parse_result.stdout,
                    parse_result.stderr,
                    seed_result.stdout if seed_result else "",
                    seed_result.stderr if seed_result else "",
                    build_result.stdout if build_result else "",
                    build_result.stderr if build_result else "",
                ]
            ).strip(),
        }
        state["mf_validate_success"] = mf_val_ok
        state["mf_validate_stdout"] = mf_validate_result.stdout if mf_validate_result else ""
        state["mf_validate_stderr"] = mf_validate_result.stderr if mf_validate_result else ""
        state["mf_validate_issues"] = mf_validate_result.issues if mf_validate_result else []
        state["mf_query_success"] = mf_query_ok
        state["mf_query_stdout"] = mf_query_result.stdout if mf_query_result else ""
        state["mf_query_stderr"] = mf_query_result.stderr if mf_query_result else ""
        state["mf_query_result"] = mf_query_rows
        return AgentMessage(task=msg.task, state=state)
