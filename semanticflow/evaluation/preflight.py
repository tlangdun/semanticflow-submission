"""Pre-flight validation of the GENERATED design before the dbt/MetricFlow run.

``verify_spec`` checks the mapper's *spec* against the schema; nothing checked the
*design* the designer/codegen actually emit. Failure modes seen in runs: the designer
silently drops every measure whose expr fails validation and still emits a metric, or a
measure expr references a column that exists nowhere — both only discovered by a full
dbt parse/build. These checks are deterministic and free, and their violations are
surfaced into ``validation_issues`` so degraded designs are visible before execution.

Conservative by construction: only flags references that are demonstrably wrong (a bare
column name absent from EVERY model), never style. Synthetic time dimensions ('ds',
'metric_time') and SQL expressions are skipped rather than guessed at.
"""
from __future__ import annotations

import re
from typing import Any

_BARE_COLUMN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SYNTHETIC_NAMES = {"ds", "metric_time"}


def _all_columns(schema_context: dict[str, Any] | None) -> set[str]:
    cols: set[str] = set()
    for cs in ((schema_context or {}).get("columns") or {}).values():
        if isinstance(cs, (list, tuple)):
            cols.update(str(c) for c in cs)
    return cols


def preflight_design(
    design_proposals: dict[str, Any] | None,
    schema_context: dict[str, Any] | None,
) -> list[str]:
    """Return violation strings for a generated design; empty list = clean."""
    design = design_proposals or {}
    violations: list[str] = []
    cols = _all_columns(schema_context)
    known_models = set((schema_context or {}).get("models") or [])

    measure_names: set[str] = set()
    for sm in design.get("semantic_models") or []:
        if not isinstance(sm, dict):
            continue
        sm_name = sm.get("name", "?")

        model_ref = sm.get("model")
        if known_models and model_ref and model_ref not in known_models:
            violations.append(
                f"semantic model '{sm_name}' references unknown dbt model '{model_ref}'"
            )

        for measure in sm.get("measures") or []:
            if not isinstance(measure, dict):
                continue
            measure_names.add(str(measure.get("name")))
            expr = measure.get("expr")
            # Only a bare column reference is checkable; SQL expressions are skipped.
            if (
                cols
                and isinstance(expr, str)
                and _BARE_COLUMN.match(expr)
                and expr not in cols
            ):
                violations.append(
                    f"measure '{measure.get('name')}' on '{sm_name}' references "
                    f"non-existent column '{expr}'"
                )

        for dim in sm.get("dimensions") or []:
            if not isinstance(dim, dict):
                continue
            if dim.get("name") in _SYNTHETIC_NAMES:
                continue
            expr = dim.get("expr") or dim.get("name")
            if (
                cols
                and isinstance(expr, str)
                and _BARE_COLUMN.match(expr)
                and expr not in cols
            ):
                violations.append(
                    f"dimension '{dim.get('name')}' on '{sm_name}' references "
                    f"non-existent column '{expr}'"
                )

    metrics = design.get("metrics") or []
    if not metrics:
        violations.append(
            "no metric emitted (no valid measure survived designer validation) — the "
            "MetricFlow query cannot succeed"
        )
    for metric in metrics:
        if not isinstance(metric, dict):
            continue
        ref = ((metric.get("type_params") or {}).get("measure"))
        ref_name = ref.get("name") if isinstance(ref, dict) else ref
        if ref_name and ref_name not in measure_names:
            violations.append(
                f"metric '{metric.get('name')}' references measure '{ref_name}' that no "
                "semantic model defines"
            )
    return violations
