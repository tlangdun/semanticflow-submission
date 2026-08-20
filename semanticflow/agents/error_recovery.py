"""Self-healing error recovery agent.

Analyzes dbt/MetricFlow errors and suggests or applies automatic fixes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from semanticflow.log import get_logger

logger = get_logger("semanticflow.error_recovery")


class ErrorType(Enum):
    """Categories of recoverable errors."""
    MISSING_COLUMN = "missing_column"
    INVALID_DIMENSION_NAME = "invalid_dimension_name"
    MISSING_TIME_SPINE = "missing_time_spine"
    DUPLICATE_METRIC = "duplicate_metric"
    INVALID_AGGREGATION = "invalid_aggregation"
    MISSING_MEASURE = "missing_measure"
    SYNTAX_ERROR = "syntax_error"
    UNKNOWN = "unknown"


# Error types that apply_recovery() can ACTUALLY fix. Other types may advertise
# auto_fixable=True (e.g. MISSING_TIME_SPINE, which is handled by the designer/codegen
# path, not here), but treating them as auto-recoverable triggers a no-op retry. Gate on
# this set so can_auto_recover is honest.
_IMPLEMENTED_FIXES = {ErrorType.INVALID_DIMENSION_NAME, ErrorType.MISSING_COLUMN}


def _is_auto_recoverable(action: RecoveryAction) -> bool:
    return (
        action.auto_fixable
        and action.confidence > 0.7
        and action.error_type in _IMPLEMENTED_FIXES
    )


@dataclass
class RecoveryAction:
    """A suggested fix for an error."""
    error_type: ErrorType
    description: str
    auto_fixable: bool = False
    fix_details: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0  # 0-1 confidence in the fix


@dataclass
class ErrorAnalysis:
    """Analysis of errors with suggested recoveries."""
    errors: list[str]
    error_types: list[ErrorType]
    recovery_actions: list[RecoveryAction]
    can_auto_recover: bool = False
    summary: str = ""


# Error patterns and their fixes
ERROR_PATTERNS: list[tuple[str, ErrorType, dict[str, Any]]] = [
    # Missing column errors
    (
        r"column[s]? ['\"]?(\w+)['\"]? (?:does not exist|not found|unknown)",
        ErrorType.MISSING_COLUMN,
        {"extract_column": 1},
    ),
    (
        r"Unknown column[s]?: ['\"]?(\w+)['\"]?",
        ErrorType.MISSING_COLUMN,
        {"extract_column": 1},
    ),
    # Invalid dimension name format
    (
        r"Name is in an incorrect format: ['\"]?(\w+)['\"]?.*should be.*<primary entity name>__<dimension_name>",
        ErrorType.INVALID_DIMENSION_NAME,
        {"extract_dimension": 1},
    ),
    # Missing time spine
    (
        r"(?:time spine|metricflow_time_spine).*(?:not found|missing|does not exist)",
        ErrorType.MISSING_TIME_SPINE,
        {},
    ),
    # Duplicate metric
    (
        r"(?:duplicate|already exists).*metric.*['\"]?(\w+)['\"]?",
        ErrorType.DUPLICATE_METRIC,
        {"extract_metric": 1},
    ),
    (
        r"Can't use label.*['\"]?(\w+)['\"]?.*already used",
        ErrorType.DUPLICATE_METRIC,
        {"extract_metric": 1},
    ),
    # Invalid aggregation
    (
        r"(?:invalid|unsupported) aggregation.*['\"]?(\w+)['\"]?",
        ErrorType.INVALID_AGGREGATION,
        {"extract_agg": 1},
    ),
    # Missing measure
    (
        r"measure ['\"]?(\w+)['\"]? (?:not found|does not exist|unknown)",
        ErrorType.MISSING_MEASURE,
        {"extract_measure": 1},
    ),
]


def analyze_errors(
    stderr: str,
    stdout: str = "",
    semantic_spec: dict[str, Any] | None = None,
    schema_context: dict[str, Any] | None = None,
) -> ErrorAnalysis:
    """Analyze error output and suggest recovery actions.
    
    Args:
        stderr: Standard error output from dbt/mf
        stdout: Standard output (sometimes contains errors)
        semantic_spec: The semantic specification that caused the error
        schema_context: Available schema information
    
    Returns:
        ErrorAnalysis with identified errors and suggested fixes
    """
    combined = f"{stdout}\n{stderr}".lower()
    errors: list[str] = []
    error_types: list[ErrorType] = []
    recovery_actions: list[RecoveryAction] = []
    
    for pattern, error_type, meta in ERROR_PATTERNS:
        matches = re.finditer(pattern, combined, re.IGNORECASE)
        for match in matches:
            error_msg = match.group(0)
            if error_msg not in errors:
                errors.append(error_msg)
                error_types.append(error_type)
                
                # Generate recovery action
                action = _generate_recovery_action(
                    error_type, match, meta, semantic_spec, schema_context
                )
                if action:
                    recovery_actions.append(action)
    
    # Check for unknown errors if nothing matched
    if not errors and ("error" in combined or "failed" in combined):
        errors.append("Unknown error detected")
        error_types.append(ErrorType.UNKNOWN)
    
    can_auto_recover = any(_is_auto_recoverable(a) for a in recovery_actions)
    
    summary = _generate_summary(errors, recovery_actions)
    
    return ErrorAnalysis(
        errors=errors,
        error_types=error_types,
        recovery_actions=recovery_actions,
        can_auto_recover=can_auto_recover,
        summary=summary,
    )


def _generate_recovery_action(
    error_type: ErrorType,
    match: re.Match,
    meta: dict[str, Any],
    semantic_spec: dict[str, Any] | None,
    schema_context: dict[str, Any] | None,
) -> RecoveryAction | None:
    """Generate a recovery action for a specific error."""
    
    if error_type == ErrorType.MISSING_COLUMN:
        column = match.group(meta.get("extract_column", 1)) if meta.get("extract_column") else None
        if column and schema_context:
            # Try to find a similar column
            all_columns = set()
            for cols in schema_context.get("columns", {}).values():
                all_columns.update(cols)
            
            suggestion = _find_similar(column, all_columns)
            if suggestion:
                return RecoveryAction(
                    error_type=error_type,
                    description=f"Column '{column}' not found. Did you mean '{suggestion}'?",
                    auto_fixable=True,
                    fix_details={"replace": column, "with": suggestion},
                    confidence=0.8,
                )
        return RecoveryAction(
            error_type=error_type,
            description=f"Column '{column}' not found in schema. Check column names.",
            auto_fixable=False,
            confidence=0.5,
        )
    
    elif error_type == ErrorType.INVALID_DIMENSION_NAME:
        dim_name = match.group(meta.get("extract_dimension", 1)) if meta.get("extract_dimension") else None
        return RecoveryAction(
            error_type=error_type,
            description=f"Dimension '{dim_name}' must use 'metric_time' or 'entity__dimension' format.",
            auto_fixable=True,
            fix_details={"dimension": dim_name, "replacement": "metric_time"},
            confidence=0.9,
        )
    
    elif error_type == ErrorType.MISSING_TIME_SPINE:
        return RecoveryAction(
            error_type=error_type,
            description="MetricFlow time spine table is missing. Will create it.",
            auto_fixable=True,
            fix_details={"create_time_spine": True},
            confidence=0.95,
        )
    
    elif error_type == ErrorType.DUPLICATE_METRIC:
        metric = match.group(meta.get("extract_metric", 1)) if meta.get("extract_metric") else None
        return RecoveryAction(
            error_type=error_type,
            description=f"Metric '{metric}' already exists. Will update existing metric.",
            auto_fixable=True,
            fix_details={"metric": metric, "action": "update"},
            confidence=0.85,
        )
    
    elif error_type == ErrorType.MISSING_MEASURE:
        measure = match.group(meta.get("extract_measure", 1)) if meta.get("extract_measure") else None
        return RecoveryAction(
            error_type=error_type,
            description=f"Measure '{measure}' not found. Check semantic model definition.",
            auto_fixable=False,
            fix_details={"measure": measure},
            confidence=0.6,
        )
    
    return None


def _find_similar(target: str, candidates: set[str], threshold: float = 0.6) -> str | None:
    """Find the most similar string from candidates using simple matching."""
    target_lower = target.lower()
    best_match = None
    best_score = 0.0
    
    for candidate in candidates:
        cand_lower = candidate.lower()
        
        # Exact match (case-insensitive)
        if target_lower == cand_lower:
            return candidate
        
        # Substring match
        if target_lower in cand_lower or cand_lower in target_lower:
            score = min(len(target_lower), len(cand_lower)) / max(len(target_lower), len(cand_lower))
            if score > best_score:
                best_score = score
                best_match = candidate
        
        # Common prefix/suffix
        common_prefix = 0
        for i, (a, b) in enumerate(zip(target_lower, cand_lower)):
            if a == b:
                common_prefix = i + 1
            else:
                break
        
        if common_prefix > 0:
            score = common_prefix / max(len(target_lower), len(cand_lower))
            if score > best_score:
                best_score = score
                best_match = candidate
    
    return best_match if best_score >= threshold else None


def _generate_summary(errors: list[str], actions: list[RecoveryAction]) -> str:
    """Generate a human-readable summary of errors and fixes."""
    if not errors:
        return "No errors detected."
    
    auto_fixable = [a for a in actions if _is_auto_recoverable(a)]
    manual_required = [a for a in actions if not _is_auto_recoverable(a)]
    
    parts = [f"Found {len(errors)} error(s)."]
    
    if auto_fixable:
        parts.append(f"{len(auto_fixable)} can be auto-fixed.")
    if manual_required:
        parts.append(f"{len(manual_required)} require manual review.")
    
    return " ".join(parts)


def apply_recovery(
    analysis: ErrorAnalysis,
    semantic_spec: dict[str, Any],
    design_proposals: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Apply auto-recoverable fixes to the semantic spec and design.
    
    Args:
        analysis: Error analysis with recovery actions
        semantic_spec: Current semantic specification
        design_proposals: Current design proposals
    
    Returns:
        Tuple of (updated_spec, updated_design, applied_fixes)
    """
    updated_spec = dict(semantic_spec)
    updated_design = dict(design_proposals)
    applied_fixes: list[str] = []
    
    for action in analysis.recovery_actions:
        if not _is_auto_recoverable(action):
            continue

        if action.error_type == ErrorType.INVALID_DIMENSION_NAME:
            # Fix dimension names in filters
            if "filters" in updated_spec:
                old_dim = action.fix_details.get("dimension", "")
                new_dim = action.fix_details.get("replacement", "metric_time")
                updated_spec["filters"] = [
                    f.replace(old_dim, new_dim) for f in updated_spec.get("filters", [])
                ]
                applied_fixes.append(f"Replaced dimension '{old_dim}' with '{new_dim}'")
        
        elif action.error_type == ErrorType.MISSING_COLUMN:
            # Replace column references
            old_col = action.fix_details.get("replace", "")
            new_col = action.fix_details.get("with", "")
            if old_col and new_col:
                # Update group_by
                if "group_by" in updated_spec:
                    updated_spec["group_by"] = [
                        new_col if g == old_col else g for g in updated_spec["group_by"]
                    ]
                # Update base_measures
                if "base_measures" in updated_spec:
                    updated_spec["base_measures"] = [
                        new_col if m == old_col else m for m in updated_spec["base_measures"]
                    ]
                applied_fixes.append(f"Replaced column '{old_col}' with '{new_col}'")
    
    logger.info(f"Applied {len(applied_fixes)} auto-fixes", fixes=applied_fixes)
    return updated_spec, updated_design, applied_fixes
