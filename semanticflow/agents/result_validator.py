"""Query result validator for sanity checking MetricFlow query outputs.

Performs automated checks on query results to catch potential issues
before presenting to users.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from semanticflow.log import get_logger

logger = get_logger("semanticflow.result_validator")


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ValidationIssue:
    """A single validation issue found in query results."""
    severity: ValidationSeverity
    check_name: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Complete validation result for query output."""
    is_valid: bool
    issues: list[ValidationIssue]
    row_count: int
    column_count: int
    summary: str
    suggestions: list[str] = field(default_factory=list)


def validate_query_result(
    rows: list[dict[str, Any]],
    semantic_spec: dict[str, Any] | None = None,
    expected_columns: list[str] | None = None,
    time_dimension: str | None = None,
) -> ValidationResult:
    """Validate query results for common issues.
    
    Args:
        rows: Query result rows
        semantic_spec: The semantic specification used
        expected_columns: Expected column names in output
        time_dimension: Time dimension column name for time-series checks
    
    Returns:
        ValidationResult with any issues found
    """
    issues: list[ValidationIssue] = []
    suggestions: list[str] = []
    
    row_count = len(rows)
    column_count = len(rows[0]) if rows else 0
    
    # Check 1: Empty results
    if row_count == 0:
        issues.append(ValidationIssue(
            severity=ValidationSeverity.WARNING,
            check_name="empty_results",
            message="Query returned no rows",
            details={"row_count": 0},
        ))
        suggestions.append("Check if filters are too restrictive")
        suggestions.append("Verify the time range includes data")
    
    # Check 2: Very few results (might indicate filter issues)
    elif row_count < 5:
        issues.append(ValidationIssue(
            severity=ValidationSeverity.INFO,
            check_name="few_results",
            message=f"Query returned only {row_count} rows",
            details={"row_count": row_count},
        ))
    
    # Check 3: Suspiciously many results
    elif row_count > 10000:
        issues.append(ValidationIssue(
            severity=ValidationSeverity.WARNING,
            check_name="many_results",
            message=f"Query returned {row_count} rows - consider adding filters",
            details={"row_count": row_count},
        ))
        suggestions.append("Add time filters to limit result set")
        suggestions.append("Consider adding LIMIT clause")
    
    if rows:
        # Check 4: Null values in key columns
        null_analysis = _check_null_values(rows, semantic_spec)
        issues.extend(null_analysis["issues"])
        suggestions.extend(null_analysis["suggestions"])
        
        # Check 5: Numeric outliers
        outlier_analysis = _check_numeric_outliers(rows)
        issues.extend(outlier_analysis["issues"])
        
        # Check 6: Time series gaps (if time dimension present)
        if time_dimension:
            time_analysis = _check_time_series(rows, time_dimension)
            issues.extend(time_analysis["issues"])
            suggestions.extend(time_analysis["suggestions"])
        
        # Check 7: Duplicate rows
        dup_analysis = _check_duplicates(rows)
        issues.extend(dup_analysis["issues"])
        
        # Check 8: Expected columns present
        if expected_columns:
            missing = _check_expected_columns(rows, expected_columns)
            issues.extend(missing["issues"])
    
    # Determine overall validity
    has_errors = any(i.severity == ValidationSeverity.ERROR for i in issues)
    has_warnings = any(i.severity == ValidationSeverity.WARNING for i in issues)
    
    is_valid = not has_errors
    
    summary = _generate_summary(row_count, column_count, issues)
    
    logger.info(
        f"Validation complete: {len(issues)} issues, valid={is_valid}",
        row_count=row_count,
        warnings=has_warnings,
    )
    
    return ValidationResult(
        is_valid=is_valid,
        issues=issues,
        row_count=row_count,
        column_count=column_count,
        summary=summary,
        suggestions=suggestions,
    )


def _check_null_values(
    rows: list[dict[str, Any]],
    semantic_spec: dict[str, Any] | None,
) -> dict[str, Any]:
    """Check for null values in key columns."""
    issues: list[ValidationIssue] = []
    suggestions: list[str] = []
    
    if not rows:
        return {"issues": issues, "suggestions": suggestions}
    
    columns = list(rows[0].keys())
    
    # Key columns that shouldn't have nulls (metrics, dimensions)
    key_columns = set()
    if semantic_spec:
        key_columns.update(semantic_spec.get("base_measures", []) or [])
        key_columns.update(semantic_spec.get("group_by", []) or [])
    # Exact (case-insensitive) membership — substring matching let a short/empty key
    # like "id" or "" flag nearly every column as a key column.
    key_columns_lower = {str(k).strip().lower() for k in key_columns if str(k).strip()}

    for col in columns:
        null_count = sum(1 for row in rows if row.get(col) is None)
        null_pct = (null_count / len(rows)) * 100

        if null_count > 0:
            # Higher severity for key columns
            is_key = col.lower() in key_columns_lower
            severity = ValidationSeverity.WARNING if is_key else ValidationSeverity.INFO
            
            if null_pct > 50:
                severity = ValidationSeverity.WARNING
            
            if null_count > 0:
                issues.append(ValidationIssue(
                    severity=severity,
                    check_name="null_values",
                    message=f"Column '{col}' has {null_count} null values ({null_pct:.1f}%)",
                    details={"column": col, "null_count": null_count, "null_pct": null_pct},
                ))
                
                if null_pct > 20:
                    suggestions.append(f"Check data quality for column '{col}'")
    
    return {"issues": issues, "suggestions": suggestions}


def _check_numeric_outliers(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Check for extreme outliers in numeric columns."""
    issues: list[ValidationIssue] = []
    
    if not rows:
        return {"issues": issues}
    
    columns = list(rows[0].keys())
    
    for col in columns:
        values = [row.get(col) for row in rows if row.get(col) is not None]
        
        # Filter to numeric values
        numeric_values = []
        for v in values:
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                numeric_values.append(v)
        
        if len(numeric_values) < 3:
            continue
        
        # Simple outlier detection using IQR
        sorted_vals = sorted(numeric_values)
        n = len(sorted_vals)
        q1 = sorted_vals[n // 4]
        q3 = sorted_vals[3 * n // 4]
        iqr = q3 - q1
        
        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr
        
        outliers = [v for v in numeric_values if v < lower_bound or v > upper_bound]
        
        if outliers and len(outliers) < len(numeric_values) * 0.1:  # Less than 10% outliers
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                check_name="numeric_outliers",
                message=f"Column '{col}' has {len(outliers)} potential outliers",
                details={
                    "column": col,
                    "outlier_count": len(outliers),
                    "min_outlier": min(outliers),
                    "max_outlier": max(outliers),
                    "bounds": [lower_bound, upper_bound],
                },
            ))
        
        # Check for negative values in typically positive metrics
        if any(kw in col.lower() for kw in ["count", "amount", "revenue", "total", "sum"]):
            negative_count = sum(1 for v in numeric_values if v < 0)
            if negative_count > 0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    check_name="negative_values",
                    message=f"Column '{col}' has {negative_count} negative values",
                    details={"column": col, "negative_count": negative_count},
                ))
    
    return {"issues": issues}


def _check_time_series(
    rows: list[dict[str, Any]],
    time_dimension: str,
) -> dict[str, Any]:
    """Check time series data for gaps or issues."""
    issues: list[ValidationIssue] = []
    suggestions: list[str] = []
    
    # Find the time column (might have granularity suffix)
    time_col = None
    for col in rows[0].keys():
        if time_dimension in col.lower() or "metric_time" in col.lower():
            time_col = col
            break
    
    if not time_col:
        return {"issues": issues, "suggestions": suggestions}
    
    # Extract and sort time values
    time_values = []
    for row in rows:
        val = row.get(time_col)
        if val is not None:
            time_values.append(str(val))
    
    if len(time_values) < 2:
        return {"issues": issues, "suggestions": suggestions}
    
    time_values.sort()
    
    # Check for duplicate time values (might indicate aggregation issues)
    unique_times = set(time_values)
    if len(unique_times) < len(time_values):
        dup_count = len(time_values) - len(unique_times)
        issues.append(ValidationIssue(
            severity=ValidationSeverity.INFO,
            check_name="duplicate_times",
            message=f"Time dimension has {dup_count} duplicate values",
            details={"column": time_col, "duplicate_count": dup_count},
        ))
    
    return {"issues": issues, "suggestions": suggestions}


def _check_duplicates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Check for exact duplicate rows."""
    issues: list[ValidationIssue] = []
    
    if not rows:
        return {"issues": issues}
    
    # Convert rows to tuples for hashing
    seen = set()
    duplicates = 0
    
    for row in rows:
        row_tuple = tuple(sorted((k, str(v)) for k, v in row.items()))
        if row_tuple in seen:
            duplicates += 1
        else:
            seen.add(row_tuple)
    
    if duplicates > 0:
        issues.append(ValidationIssue(
            severity=ValidationSeverity.WARNING,
            check_name="duplicate_rows",
            message=f"Found {duplicates} exact duplicate rows",
            details={"duplicate_count": duplicates},
        ))
    
    return {"issues": issues}


def _check_expected_columns(
    rows: list[dict[str, Any]],
    expected: list[str],
) -> dict[str, Any]:
    """Check that expected columns are present in results."""
    issues: list[ValidationIssue] = []
    
    if not rows:
        return {"issues": issues}
    
    actual_columns = set(rows[0].keys())
    
    for col in expected:
        # Check for exact match or partial match (with granularity suffix)
        found = col in actual_columns or any(col in c for c in actual_columns)
        if not found:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                check_name="missing_column",
                message=f"Expected column '{col}' not found in results",
                details={"expected": col, "actual": list(actual_columns)},
            ))
    
    return {"issues": issues}


def _generate_summary(
    row_count: int,
    column_count: int,
    issues: list[ValidationIssue],
) -> str:
    """Generate a human-readable summary."""
    error_count = sum(1 for i in issues if i.severity == ValidationSeverity.ERROR)
    warning_count = sum(1 for i in issues if i.severity == ValidationSeverity.WARNING)
    info_count = sum(1 for i in issues if i.severity == ValidationSeverity.INFO)
    
    parts = [f"{row_count} rows, {column_count} columns."]
    
    if error_count:
        parts.append(f"{error_count} error(s).")
    if warning_count:
        parts.append(f"{warning_count} warning(s).")
    if info_count:
        parts.append(f"{info_count} info note(s).")
    
    if not issues:
        parts.append("All checks passed.")
    
    return " ".join(parts)
