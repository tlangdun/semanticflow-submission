"""Utilities for translating abstract filter concepts to valid MetricFlow syntax.

MetricFlow requires filters to use Jinja syntax like:
    {{ TimeDimension('metric_time', 'day') }} >= current_date - interval '30 days'

IMPORTANT: MetricFlow dimension names must be either:
1. The special 'metric_time' dimension (preferred for time filters)
2. In entity__dimension format (e.g., 'order__order_date')

This module translates abstract concepts like 'last_30_days' into valid syntax.
"""

import re
from typing import Any

# Pattern to match abstract time filters like 'last_30_days', 'last_7_days_inclusive_of_today'
ABSTRACT_TIME_PATTERN = re.compile(
    r"^last_(\d+)_(days?|weeks?|months?|quarters?|years?)(?:_inclusive_of_today)?$",
    re.IGNORECASE,
)

# Pattern for simple relative time references
SIMPLE_TIME_PATTERN = re.compile(
    r"^last_(month|quarter|year|week)$",
    re.IGNORECASE,
)

# Pattern to match TimeDimension calls with bare column names (not entity__dimension format)
# e.g., TimeDimension('order_date', 'day') should become TimeDimension('metric_time', 'day')
TIME_DIMENSION_PATTERN = re.compile(
    r"\{\{\s*TimeDimension\s*\(\s*['\"]([^'\"]+)['\"]\s*,\s*['\"]([^'\"]+)['\"]\s*\)\s*\}\}",
    re.IGNORECASE,
)


# Operator aliases for structured (dict-shaped) filters. Frontier-tier mappers and the
# execution-repair LLM return filters as {"field", "operator", "value"/"values"} objects
# rather than strings; cheap models emit strings — both must land on the same canonical
# string form or (a) the designer crashes on .strip() and (b) the cross-model
# disagreement signal sees phantom disagreement between equivalent filters.
_FILTER_OPS: dict[str, str] = {
    "equals": "=", "equal": "=", "eq": "=", "==": "=", "=": "=",
    "not_equals": "!=", "not_equal": "!=", "ne": "!=", "!=": "!=", "<>": "!=",
    "gt": ">", "greater_than": ">", ">": ">",
    "gte": ">=", "greater_than_or_equal": ">=", ">=": ">=",
    "lt": "<", "less_than": "<", "<": "<",
    "lte": "<=", "less_than_or_equal": "<=", "<=": "<=",
    "in": "in", "not_in": "not in", "not in": "not in",
    "like": "like", "between": "between",
}


def _quote_filter_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value).strip()
    if text.startswith("'") and text.endswith("'"):
        return text
    return f"'{text}'"


def coerce_filter_to_strings(item: Any) -> list[str]:
    """Render ONE filter (string or structured dict) into canonical string filter(s).

    A `between` dict expands into two range clauses (the form the designer's sanitizer
    and the executor's where-builder already understand); `in`/`not in` renders a value
    list. Returns [] for an un-renderable item (no field) rather than emitting a string
    that compiles into broken SQL."""
    if item is None:
        return []
    if isinstance(item, dict):
        field = item.get("field") or item.get("column") or item.get("dimension")
        if not field:
            return []
        op_raw = str(item.get("operator") or item.get("op") or "=").strip().lower()
        op = _FILTER_OPS.get(op_raw, op_raw)
        values = item.get("values")
        if values is None:
            value = item.get("value")
            values = value if isinstance(value, list) else ([] if value is None else [value])
        elif not isinstance(values, (list, tuple)):
            values = [values]
        if op == "between" and len(values) >= 2:
            return [
                f"{field} >= {_quote_filter_value(values[0])}",
                f"{field} <= {_quote_filter_value(values[1])}",
            ]
        if op in ("in", "not in") and values:
            rendered = ", ".join(_quote_filter_value(v) for v in values)
            return [f"{field} {op} ({rendered})"]
        if values:
            return [f"{field} {op} {_quote_filter_value(values[0])}"]
        return []
    text = str(item).strip()
    return [text] if text else []


def coerce_filters_to_strings(filters: Any) -> list[str]:
    """Coerce a spec's ``filters`` value (string, dict, or mixed list) to list[str]."""
    if filters is None:
        return []
    items = filters if isinstance(filters, list) else [filters]
    out: list[str] = []
    for item in items:
        out.extend(coerce_filter_to_strings(item))
    return out


def _normalize_time_dimension_names(filter_value: str) -> str:
    """Normalize TimeDimension calls to use metric_time instead of bare column names.
    
    MetricFlow requires dimension names in one of two formats:
    1. 'metric_time' - the special built-in time dimension
    2. 'entity__dimension' - e.g., 'order__order_date'
    
    This function converts bare column names like 'order_date' to 'metric_time'.
    
    Args:
        filter_value: The filter string potentially containing TimeDimension calls
        
    Returns:
        Filter string with normalized dimension names
    """
    def replace_dimension(match: re.Match) -> str:
        dim_name = match.group(1)
        granularity = match.group(2)
        
        # If already using metric_time or entity__dimension format, keep it
        if dim_name == "metric_time" or "__" in dim_name:
            return match.group(0)
        
        # Replace bare column names with metric_time
        return f"{{{{ TimeDimension('metric_time', '{granularity}') }}}}"
    
    return TIME_DIMENSION_PATTERN.sub(replace_dimension, filter_value)


def is_abstract_filter(filter_value: str | None) -> bool:
    """Check if a filter value is an abstract concept that needs translation.
    
    Args:
        filter_value: The filter string to check
        
    Returns:
        True if the filter is abstract and needs translation, False otherwise
    """
    if not filter_value or not isinstance(filter_value, str):
        return False
    
    value = filter_value.strip()
    
    # Check for abstract time patterns
    if ABSTRACT_TIME_PATTERN.match(value):
        return True
    
    if SIMPLE_TIME_PATTERN.match(value):
        return True
    
    return False


def _get_interval_sql(num: int, unit: str) -> str:
    """Generate SQL interval syntax for DuckDB/standard SQL.
    
    Args:
        num: Number of units
        unit: Time unit (day, week, month, etc.)
        
    Returns:
        SQL interval string
    """
    # Normalize unit to singular
    unit_lower = unit.lower().rstrip('s')
    
    # Map to SQL interval units
    unit_map = {
        'day': 'day',
        'week': 'week', 
        'month': 'month',
        'quarter': 'month',  # Convert quarter to months
        'year': 'year',
    }
    
    sql_unit = unit_map.get(unit_lower, 'day')
    
    # Handle quarter conversion
    if unit_lower == 'quarter':
        num = num * 3
    
    return f"interval '{num} {sql_unit}s'"


def translate_filter(
    filter_value: str | None,
    time_dimension: str = "metric_time",
    granularity: str = "day",
) -> str | None:
    """Translate an abstract filter to valid MetricFlow Jinja syntax.
    
    Args:
        filter_value: The filter string (abstract or concrete)
        time_dimension: The time dimension to use (default: metric_time)
        granularity: The time granularity (default: day)
        
    Returns:
        Valid MetricFlow filter syntax, or None for empty input
        
    Examples:
        >>> translate_filter("last_30_days")
        "{{ TimeDimension('metric_time', 'day') }} >= current_date - interval '30 days'"
        
        >>> translate_filter("status = 'completed'")
        "status = 'completed'"
    """
    if not filter_value or not isinstance(filter_value, str):
        return None
    
    value = filter_value.strip()
    if not value:
        return None
    
    # Check for abstract time patterns like 'last_30_days'
    match = ABSTRACT_TIME_PATTERN.match(value)
    if match:
        num = int(match.group(1))
        unit = match.group(2)
        interval = _get_interval_sql(num, unit)
        return f"{{{{ TimeDimension('{time_dimension}', '{granularity}') }}}} >= current_date - {interval}"
    
    # Check for simple patterns like 'last_month'
    simple_match = SIMPLE_TIME_PATTERN.match(value)
    if simple_match:
        unit = simple_match.group(1).lower()
        unit_to_num = {
            'week': (7, 'day'),
            'month': (1, 'month'),
            'quarter': (3, 'month'),
            'year': (1, 'year'),
        }
        num, sql_unit = unit_to_num.get(unit, (30, 'day'))
        interval = f"interval '{num} {sql_unit}s'"
        return f"{{{{ TimeDimension('{time_dimension}', '{granularity}') }}}} >= current_date - {interval}"
    
    # Not an abstract filter - normalize any invalid dimension names in Jinja syntax
    return _normalize_time_dimension_names(value)


def translate_filters(
    filters: list[Any] | None,
    time_dimension: str = "metric_time",
    granularity: str = "day",
) -> list[str]:
    """Translate a list of filters, removing None/empty values.
    
    Args:
        filters: List of filter values (abstract or concrete)
        time_dimension: The time dimension to use
        granularity: The time granularity
        
    Returns:
        List of valid MetricFlow filter strings
    """
    if not filters:
        return []
    
    result = []
    for f in filters:
        if f is None:
            continue
        translated = translate_filter(str(f), time_dimension, granularity)
        if translated:
            result.append(translated)
    
    return result
