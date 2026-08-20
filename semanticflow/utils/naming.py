"""Utilities for normalizing names in SemanticFlow.

Provides consistent naming for metrics, semantic models, and other entities.
"""
from __future__ import annotations

import re


def to_snake_case(name: str) -> str:
    """Convert a name to snake_case.
    
    Args:
        name: Input name in any format (camelCase, PascalCase, spaces, etc.)
    
    Returns:
        snake_case version of the name
    
    Examples:
        >>> to_snake_case("OrdersPerDay")
        'orders_per_day'
        >>> to_snake_case("orders-per-day")
        'orders_per_day'
        >>> to_snake_case("Orders Per Day")
        'orders_per_day'
    """
    # Replace hyphens and spaces with underscores
    name = re.sub(r"[-\s]+", "_", name)
    # Insert underscore before uppercase letters (for camelCase/PascalCase)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    # Convert to lowercase and clean up multiple underscores
    name = re.sub(r"_+", "_", name.lower())
    # Remove leading/trailing underscores
    return name.strip("_")


def normalize_metric_name(
    name: str,
    normalize: bool = True,
    prefix: str | None = None,
) -> str:
    """Normalize a metric name for consistency.
    
    Args:
        name: The original metric name
        normalize: Whether to convert to snake_case
        prefix: Optional prefix to add
    
    Returns:
        Normalized metric name
    
    Examples:
        >>> normalize_metric_name("Orders Per Day")
        'orders_per_day'
        >>> normalize_metric_name("revenue", prefix="daily")
        'daily_revenue'
    """
    if normalize:
        name = to_snake_case(name)
    
    if prefix:
        prefix = to_snake_case(prefix) if normalize else prefix
        if not name.startswith(f"{prefix}_"):
            name = f"{prefix}_{name}"
    
    return name


def suggest_metric_name(
    base_measures: list[str],
    group_by: list[str],
    filters: list[str] | None = None,
) -> str:
    """Suggest a metric name based on its components.
    
    Args:
        base_measures: List of measure names
        group_by: List of dimension names
        filters: Optional list of filter descriptions
    
    Returns:
        Suggested metric name
    
    Examples:
        >>> suggest_metric_name(["order_count"], ["order_date"])
        'order_count_by_order_date'
        >>> suggest_metric_name(["revenue"], ["customer"], ["last_30_days"])
        'revenue_by_customer_last_30_days'
    """
    parts: list[str] = []
    
    # Start with measure(s)
    if base_measures:
        parts.append(to_snake_case(base_measures[0]))
    else:
        parts.append("metric")
    
    # Add dimensions
    if group_by:
        dim_part = "_".join(to_snake_case(d) for d in group_by[:2])  # Limit to 2 dims
        parts.append(f"by_{dim_part}")
    
    # Add filter hints (simplified)
    if filters:
        for f in filters[:1]:  # Only first filter
            f_lower = f.lower()
            if "last" in f_lower and "day" in f_lower:
                match = re.search(r"last_?(\d+)_?days?", f_lower)
                if match:
                    parts.append(f"last_{match.group(1)}_days")
                    break
            elif "last" in f_lower and "month" in f_lower:
                parts.append("last_month")
                break
    
    return "_".join(parts)
