from __future__ import annotations

import re
from typing import Any

from semanticflow.dbt_integration.filter_utils import translate_filter


def _find_columns(schema_context: dict) -> set[str]:
    columns = set()
    for cols in schema_context.get("columns", {}).values():
        for col in cols:
            columns.add(col)
    return columns


def _parse_limit(text: str) -> int | None:
    match = re.search(r"top\s+(\d+)", text)
    if match:
        return int(match.group(1))
    return None


def heuristic_semantic_spec(nl_request: str, schema_context: dict) -> dict[str, Any]:
    text = nl_request.lower()
    columns = _find_columns(schema_context)
    group_by: list[str] = []
    filters: list[str] = []
    ambiguity_points: list[str] = []
    clarifying_questions: list[str] = []

    time_granularity = None
    if "per day" in text or "daily" in text or "by day" in text:
        group_by.append("order_date")
        time_granularity = "day"
    if "per month" in text or "monthly" in text:
        group_by.append("order_date")
        time_granularity = "month"

    # Generate valid MetricFlow filter syntax instead of abstract names
    if "last 30 days" in text:
        filters.append(translate_filter("last_30_days", time_dimension="metric_time"))
    if "last quarter" in text:
        filters.append(translate_filter("last_quarter", time_dimension="metric_time"))
    if "last month" in text:
        filters.append(translate_filter("last_month", time_dimension="metric_time"))

    if "top" in text and "customer" in text:
        group_by.append("customer")

    if "region" in text or "country" in text:
        if "country" in columns:
            group_by.append("country")
        elif "customers" in schema_context.get("models", []):
            group_by.append("customer_country")
        else:
            group_by.append("region")

    base_measures: list[str] = []
    metric_name = None

    if "lifetime value" in text:
        base_measures = ["customer_lifetime_value"]
        metric_name = "customer_lifetime_value"
    elif "revenue" in text or "amount" in text:
        base_measures = ["gross_revenue"]
    elif "order" in text:
        base_measures = ["order_count"]

    if metric_name is None:
        if base_measures == ["order_count"] and time_granularity == "day":
            metric_name = "orders_per_day"
        elif base_measures == ["gross_revenue"] and "region" in text:
            metric_name = "gross_revenue_by_region"
        elif base_measures and group_by:
            metric_name = f"{base_measures[0]}_by_{group_by[0]}"
        elif base_measures:
            metric_name = base_measures[0]
        else:
            metric_name = "metric"

    uncertainty_score = 0.2
    if "region" in text:
        ambiguity_points.append("Region may map to customer country or another geo field.")
        clarifying_questions.append("Should region correspond to customer country?")
        uncertainty_score = max(uncertainty_score, 0.6)
    if "revenue" in text:
        ambiguity_points.append("Revenue could mean gross order amount or a net definition.")
        clarifying_questions.append("Do you mean gross revenue from orders or a net revenue definition?")
        uncertainty_score = max(uncertainty_score, 0.5)

    order_by = None
    limit = _parse_limit(text)
    if "top" in text or "highest" in text:
        field = base_measures[0] if base_measures else metric_name
        order_by = [{"field": field, "direction": "desc"}]

    return {
        "metric_name": metric_name,
        "base_measures": base_measures,
        "group_by": list(dict.fromkeys(group_by)),
        "time_granularity": time_granularity,
        "filters": filters,
        "definition_notes": "",
        "confidence": 0.4,
        "uncertainty_score": uncertainty_score,
        "ambiguity_points": ambiguity_points,
        "clarifying_questions": clarifying_questions,
        "order_by": order_by,
        "limit": limit,
    }
