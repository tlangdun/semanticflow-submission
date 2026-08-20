from __future__ import annotations

from typing import Any

import yaml
from pydantic import BaseModel, Field

from semanticflow.dbt_integration.filter_utils import translate_filter


def _normalize_model_ref(model: str) -> str:
    if model.strip().startswith("ref("):
        return model
    return f"ref('{model}')"


def _normalize_metric_filter(value: str | None, time_dimension: str = "metric_time") -> str | None:
    """Normalize and translate a metric filter to valid MetricFlow syntax.
    
    Args:
        value: The filter value (abstract or concrete)
        time_dimension: The time dimension to use for translation
        
    Returns:
        Valid MetricFlow filter syntax, or None for empty input
    """
    if not value or not isinstance(value, str):
        return None
    
    # Translate abstract filters to valid MetricFlow syntax
    return translate_filter(value, time_dimension=time_dimension)


class EntitySpec(BaseModel):
    name: str
    type: str = "primary"
    expr: str | None = None
    description: str | None = None


class DimensionSpec(BaseModel):
    name: str
    type: str = "categorical"
    expr: str | None = None
    type_params: dict[str, Any] | None = None
    description: str | None = None


class MeasureSpec(BaseModel):
    name: str
    agg: str = "sum"
    expr: str | None = None
    description: str | None = None


class SemanticModelSpec(BaseModel):
    name: str
    description: str | None = None
    model: str
    entities: list[EntitySpec] = Field(default_factory=list)
    measures: list[MeasureSpec] = Field(default_factory=list)
    dimensions: list[DimensionSpec] = Field(default_factory=list)
    primary_entity: str | None = None
    defaults: dict[str, Any] | None = None


class MetricSpec(BaseModel):
    name: str
    description: str | None = None
    type: str = "simple"
    type_params: dict[str, Any] = Field(default_factory=dict)
    filter: str | None = None
    filters: list[str] | None = None


def _dump_yaml(payload: dict[str, Any]) -> str:
    return yaml.safe_dump(payload, sort_keys=False)


def semantic_model_to_dict(spec: SemanticModelSpec) -> dict[str, Any]:
    data = spec.model_dump(exclude_none=True)
    data["model"] = _normalize_model_ref(spec.model)
    entities = data.get("entities") or []
    if any(entity.get("type") == "primary" for entity in entities if isinstance(entity, dict)):
        data.pop("primary_entity", None)
    return data


def metric_to_dict(spec: MetricSpec) -> dict[str, Any]:
    data = spec.model_dump(exclude_none=True)
    if "label" not in data:
        data["label"] = data.get("description") or data.get("name")
    filters = data.pop("filters", None)
    if "filter" in data:
        normalized_filter = _normalize_metric_filter(data.get("filter"))
        if normalized_filter:
            data["filter"] = normalized_filter
        else:
            data.pop("filter", None)
    if filters and not data.get("filter"):
        normalized = [_normalize_metric_filter(item) for item in filters if item]
        normalized = [item for item in normalized if item]
        if normalized:
            data["filter"] = " and ".join(normalized)
    return data


def render_semantic_model_yaml(spec: SemanticModelSpec) -> str:
    payload = {"semantic_models": [semantic_model_to_dict(spec)]}
    return _dump_yaml(payload)


def render_metric_yaml(spec: MetricSpec) -> str:
    payload = {"metrics": [metric_to_dict(spec)]}
    return _dump_yaml(payload)
