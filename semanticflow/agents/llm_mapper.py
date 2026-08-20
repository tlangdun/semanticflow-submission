"""
LLM-powered mapping utilities for semantic layer generation.

Uses LLM calls to intelligently map:
- Measure concepts to actual schema columns
- Aggregation types based on column semantics
- Entity relationships between models

Designed for robustness:
- Validates LLM output against actual schema
- Handles malformed responses gracefully
- Provides detailed logging for debugging
- Returns None on any failure (caller uses fallback)
"""
from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field, field_validator

from semanticflow.llm import BaseLLMClient
from semanticflow.log import get_logger

logger = get_logger("semanticflow.llm_mapper")

# Valid aggregation types for MetricFlow
VALID_AGGREGATIONS = {"sum", "count", "avg", "min", "max", "count_distinct", "average"}
# Confidence threshold - below this we don't trust the LLM result
MIN_CONFIDENCE_THRESHOLD = 0.3


class MeasureMapping(BaseModel):
    """Result of LLM measure-to-column mapping."""
    column: str = Field(description="The actual column name to use")
    aggregation: str = Field(description="Aggregation type: sum, count, avg, min, max")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Confidence in this mapping")
    reasoning: str = Field(default="", description="Why this mapping was chosen")
    
    @field_validator("aggregation")
    @classmethod
    def normalize_aggregation(cls, v: str) -> str:
        """Normalize aggregation type to valid MetricFlow values."""
        v_lower = v.lower().strip()
        # Map common variations
        if v_lower in ("average", "avg", "mean"):
            return "average"
        if v_lower in ("count", "count_distinct"):
            return v_lower
        if v_lower in VALID_AGGREGATIONS:
            return v_lower
        # Default to sum for unknown
        logger.warning(f"Unknown aggregation '{v}', defaulting to 'sum'")
        return "sum"


class MeasureMappingResult(BaseModel):
    """Wrapper for LLM mapping result with validation status."""
    mapping: MeasureMapping | None = None
    success: bool = False
    error: str | None = None
    latency_ms: float = 0.0
    
    def __bool__(self) -> bool:
        return self.success and self.mapping is not None


# `from __future__ import annotations` turns the `mapping: MeasureMapping | None` field
# into a string annotation; under pydantic 2.12 the model must be rebuilt so the forward
# reference resolves, otherwise the first instantiation raises PydanticUserError.
MeasureMappingResult.model_rebuild()


class EntityMapping(BaseModel):
    """Result of LLM entity relationship inference."""
    target_model: str = Field(description="The target model for this entity")
    join_column: str = Field(description="The column to join on")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


def _validate_mapping(
    result: MeasureMapping,
    columns: list[str],
    measure_concept: str,
) -> tuple[bool, str]:
    """
    Validate that the LLM mapping is usable.
    
    Returns:
        (is_valid, error_message)
    """
    # Check column exists in schema
    columns_lower = {c.lower(): c for c in columns}
    result_col_lower = result.column.lower()
    
    if result_col_lower not in columns_lower:
        # Try fuzzy match - maybe LLM got the case wrong
        return False, f"Column '{result.column}' not found in schema. Available: {columns[:5]}..."
    
    # Normalize column name to actual case from schema
    result.column = columns_lower[result_col_lower]
    
    # Check confidence threshold
    if result.confidence < MIN_CONFIDENCE_THRESHOLD:
        return False, f"Confidence {result.confidence:.2f} below threshold {MIN_CONFIDENCE_THRESHOLD}"
    
    # Check aggregation is valid
    if result.aggregation not in VALID_AGGREGATIONS:
        return False, f"Invalid aggregation '{result.aggregation}'"
    
    return True, ""


async def llm_map_measure(
    measure_concept: str,
    columns: list[str],
    model_name: str,
    llm_client: BaseLLMClient | None,
    data_types: dict[str, str] | None = None,
) -> MeasureMappingResult:
    """
    Use LLM to map a measure concept to an actual column.
    
    Args:
        measure_concept: The conceptual measure name (e.g., "order_count", "revenue")
        columns: Available columns in the model
        model_name: Name of the dbt model
        llm_client: LLM client for making the call
        data_types: Optional column data types
        
    Returns:
        MeasureMappingResult with mapping or error details
    """
    start_time = time.time()
    
    # Input validation
    if not measure_concept or not measure_concept.strip():
        return MeasureMappingResult(error="Empty measure concept")
    
    if not columns:
        return MeasureMappingResult(error="No columns provided")
    
    if not llm_client:
        return MeasureMappingResult(error="No LLM client available")
    
    # Build column info string with data types
    col_info = []
    for col in columns:
        dtype = data_types.get(col, "unknown") if data_types else "unknown"
        col_info.append(f"- {col} ({dtype})")
    
    prompt = f"""Map the measure concept to an actual column from the schema.

Measure concept: "{measure_concept}"
Model: {model_name}
Available columns (you MUST pick one of these exact names):
{chr(10).join(col_info)}

Rules:
1. The "column" field MUST be one of the exact column names listed above
2. If the measure implies counting (e.g., "order_count", "num_users"):
   - If a pre-aggregated column exists (like "number_of_orders"), use it with aggregation="sum"
   - Otherwise, find an ID column and use aggregation="count"
3. If the measure implies summing (e.g., "revenue", "total_amount"):
   - Find a numeric column (amount, price, value, etc.) and use aggregation="sum"
4. Set confidence based on how sure you are (0.0-1.0)
5. If no column matches well, set confidence < 0.3

Return JSON with these exact fields:
- column: exact column name from the list (case-sensitive)
- aggregation: one of [sum, count, avg, min, max]
- confidence: 0.0 to 1.0
- reasoning: brief explanation of your choice
"""

    logger.debug(
        "LLM measure mapping request",
        extra={
            "measure_concept": measure_concept,
            "model_name": model_name,
            "num_columns": len(columns),
            "columns": columns[:10],
        }
    )

    try:
        result = llm_client.complete_json(
            "You are an expert at mapping semantic measure concepts to database columns. "
            "Always return valid JSON with the exact column name from the provided list.",
            prompt,
            MeasureMapping,
            temperature=0.1,
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Validate the result
        is_valid, error = _validate_mapping(result, columns, measure_concept)
        
        if not is_valid:
            logger.warning(
                "LLM mapping validation failed",
                extra={
                    "measure_concept": measure_concept,
                    "llm_column": result.column,
                    "error": error,
                    "latency_ms": latency_ms,
                }
            )
            return MeasureMappingResult(error=error, latency_ms=latency_ms)
        
        logger.info(
            f"LLM mapped '{measure_concept}' -> {result.column} ({result.aggregation})",
            extra={
                "measure_concept": measure_concept,
                "column": result.column,
                "aggregation": result.aggregation,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
                "latency_ms": latency_ms,
            }
        )
        
        return MeasureMappingResult(
            mapping=result,
            success=True,
            latency_ms=latency_ms,
        )
        
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(
            "LLM measure mapping failed",
            extra={
                "measure_concept": measure_concept,
                "model_name": model_name,
                "error": error_msg,
                "latency_ms": latency_ms,
            },
            exc_info=True,
        )
        return MeasureMappingResult(error=error_msg, latency_ms=latency_ms)


async def llm_map_entity(
    entity_name: str,
    available_models: list[str],
    schema_context: dict[str, Any],
    llm_client: BaseLLMClient | None,
) -> EntityMapping | None:
    """
    Use LLM to infer the target model for an entity relationship.
    
    Args:
        entity_name: The entity name (e.g., "customer")
        available_models: List of available dbt models
        schema_context: Full schema context with columns
        llm_client: LLM client
        
    Returns:
        EntityMapping with target model and join column, or None on failure
    """
    if not llm_client or not entity_name or not available_models:
        return None
    
    # Build model info
    model_info = []
    for model in available_models:
        cols = schema_context.get("columns", {}).get(model, [])
        model_info.append(f"- {model}: columns={cols[:5]}{'...' if len(cols) > 5 else ''}")
    
    prompt = f"""Find the target model for this entity relationship.

Entity: "{entity_name}" (likely a foreign key like {entity_name}_id)
Available models (pick one):
{chr(10).join(model_info)}

Return JSON with:
- target_model: exact model name from the list above
- join_column: the column to join on (usually the primary key)
- confidence: 0.0-1.0
"""

    try:
        result = llm_client.complete_json(
            "You are an expert at inferring database relationships.",
            prompt,
            EntityMapping,
            temperature=0.1,
        )
        # Validate target_model exists
        if result.target_model not in available_models:
            logger.warning(f"LLM returned invalid model '{result.target_model}'")
            return None
        return result
    except Exception as e:
        logger.warning(f"LLM entity mapping failed: {e}")
        return None


def sync_map_measure(
    measure_concept: str,
    columns: list[str],
    model_name: str,
    llm_client: BaseLLMClient | None,
    data_types: dict[str, str] | None = None,
) -> MeasureMapping | None:
    """
    Synchronous wrapper for llm_map_measure.
    
    Returns the MeasureMapping if successful, None otherwise.
    Use this from sync code that needs LLM measure mapping.
    """
    import asyncio
    
    async def _async_map() -> MeasureMappingResult:
        return await llm_map_measure(measure_concept, columns, model_name, llm_client, data_types)
    
    try:
        # Try to get existing event loop
        try:
            loop = asyncio.get_running_loop()
            # Already in async context - create a new thread to run
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _async_map())
                result = future.result(timeout=30)  # 30 second timeout
        except RuntimeError:
            # No running loop - safe to use asyncio.run
            result = asyncio.run(_async_map())
        
        if result and result.success and result.mapping:
            return result.mapping
        
        if result and result.error:
            logger.debug(f"sync_map_measure failed: {result.error}")
        
        return None
        
    except Exception as e:
        logger.warning(f"sync_map_measure exception: {e}")
        return None


# ============================================================================
# Testing utilities
# ============================================================================

class MockLLMClient:
    """Mock LLM client for testing."""
    
    def __init__(self, responses: dict[str, MeasureMapping] | None = None):
        self.responses = responses or {}
        self.calls: list[dict[str, Any]] = []
    
    def complete_json(
        self,
        system: str,
        prompt: str,
        response_model: type,
        temperature: float = 0.1,
    ) -> Any:
        """Return mock response based on measure concept in prompt."""
        self.calls.append({
            "system": system,
            "prompt": prompt,
            "response_model": response_model,
            "temperature": temperature,
        })
        
        # Extract measure concept from prompt
        import re
        match = re.search(r'Measure concept: "([^"]+)"', prompt)
        if match:
            concept = match.group(1)
            if concept in self.responses:
                return self.responses[concept]
        
        # Default response
        return MeasureMapping(
            column="id",
            aggregation="count",
            confidence=0.5,
            reasoning="Mock response",
        )
