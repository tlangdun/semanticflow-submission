from __future__ import annotations

from pathlib import Path
from typing import Any

from semanticflow.agents.types import Agent, AgentMessage
from semanticflow.dbt_integration import (
    DimensionSpec,
    EntitySpec,
    MeasureSpec,
    MetricSpec,
    SemanticModelSpec,
    metric_exists,
    semantic_model_exists,
)
from semanticflow.log import get_logger
from semanticflow.utils.naming import normalize_metric_name, suggest_metric_name

logger = get_logger("semanticflow.designer")

# Expression for the synthetic time dimension injected when a model has no real date
# column (MetricFlow requires every measure to have an agg_time_dimension). It must be a
# DATE *literal*, not the `current_date` function: MetricFlow qualifies a bare identifier
# with the source-table alias (e.g. `stg_payments_src.current_date`), which fails the
# binder in join/validation contexts. A constant inside the warehouse's 2018 data range is
# also safer than an out-of-range `current_date` for any accidental time grouping.
_SYNTHETIC_DS_EXPR = "cast('2018-01-01' as date)"


def _sanitize_name(name: str) -> str:
    """Sanitize a name to be valid for MetricFlow/dbt.
    
    MetricFlow names must:
    - Only contain lowercase letters, numbers, underscores
    - Start with a lowercase letter
    - Not end with underscore
    - Not contain double underscores
    - Be at least 2 characters
    
    This handles common LLM outputs like:
    - COUNT(orders.order_id) -> order_count
    - orders.order_date -> order_date
    - SUM(amount) -> amount_sum
    """
    import re
    
    original = name
    
    # Handle SQL function patterns: COUNT(x), SUM(x), AVG(x), etc.
    sql_func_match = re.match(r'(COUNT|SUM|AVG|MIN|MAX)\s*\(\s*(?:[\w.]+\.)?(\w+)\s*\)', name, re.IGNORECASE)
    if sql_func_match:
        func, column = sql_func_match.groups()
        func_lower = func.lower()
        # Map to semantic names
        if func_lower == 'count':
            name = f"{column}_count" if column != 'id' else 'record_count'
        elif func_lower == 'sum':
            name = f"{column}_sum"
        elif func_lower == 'avg':
            name = f"{column}_avg"
        else:
            name = f"{column}_{func_lower}"
    
    # Remove table prefixes: orders.order_date -> order_date
    if '.' in name:
        name = name.split('.')[-1]
    
    # Remove any remaining parentheses and their contents
    name = re.sub(r'\([^)]*\)', '', name)
    
    # Replace invalid characters with underscores
    name = re.sub(r'[^a-z0-9_]', '_', name.lower())
    
    # Remove leading/trailing underscores
    name = name.strip('_')
    
    # Replace multiple underscores with single
    name = re.sub(r'_+', '_', name)
    
    # Ensure starts with letter
    if name and not name[0].isalpha():
        name = 'col_' + name
    
    # Ensure at least 2 characters
    if len(name) < 2:
        name = name + '_col' if name else 'column'
    
    if name != original:
        logger.debug(f"Sanitized name: '{original}' -> '{name}'")
    
    return name


def _sanitize_measure_name(name: str, model_columns: set[str]) -> tuple[str, str | None]:
    """Sanitize measure name and determine expression.
    
    Returns (sanitized_name, expr) where expr is the column to aggregate.
    """
    import re
    
    # Handle SQL function patterns
    sql_func_match = re.match(r'(COUNT|SUM|AVG|MIN|MAX)\s*\(\s*(?:[\w.]+\.)?(\w+)\s*\)', name, re.IGNORECASE)
    if sql_func_match:
        func, column = sql_func_match.groups()
        # Clean column name (remove any table prefix)
        column = column.split('.')[-1] if '.' in column else column
        
        func_lower = func.lower()
        if func_lower == 'count':
            sanitized = f"{column}_count" if column not in ('id', '*') else 'record_count'
            expr = column if column in model_columns else None
        else:
            sanitized = f"{column}_{func_lower}"
            expr = column if column in model_columns else None
        return sanitized, expr
    
    # Simple name - just sanitize
    sanitized = _sanitize_name(name)
    expr = name if name in model_columns else None
    return sanitized, expr


def _normalize_enum_literals(value_expr: str) -> str:
    """Normalize categorical string literals to the warehouse's enum convention.

    LLMs frequently hyphenate multi-word enum values ('return-pending') where seeds use
    underscores ('return_pending'), so a syntactically valid filter matches zero rows. We
    rewrite hyphens/spaces to underscores INSIDE single-quoted literals only, and never
    touch date-shaped literals ('2018-04-01') so time bounds are preserved.
    """
    import re

    def repl(m: re.Match) -> str:
        inner = m.group(1)
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", inner):  # date literal — leave intact
            return m.group(0)
        return "'" + re.sub(r"[-\s]+", "_", inner) + "'"

    return re.sub(r"'([^']*)'", repl, value_expr)


def _extract_filter_columns(filters: list[str]) -> set[str]:
    """Column names referenced on the left of each filter comparison.

    A filter can only resolve if its column is a registered dimension; columns that appear
    *only* in filters (never in group_by) must still be emitted as dimensions or MetricFlow
    raises a binder error on the metric's WHERE clause. Handles bare SQL and the Jinja
    Dimension form.
    """
    import re

    cols: set[str] = set()
    for raw in filters or []:
        s = str(raw)
        if "{{" in s:
            cols.update(re.findall(r"(?:Dimension|TimeDimension)\('[^']*__(\w+)'\)", s))
            continue
        m = re.match(
            r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:NOT\s+IN|IN|!=|<>|>=|<=|=|>|<)",
            s, re.IGNORECASE,
        )
        if m:
            cols.add(m.group(1))
    return cols


def _sanitize_filter(
    filter_str: str,
    model: str,
    primary_entity: str | None = None,
    dimension_names: set[str] | None = None,
    dim_entity_map: dict[str, str] | None = None,
) -> str | None:
    """Sanitize filter to use proper MetricFlow syntax.

    Converts column references like:
    - orders.order_date >= current_date -> {{ TimeDimension('metric_time', 'day') }} >= current_date
    - status = 'completed' -> {{ Dimension('order__status') }} = 'completed'

    A metric's ``filter:`` is rendered into the warehouse query, so a bare-SQL column
    reference (``WHERE status = 'completed'``) raises a binder error — MetricFlow requires
    the Jinja ``Dimension``/``TimeDimension`` form. Categorical conversion is scoped to
    known dimension columns so we never rewrite a measure threshold or a subquery.
    """
    import re

    if not filter_str or not filter_str.strip():
        return None

    # Unwrap a `col IN (SELECT col FROM model WHERE <inner>)` the model sometimes emits in
    # place of a plain dimension filter — keep just the inner condition so it can be
    # translated (e.g. `order_id IN (SELECT ... WHERE status = 'completed')` -> `status =
    # 'completed'`). Only unwrap when the subselect carries a WHERE clause.
    sub = re.match(
        r"^\s*[A-Za-z_][\w.]*\s+(?:NOT\s+)?IN\s*\(\s*SELECT\b.*?\bWHERE\b(.+?)\)\s*$",
        filter_str, re.IGNORECASE | re.DOTALL,
    )
    if sub:
        filter_str = sub.group(1).strip()

    # Already has Jinja syntax - pass through
    if '{{' in filter_str:
        return filter_str

    # Check if filter references a date/time column with table prefix
    # Pattern: table.column or just column with date-related comparison
    # The date/time token must start the column or follow an underscore, so a substring
    # like "time" inside "customer_lifetime_value" is NOT misread as a time column.
    time_col_pattern = rf'{model}\.((?:\w*_)?(?:date|time|day|created|updated)\w*)'
    match = re.search(time_col_pattern, filter_str, re.IGNORECASE)
    
    if match:
        # Replace table.column with TimeDimension syntax
        col_name = match.group(1)
        jinja_dim = "{{ TimeDimension('metric_time', 'day') }}"
        sanitized = re.sub(time_col_pattern, jinja_dim, filter_str, flags=re.IGNORECASE)
        logger.debug(f"Sanitized filter: '{filter_str}' -> '{sanitized}'")
        return sanitized
    
    # Check for bare date column references (no table prefix)
    bare_time_pattern = r'\b((?:\w*_)?(?:date|time|day|created|updated)\w*)\s*(>=|<=|>|<|=)'
    bare_match = re.search(bare_time_pattern, filter_str, re.IGNORECASE)
    
    if bare_match and '{{' not in filter_str:
        col_name = bare_match.group(1)
        jinja_dim = "{{ TimeDimension('metric_time', 'day') }}"
        sanitized = re.sub(rf'\b{col_name}\b', jinja_dim, filter_str, count=1)
        logger.debug(f"Sanitized filter: '{filter_str}' -> '{sanitized}'")
        return sanitized
    
    # Categorical dimension filter: `status = 'completed'`, `status != 'returned'`,
    # `status IN ('a','b')` -> `{{ Dimension('order__status') }} <op> <value>`. Scoped to
    # real dimension columns so measure thresholds / subqueries are left untouched.
    if primary_entity or dim_entity_map:
        cat = re.match(
            r"^([A-Za-z_][A-Za-z0-9_]*)\s+(NOT\s+IN|IN|!=|<>|>=|<=|>|<|=)\s+(.+)$",
            filter_str.strip(),
            re.IGNORECASE,
        )
        if cat:
            field, op, val = cat.group(1), cat.group(2).upper(), cat.group(3).strip()
            # Prefer the field's own model entity (for cross-model filters, e.g. a `status`
            # filter on `orders` while the measure lives on `stg_payments`); fall back to the
            # primary entity for columns on the primary model.
            entity = (dim_entity_map or {}).get(field)
            if entity is None and dimension_names and field in dimension_names:
                entity = primary_entity
            if entity:
                # Numeric thresholds (`> 30`, `>= 2`) keep their literal value; only
                # categorical equality/membership needs enum-literal quoting.
                if op not in (">", "<", ">=", "<="):
                    val = _normalize_enum_literals(val)
                sanitized = f"{{{{ Dimension('{entity}__{field}') }}}} {op} {val}"
                logger.debug(f"Sanitized filter: '{filter_str}' -> '{sanitized}'")
                return sanitized

    # Non-time filter - return as-is
    return filter_str


def _measure_to_column(measure_name: str) -> list[str]:
    """
    Generate candidate column names from a measure concept.
    
    Fully generalizable - extracts core concept and generates variations.
    E.g., "order_count" -> ["order_count", "order", "number_of_order", "order_id", ...]
    """
    lowered = measure_name.lower()
    
    # Remove aggregation suffixes/prefixes to get core concept
    agg_words = {"sum", "count", "avg", "total", "num", "number", "of"}
    parts = lowered.replace("_", " ").split()
    core_parts = [p for p in parts if p not in agg_words and len(p) >= 2]
    
    candidates = [measure_name, lowered]
    
    if core_parts:
        base = "_".join(core_parts)
        candidates.extend([
            base,
            f"{base}_id",  # For count measures
            f"number_of_{base}",
            f"num_{base}",
            f"{base}_count",
            f"total_{base}",
            f"{base}_total",
        ])
        # Also try singular/plural variations
        for part in core_parts:
            candidates.append(part)
            candidates.append(f"{part}_id")
            candidates.append(f"number_of_{part}s" if not part.endswith("s") else f"number_of_{part}")
    
    return candidates


def _choose_model(semantic_spec: dict[str, Any], schema_context: dict[str, Any]) -> str:
    """Choose the best model based on which one contains the required measure columns."""
    models = schema_context.get("models", [])
    columns_by_model = schema_context.get("columns", {})
    base_measures = semantic_spec.get("base_measures", [])
    group_by = [str(g).lower() for g in (semantic_spec.get("group_by") or [])]

    def _cols(model: str) -> set[str]:
        return {c.lower() for c in columns_by_model.get(model, [])}

    # First priority: models that contain the measure column (direct, then via likely names).
    measure_models: list[str] = []
    for measure in base_measures:
        measure_lower = measure.lower()
        cands = [m for m in models if measure_lower in _cols(m)]
        if not cands:
            likely = {c.lower() for c in _measure_to_column(measure)}
            cands = [m for m in models if _cols(m) & likely]
        for m in cands:
            if m not in measure_models:
                measure_models.append(m)

    if measure_models:
        # Prefer a measure-bearing model that ALSO carries the group-by dimensions, so the
        # measure and its grouping are co-located. Grouping a measure by a dimension on a
        # *different* model forces a one-to-many (fan-out) join that MetricFlow rejects;
        # co-locating avoids it (e.g. SUM(stg_payments.amount) BY stg_payments.payment_method
        # rather than SUM(orders.amount) fanned out across payments).
        if group_by:
            best = max(measure_models, key=lambda m: sum(1 for g in group_by if g in _cols(m)))
            if any(g in _cols(best) for g in group_by):
                return best
        return measure_models[0]
    
    # Second priority: use word overlap between signals and model names
    # No hardcoded model names - works for any database
    signals = " ".join(
        base_measures
        + semantic_spec.get("group_by", [])
        + [semantic_spec.get("metric_name", "")]
    ).lower()
    signal_words = set(signals.replace("_", " ").split())
    
    # Score each model by how many signal words overlap with the model name
    best_model = None
    best_score = 0
    for model in models:
        model_words = set(model.lower().replace("_", " ").split())
        # Also check singular/plural variations
        model_variations = model_words | {w.rstrip("s") for w in model_words} | {w + "s" for w in model_words}
        score = len(signal_words & model_variations)
        if score > best_score:
            best_score = score
            best_model = model
    
    if best_model:
        return best_model
    
    # Fallback: return first model (no hardcoded default)
    return models[0] if models else ""


def _column_lookup(schema_context: dict[str, Any], model: str) -> set[str]:
    return set(schema_context.get("columns", {}).get(model, []))


def _find_field_model(field: str, schema_context: dict[str, Any]) -> str | None:
    """Find which model contains a given field/column."""
    columns_by_model = schema_context.get("columns", {})
    field_lower = field.lower()
    
    for model, cols in columns_by_model.items():
        if field_lower in {c.lower() for c in cols}:
            return model
    return None


def _find_join_key(model1: str, model2: str, schema_context: dict[str, Any]) -> str | None:
    """Find a common join key between two models (e.g., customer_id)."""
    cols1 = set(schema_context.get("columns", {}).get(model1, []))
    cols2 = set(schema_context.get("columns", {}).get(model2, []))
    
    # Look for common *_id columns
    common = cols1 & cols2
    for col in common:
        if col.endswith("_id"):
            return col
    
    # Look for foreign key pattern: model2_singular_id in model1
    model2_singular = model2.rstrip('s') if model2.endswith('s') else model2
    fk_candidate = f"{model2_singular}_id"
    if fk_candidate in cols1:
        return fk_candidate
    
    return None


def _build_entities(model: str, model_columns: set[str], data_types: dict[str, str] | None = None) -> list[EntitySpec]:
    """Infer entities from model columns using naming conventions and data types.
    
    Primary entity: Look for {model_singular}_id or id column
    Foreign entities: Look for *_id columns that reference other tables
    """
    entities: list[EntitySpec] = []
    
    # Derive singular form of model name for primary key detection
    model_singular = model.rstrip('s') if model.endswith('s') else model
    
    # Common primary key patterns
    primary_candidates = [
        f"{model_singular}_id",  # e.g., order_id for orders
        "id",
        f"{model}_id",
    ]
    
    # Find primary entity
    primary_col = None
    for candidate in primary_candidates:
        if candidate in model_columns:
            primary_col = candidate
            break
    
    if primary_col:
        entities.append(EntitySpec(
            name=model_singular,
            type="primary",
            expr=primary_col
        ))
    
    # Find foreign entities (columns ending in _id that aren't the primary key)
    for col in model_columns:
        if col.endswith("_id") and col != primary_col:
            # Derive entity name from column (e.g., customer_id -> customer)
            entity_name = col[:-3]  # Remove "_id" suffix
            entities.append(EntitySpec(
                name=entity_name,
                type="foreign",
                expr=col
            ))
    
    return entities


# Numeric types that support SUM aggregation
NUMERIC_TYPES = {"integer", "int", "bigint", "smallint", "decimal", "numeric", "real", "double", "float", "money"}
# Types that are typically used for COUNT
ID_TYPES = {"uuid", "varchar", "text", "string"}


# MetricFlow measure aggregations we accept as an explicit request-level override. Maps
# common synonyms the mapper may emit onto MetricFlow's canonical agg names. "ratio"/
# "percentage" is deliberately absent: it is a ratio *metric* (two measures), not a measure
# agg, so we leave those to the verifier/HITL rather than silently picking a wrong agg.
_AGG_SYNONYMS = {
    "sum": "sum", "total": "sum",
    "average": "average", "avg": "average", "mean": "average",
    "count": "count", "count_distinct": "count_distinct",
    "distinct_count": "count_distinct", "distinct": "count_distinct", "unique": "count_distinct",
    "min": "min", "minimum": "min",
    "max": "max", "maximum": "max",
    "median": "median",
}


def normalize_requested_agg(value: Any) -> str | None:
    """Map a mapper-supplied aggregation onto a canonical MetricFlow agg, or None.

    None means "no explicit request" — the designer keeps its column-name inference. This
    is the fix for the avg->SUM bug: "average order value" used to infer SUM from the
    column name 'amount'; an explicit aggregation='average' now overrides that.
    """
    if not isinstance(value, str):
        return None
    return _AGG_SYNONYMS.get(value.strip().lower())


# Request phrases -> canonical agg, checked in priority order so "distinct" beats "count"
# and "average" beats "sum". Used as a deterministic fallback when the mapper LLM omits the
# aggregation field (observed: Claude frequently returns aggregation=null even when asked).
# "percentage"/"ratio" is intentionally excluded — it is a ratio metric, not a measure agg.
_REQUEST_AGG_PHRASES = [
    (("average ", "avg ", " mean ", "mean "), "average"),
    (("median",), "median"),
    (("distinct", "unique", "how many different"), "count_distinct"),
    (("maximum", "highest", "largest", "most "), "max"),
    (("minimum", "lowest", "smallest", "least "), "min"),
    (("how many", "number of", "count of", "count ", "# of"), "count"),
    (("total ", "sum of", "how much", "overall "), "sum"),
]


def agg_from_request(nl_request: str | None) -> str | None:
    """Detect the requested aggregation from the NL verb, or None if none is implied."""
    nl = f" {(nl_request or '').lower().strip()} "
    if any(p in nl for p in ("percentage", "percent", " ratio", "proportion", " rate ")):
        return None  # ratio metric — leave to verifier/HITL, don't pick a wrong agg
    for phrases, agg in _REQUEST_AGG_PHRASES:
        if any(p in nl for p in phrases):
            return agg
    return None


def _infer_aggregation(column: str, data_types: dict[str, str] | None) -> str:
    """Infer appropriate aggregation based on column name and data type."""
    # Check column naming conventions first
    if column.endswith("_id") or column == "id":
        return "count"
    if any(kw in column.lower() for kw in ["count", "qty", "quantity"]):
        return "sum"
    if any(kw in column.lower() for kw in ["amount", "price", "total", "revenue", "cost", "value"]):
        return "sum"
    if any(kw in column.lower() for kw in ["avg", "average", "rate"]):
        return "average"
    
    # Check data type if available
    if data_types:
        col_type = data_types.get(column, "").lower()
        if any(t in col_type for t in NUMERIC_TYPES):
            return "sum"
        if any(t in col_type for t in ID_TYPES):
            return "count"
    
    return "sum"  # Default to sum


# Patterns indicating pre-aggregated count columns (use SUM, not COUNT)
PREAGG_COUNT_PATTERNS = {
    "prefixes": ["num_", "number_of_", "count_", "total_", "n_"],
    "suffixes": ["_count", "_total", "_num", "_number"],
    "exact": ["count", "total"],
}

# Patterns indicating pre-aggregated sum columns
PREAGG_SUM_PATTERNS = {
    "prefixes": ["sum_", "total_", "gross_", "net_"],
    "suffixes": ["_sum", "_total", "_amount", "_value", "_revenue"],
}


def _is_preaggregated_count_column(col: str) -> bool:
    """Detect if a column is a pre-aggregated count (should use SUM, not COUNT)."""
    col_lower = col.lower()
    for prefix in PREAGG_COUNT_PATTERNS["prefixes"]:
        if col_lower.startswith(prefix):
            return True
    for suffix in PREAGG_COUNT_PATTERNS["suffixes"]:
        if col_lower.endswith(suffix):
            return True
    return col_lower in PREAGG_COUNT_PATTERNS["exact"]


def _find_matching_column(concept: str, model_columns: set[str]) -> tuple[str | None, str]:
    """
    Find a column that semantically matches a measure concept.
    Returns (column_name, aggregation_type).
    
    Uses pure word overlap - no hardcoded domain-specific column names.
    E.g., concept="order_count" matches "number_of_orders" (both have "order")
    """
    concept_lower = concept.lower()
    concept_parts = set(concept_lower.replace("_", " ").split())
    # Remove common aggregation indicator words to get the core concept
    agg_indicators = {"count", "sum", "avg", "total", "number", "num", "of", "the", "a"}
    core_parts = concept_parts - agg_indicators
    # Does the CONCEPT itself ask for a count? ("order_count", "number_of_orders")
    concept_is_count = bool(concept_parts & {"count", "num", "number", "qty", "quantity"})

    if not core_parts:
        return None, "sum"

    # Check each column for semantic match based on word overlap
    for col in model_columns:
        col_lower = col.lower()
        col_parts = set(col_lower.replace("_", " ").split())
        col_core = col_parts - agg_indicators

        # Check for overlap in core concepts (e.g., "order" in both)
        if core_parts & col_core:
            # Determine aggregation from column naming patterns (not hardcoded words)
            if _is_preaggregated_count_column(col):
                return col, "sum"
            # A count concept (order_COUNT) matched to a raw id/key column means COUNT
            # the rows, not SUM the ids — SUM(order_id) is meaningless (gave 4950 for 99
            # orders). Only when the matched column is itself pre-aggregated do we SUM.
            if concept_is_count and (col_lower.endswith("_id") or col_lower == "id"):
                return col, "count"
            # Default to sum for numeric columns
            return col, "sum"

    return None, "sum"


def _find_id_column(model_columns: set[str], entity_hint: str | None = None) -> str | None:
    """Find an ID column to use for counting rows."""
    cols_lower = {c.lower(): c for c in model_columns}
    
    # If we have an entity hint (e.g., "order"), look for order_id first
    if entity_hint:
        hint_id = f"{entity_hint.lower()}_id"
        if hint_id in cols_lower:
            return cols_lower[hint_id]
    
    # Look for common ID patterns
    for pattern in ["_id", "id"]:
        for col_lower, col in cols_lower.items():
            if col_lower.endswith(pattern) or col_lower == "id":
                return col
    
    return None


def _measure_for(
    name: str,
    model_columns: set[str],
    data_types: dict[str, str] | None = None,
    llm_client: Any = None,
    model_name: str = "",
) -> MeasureSpec:
    """
    Create a measure spec by matching measure concept to actual columns.
    
    LLM-first approach for maximum generalizability:
    1. Use LLM to intelligently map measure concept to column (when available)
    2. Fall back to pattern matching only if LLM unavailable
    """
    lowered = name.lower()
    cols_lower = {c.lower(): c for c in model_columns}
    
    # Step 1: Try LLM first - it understands semantics better than patterns
    if llm_client and model_columns:
        try:
            from semanticflow.agents.llm_mapper import sync_map_measure
            # sorted(): set iteration order is randomized per process, so an unsorted column
            # list makes the prompt (and thus the cache key + the run) non-deterministic.
            llm_result = sync_map_measure(
                name, sorted(model_columns), model_name, llm_client, data_types
            )
            if llm_result and llm_result.column in model_columns:
                logger.info(f"LLM mapped '{name}' -> {llm_result.column} ({llm_result.aggregation})")
                return MeasureSpec(name=name, agg=llm_result.aggregation, expr=llm_result.column)
        except Exception as e:
            logger.debug(f"LLM measure mapping failed, falling back to patterns: {e}")
    
    # ===== FALLBACK: Pattern-based matching when LLM unavailable =====
    
    # Fallback Step 1: Exact column match (case-insensitive)
    if lowered in cols_lower:
        actual_col = cols_lower[lowered]
        agg = _infer_aggregation(actual_col, data_types)
        if _is_preaggregated_count_column(actual_col):
            agg = "sum"
        return MeasureSpec(name=name, agg=agg, expr=actual_col)
    
    # Fallback Step 2: Semantic word overlap matching
    matched_col, matched_agg = _find_matching_column(name, model_columns)
    if matched_col:
        return MeasureSpec(name=name, agg=matched_agg, expr=matched_col)
    
    # Fallback Step 3: Count-type measure detection
    count_indicators = {"count", "num", "number", "qty", "quantity"}
    name_words = set(lowered.replace("_", " ").split())
    is_count_measure = bool(name_words & count_indicators)
    
    if is_count_measure:
        entity_words = name_words - count_indicators - {"of", "the", "a"}
        entity_hint = "_".join(entity_words) if entity_words else None
        id_col = _find_id_column(model_columns, entity_hint)
        expr = id_col if id_col else "1"
        return MeasureSpec(name=name, agg="count", expr=expr)
    
    # Fallback Step 4: Broader word overlap
    name_parts = [p for p in lowered.replace("_", " ").split() if len(p) >= 3]
    for part in name_parts:
        for col in model_columns:
            if part in col.lower():
                agg = _infer_aggregation(col, data_types)
                if _is_preaggregated_count_column(col):
                    agg = "sum"
                return MeasureSpec(name=name, agg=agg, expr=col)
    
    # Last resort
    if is_count_measure:
        return MeasureSpec(name=name, agg="count", expr="1")
    
    agg = _infer_aggregation(name, data_types)
    return MeasureSpec(name=name, agg=agg, expr=name)


# Whole-token names that indicate a time dimension. Matched against name TOKENS (split on
# non-alphanumerics), NOT as substrings — the old substring set included "at"/"day"/"year",
# so "status" (contains "at") and "category"/"birthday" were misclassified as time columns,
# which made MetricFlow emit DATE_TRUNC('day', status) and fail to validate.
TIME_NAME_TOKENS = {
    "date", "time", "timestamp", "datetime", "created", "updated",
    "day", "month", "year", "week", "quarter",
}
# Name suffixes that reliably indicate a timestamp column (e.g. created_at, order_ts).
TIME_NAME_SUFFIXES = ("_at", "_date", "_time", "_ts", "_datetime")
# SQL types that are time-based
TIME_TYPES = {"date", "datetime", "timestamp", "timestamptz", "time"}


def _is_time_column(column: str, data_types: dict[str, str] | None) -> bool:
    """Determine if a column is a time dimension based on name or type."""
    import re

    col_lower = column.lower()
    # Data type is authoritative when known: a varchar 'status' is never a time dimension.
    if data_types and data_types.get(column):
        return any(t in data_types[column].lower() for t in TIME_TYPES)

    tokens = {t for t in re.split(r"[^a-z0-9]+", col_lower) if t}
    if tokens & TIME_NAME_TOKENS:
        return True
    return col_lower.endswith(TIME_NAME_SUFFIXES)


def _dimension_for(
    field: str,
    model_columns: set[str],
    data_types: dict[str, str] | None = None,
) -> DimensionSpec | None:
    """Create a dimension spec, inferring type from column name and data type.

    Returns None if the field doesn't exist in the model columns.
    """
    expr = field if field in model_columns else None

    # Check if this is a time dimension
    if _is_time_column(field, data_types):
        # Time dimensions must exist in the model
        if not expr:
            logger.warning(f"Time dimension '{field}' not found in model columns - skipping")
            return None
        # The declared grain is the DATA grain (daily order_date etc.), never the QUERY
        # granularity: declaring at e.g. 'month' makes MetricFlow offer only
        # metric_time__month, so a `{{ TimeDimension('metric_time', 'day') }}` filter
        # cannot resolve. The query-time grain is passed separately by the executor
        # (group-by `metric_time__<grain>` and `--where`), and coarser grains derive
        # from day automatically.
        type_params = {"time_granularity": "day"}
        return DimensionSpec(name=field, type="time", expr=expr, type_params=type_params)
    
    # Handle entity references (e.g., "customer" -> "customer_id")
    if not expr and f"{field}_id" in model_columns:
        expr = f"{field}_id"
    
    # Skip dimensions that don't exist in the model
    if not expr:
        logger.warning(f"Dimension '{field}' not found in model columns - skipping")
        return None
    
    return DimensionSpec(name=field, type="categorical", expr=expr)


def _resolve_measure_dimension_collisions(
    measures: list[MeasureSpec], dimensions: list[DimensionSpec]
) -> list[MeasureSpec]:
    """Rename measures whose name collides with a dimension in the same model.

    Consensus can merge one provider's measure list with another's group-by, so the
    same column (e.g. `customer_id`) lands in both. dbt parse then rejects the model:
    "element `customer_id` is of type DIMENSION, but it is used as types
    [DIMENSION, MEASURE] across the model". Suffix the measure with its aggregation
    (`customer_id` -> `customer_id_count`), keeping `expr` on the original column;
    the metric's `type_params.measure` is built from the (renamed) measure later.
    """
    dimension_names = {d.name for d in dimensions}
    resolved: list[MeasureSpec] = []
    for measure in measures:
        if measure.name in dimension_names:
            new_name = f"{measure.name}_{measure.agg}"
            while new_name in dimension_names:
                new_name = f"{new_name}_measure"
            logger.info(
                f"Measure '{measure.name}' collides with a dimension; renaming to "
                f"'{new_name}' (expr stays on '{measure.expr or measure.name}')"
            )
            measure = MeasureSpec(
                name=new_name,
                agg=measure.agg,
                expr=measure.expr or measure.name,
                description=measure.description,
            )
        resolved.append(measure)
    return resolved


def _time_spine_exists(project_dir: str) -> bool:
    root = Path(project_dir) / "models"
    if not root.exists():
        return False
    return any(root.rglob("metricflow_time_spine.sql"))


def _time_spine_sql(granularity: str) -> tuple[str, str]:
    if granularity == "hour":
        return (
            "date_hour",
            """{{ config(materialized='table') }}

with spine as (
  select * from generate_series(
    timestamp '2010-01-01 00:00:00',
    timestamp '2030-01-01 00:00:00',
    interval 1 hour
  ) as t(date_hour)
)

select date_hour
from spine
""",
        )
    return (
        "date_day",
        """{{ config(materialized='table') }}

with spine as (
  select * from generate_series(
    date '2010-01-01',
    date '2030-01-01',
    interval 1 day
  ) as t(date_day)
)

select date_day
from spine
""",
    )


class DesignerAgent(Agent):
    name: str = "designer"
    llm_client: Any = None  # Optional LLM client for intelligent measure mapping

    async def run(self, msg: AgentMessage) -> AgentMessage:
        semantic_spec = dict(msg.state.get("semantic_spec", {}))
        # Filters must be canonical strings here. Frontier-tier mappers and the repair
        # LLM emit structured dicts ({"field","operator","values"}); every consumer
        # below (filter-column extraction, sanitizer) assumes strings.
        from semanticflow.dbt_integration.filter_utils import coerce_filters_to_strings

        semantic_spec["filters"] = coerce_filters_to_strings(semantic_spec.get("filters"))
        schema_context = msg.task.schema_context
        project_dir = msg.task.dbt_project or "third_party/jaffle_shop_duckdb"
        available_models = schema_context.get("models", [])
        
        # Use source_model from semantic spec if provided by LLM, otherwise fall back to heuristics
        primary_model = semantic_spec.get("source_model")
        if not primary_model or primary_model not in available_models:
            logger.debug(f"source_model '{primary_model}' not in available models {available_models}, using heuristic")
            primary_model = _choose_model(semantic_spec, schema_context)
        else:
            logger.debug(f"Using LLM-specified source_model: {primary_model}")
        
        primary_model_columns = _column_lookup(schema_context, primary_model)

        # Get data types for smarter inference
        data_types = schema_context.get("data_types", {}).get(primary_model, {})
        
        # Get all available columns across all models for validation
        all_columns: set[str] = set()
        for cols in (schema_context.get("columns") or {}).values():
            all_columns.update(cols)
        
        # Detect cross-model scenario: find which model each group_by field belongs to
        group_by_fields = semantic_spec.get("group_by", []) or []
        fields_by_model: dict[str, list[str]] = {}
        for field in group_by_fields:
            sanitized_field = _sanitize_name(field)
            field_model = _find_field_model(sanitized_field, schema_context)
            if field_model:
                fields_by_model.setdefault(field_model, []).append(sanitized_field)
            elif sanitized_field in primary_model_columns:
                fields_by_model.setdefault(primary_model, []).append(sanitized_field)
            else:
                logger.warning(f"Field '{sanitized_field}' not found in any model - skipping")

        # Columns referenced in filters must be emitted as dimensions, or the metric's WHERE
        # clause references a column MetricFlow has not registered (binder error). This
        # includes numeric columns the model placed in base_measures but uses as a row-level
        # threshold (e.g. `customer_lifetime_value > 30`, `coupon_amount > 0`): registering
        # the column as a dimension makes the threshold filter resolvable. (MetricFlow does
        # NOT allow the same NAME to be both a measure and a dimension — any colliding
        # measure is renamed later by _resolve_measure_dimension_collisions, with its expr
        # still pointing at the original column.)
        for col in _extract_filter_columns(semantic_spec.get("filters") or []):
            field_model = _find_field_model(col, schema_context) or (
                primary_model if col in primary_model_columns else None
            )
            if field_model and col not in fields_by_model.get(field_model, []):
                fields_by_model.setdefault(field_model, []).append(col)

        # Determine which models we need semantic models for
        required_models = {primary_model}  # Always include primary (has measures)
        for model_name in fields_by_model:
            if model_name != primary_model:
                required_models.add(model_name)
        
        if len(required_models) > 1:
            logger.info(f"Cross-model query detected: {required_models}")
        
        # An explicit request-level aggregation (e.g. "average") overrides the per-column
        # inference below, which only sees the column name and would default 'amount' to SUM.
        # Prefer the mapper's field; fall back to deterministic NL-verb detection because the
        # mapper LLM often omits the field even when asked.
        requested_agg = normalize_requested_agg(
            semantic_spec.get("aggregation")
        ) or agg_from_request(msg.task.nl_request)

        measures: list[MeasureSpec] = []
        for measure in semantic_spec.get("base_measures", []) or []:
            # Sanitize measure name first (handle SQL syntax like COUNT(orders.order_id))
            sanitized_name, expr_from_parse = _sanitize_measure_name(measure, primary_model_columns)
            # Use LLM-first approach for measure mapping when client available
            measure_spec = _measure_for(
                sanitized_name,
                primary_model_columns,
                data_types,
                llm_client=self.llm_client,
                model_name=primary_model,
            )
            # Honor an explicit requested aggregation over the inferred one. A count over a
            # measure column needs no expr change; for count we keep the mapped expr.
            if requested_agg and measure_spec.agg != requested_agg:
                logger.info(
                    f"Overriding inferred agg '{measure_spec.agg}' with requested "
                    f"'{requested_agg}' for measure '{measure_spec.name}'"
                )
                measure_spec = MeasureSpec(
                    name=measure_spec.name, agg=requested_agg,
                    expr=measure_spec.expr, description=measure_spec.description,
                )
            # If we extracted an expression from the SQL syntax, use it
            if expr_from_parse and measure_spec.expr is None:
                measure_spec = MeasureSpec(
                    name=measure_spec.name,
                    agg=measure_spec.agg,
                    expr=expr_from_parse,
                    description=measure_spec.description
                )
            # Validate: expr must be an actual column in the model or all available columns
            if measure_spec.expr and measure_spec.expr not in primary_model_columns and measure_spec.expr not in all_columns:
                logger.warning(f"Measure '{measure_spec.name}' references non-existent column '{measure_spec.expr}' - skipping")
                continue
            measures.append(measure_spec)

        # Build dimensions only for the primary model
        dimensions: list[DimensionSpec] = []
        for field in fields_by_model.get(primary_model, []):
            dimension = _dimension_for(field, primary_model_columns, data_types)
            if dimension:
                dimensions.append(dimension)

        # If a time grouping is intended (the virtual name "metric_time" or a granularity)
        # but no real time dimension was built, bind it to the model's actual date column
        # rather than falling back to a synthetic current_date (which collapses every row
        # onto "today" and makes time filters/grouping wrong).
        _group_by_fields = semantic_spec.get("group_by", []) or []
        _wants_time = bool(semantic_spec.get("time_granularity")) or any(
            isinstance(f, str) and f.startswith("metric_time") for f in _group_by_fields
        )
        if _wants_time and not any(d.type == "time" for d in dimensions):
            real_time_col = next(
                (c for c in sorted(primary_model_columns) if _is_time_column(c, data_types)), None
            )
            if real_time_col:
                dimensions.append(DimensionSpec(
                    name=real_time_col, type="time", expr=real_time_col,
                    # Data grain, not query grain (see _dimension_for).
                    type_params={"time_granularity": "day"},
                ))
                logger.info(f"Bound metric_time to real time column '{real_time_col}' on {primary_model}")

        # MetricFlow rejects the same name used as both MEASURE and DIMENSION in one
        # model; rename colliding measures by suffixing their aggregation.
        measures = _resolve_measure_dimension_collisions(measures, dimensions)

        sql_models: list[dict[str, Any]] = []
        time_spine_info = None
        if any(dim.type == "time" for dim in dimensions):
            # Time dimensions are always declared at day grain (see _dimension_for),
            # so the spine must be daily too.
            granularity = "day"
            exists = _time_spine_exists(project_dir)
            column_name, sql = _time_spine_sql(granularity)
            time_spine_info = {
                "required": True,
                "exists": exists,
                "time_granularity": granularity,
                "path": "models/semantic/metricflow_time_spine.sql",
                "column": column_name,
            }
            if not exists:
                sql_models.append(
                    {
                        "name": "metricflow_time_spine",
                        "path": "semantic/metricflow_time_spine.sql",
                        "sql": sql,
                        "time_granularity": granularity,
                        "column": column_name,
                    }
                )

        # Use generic entity inference from schema context
        primary_data_types = schema_context.get("data_types", {}).get(primary_model, {})
        primary_entities = _build_entities(primary_model, primary_model_columns, primary_data_types)
        
        # Set default agg_time_dimension if we have a time dimension
        defaults = None
        synthetic_primary_time = False
        time_dims = [dim for dim in dimensions if dim.type == "time"]
        if time_dims:
            defaults = {"agg_time_dimension": time_dims[0].name}
        elif measures:
            # MetricFlow requires agg_time_dimension for measures even if no natural time dimension
            synthetic_primary_time = True
            logger.warning(
                f"No real time dimension for model {primary_model}; injecting synthetic "
                "'ds'=current_date. Any time grouping/filtering will be WRONG (all rows "
                "collapse to today) — results for time-based tasks are not trustworthy."
            )
            dimensions.append(DimensionSpec(
                name="ds",
                type="time",
                expr=_SYNTHETIC_DS_EXPR,  # Synthetic constant date (see constant note)
                type_params={"time_granularity": "day"},
            ))
            defaults = {"agg_time_dimension": "ds"}
            # Also ensure time spine exists
            if not _time_spine_exists(project_dir):
                column_name, sql = _time_spine_sql("day")
                sql_models.append(
                    {
                        "name": "metricflow_time_spine",
                        "path": "semantic/metricflow_time_spine.sql",
                        "sql": sql,
                        "time_granularity": "day",
                        "column": column_name,
                    }
                )
            time_spine_info = {
                "required": True,
                "exists": _time_spine_exists(project_dir),
                "time_granularity": "day",
                "path": "models/semantic/metricflow_time_spine.sql",
                "column": "date_day",
            }

        # Build all semantic models (primary + any secondary models for cross-model queries)
        semantic_models_list: list[SemanticModelSpec] = []
        
        # Primary semantic model (has the measures)
        primary_semantic_model = SemanticModelSpec(
            name=primary_model,
            description=f"Semantic model for {primary_model}. Task: {msg.task.nl_request}",
            model=primary_model,
            entities=primary_entities,
            measures=measures,
            dimensions=dimensions,
            primary_entity=primary_entities[0].name if primary_entities else None,
            defaults=defaults,
        )
        semantic_models_list.append(primary_semantic_model)
        
        # Build secondary semantic models for cross-model dimensions
        for secondary_model in required_models:
            if secondary_model == primary_model:
                continue
            
            secondary_columns = _column_lookup(schema_context, secondary_model)
            secondary_data_types = schema_context.get("data_types", {}).get(secondary_model, {})
            secondary_entities = _build_entities(secondary_model, secondary_columns, secondary_data_types)
            
            # Build dimensions for this secondary model
            secondary_dimensions: list[DimensionSpec] = []
            for field in fields_by_model.get(secondary_model, []):
                dim = _dimension_for(field, secondary_columns, secondary_data_types)
                if dim:
                    secondary_dimensions.append(dim)
            
            # No synthetic 'ds' time dimension (or defaults) on secondary models: they
            # carry no measures, so MetricFlow needs no agg_time_dimension, and stamping
            # the shared synthetic 'ds' on both models triggers a dbt parse error when
            # the models resolve to the same primary entity ("Duplicate dimension +
            # primary entity pairing", e.g. (`order`, `ds`) on both `orders` and
            # `stg_payments`). Likewise primary_entity stays None: the entities list
            # alone carries the join keys, and naming a foreign entity as primary here
            # recreates the shared-primary-entity collision.
            secondary_semantic_model = SemanticModelSpec(
                name=secondary_model,
                description=f"Semantic model for {secondary_model} (dimension source for cross-model query).",
                model=secondary_model,
                entities=secondary_entities,
                measures=[],  # No measures on secondary models
                dimensions=secondary_dimensions,
                primary_entity=None,
                defaults=None,
            )
            semantic_models_list.append(secondary_semantic_model)
            logger.info(f"Created secondary semantic model for '{secondary_model}' with dimensions: {[d.name for d in secondary_dimensions]}")

        # Normalize metric name for consistency
        raw_metric_name = semantic_spec.get("metric_name")
        base_measures = semantic_spec.get("base_measures", [])
        group_by = semantic_spec.get("group_by", [])
        raw_filters = semantic_spec.get("filters", []) or []
        
        if not raw_metric_name:
            # Suggest a name based on measures and dimensions
            raw_metric_name = suggest_metric_name(base_measures, group_by, raw_filters)
        
        metric_name = normalize_metric_name(raw_metric_name)
        logger.debug(f"Metric name: {raw_metric_name} -> {metric_name}", original=raw_metric_name)
        
        # Sanitize filters - convert column references to MetricFlow Jinja (time + categorical)
        primary_entity_name = primary_entities[0].name if primary_entities else None
        dimension_names = {d.name for d in dimensions}
        # Map every (non-time) dimension column to its model's entity prefix, so a filter on
        # a secondary model's column (e.g. `status` on `orders` while the measure is on
        # `stg_payments`) is rendered with the correct entity (`order__status`).
        dim_entity_map: dict[str, str] = {}
        for sm in semantic_models_list:
            ent = sm.primary_entity or next(
                (e.name for e in (sm.entities or []) if getattr(e, "type", None) == "primary"),
                None,
            )
            if not ent:
                continue
            for d in (sm.dimensions or []):
                if getattr(d, "type", None) != "time":
                    dim_entity_map.setdefault(d.name, ent)
        sanitized_filters = []
        for f in raw_filters:
            sanitized_f = _sanitize_filter(
                f, primary_model, primary_entity_name, dimension_names, dim_entity_map
            )
            if sanitized_f:
                sanitized_filters.append(sanitized_f)
        
        # A MetricFlow `simple` metric requires a measure; emitting one without it
        # produces YAML that fails to parse. If no measure survived validation, skip
        # metric emission and flag the degraded design rather than write invalid YAML.
        metrics_payload: list[dict[str, Any]] = []
        design_warnings: list[str] = []
        if synthetic_primary_time:
            design_warnings.append(
                "No real time dimension found; a synthetic 'ds'=current_date was used, "
                "so time grouping/filtering is unreliable for this task."
            )
        if measures:
            metric = MetricSpec(
                name=metric_name,
                description=f"Metric for: {msg.task.nl_request}",
                type="simple",
                type_params={"measure": measures[0].name},
                filters=sanitized_filters or None,
            )
            metrics_payload = [metric.model_dump()]
        else:
            msg_no_measure = f"No valid measure resolved for '{metric_name}'; metric not emitted."
            design_warnings.append(msg_no_measure)
            logger.warning(msg_no_measure)
        design_warning = "; ".join(design_warnings) or None

        state = dict(msg.state)
        state["design_proposals"] = {
            "semantic_models": [sm.model_dump() for sm in semantic_models_list],
            "metrics": metrics_payload,
            "design_warning": design_warning,
            "sql_models": sql_models,
            "time_spine": time_spine_info,
            "semantic_model_exists": semantic_model_exists(project_dir, primary_semantic_model.name),
            # Use metric_name (always defined); `metric` only exists when a measure
            # survived validation, so referencing it here crashed the degraded path.
            "metric_exists": metric_exists(project_dir, metric_name) if measures else False,
        }
        return AgentMessage(task=msg.task, state=state)
