"""Execution-feedback repair: turn a concrete MetricFlow execution failure into one
corrected semantic spec.

The mapper produces a spec; the designer/codegen/executor turn it into a MetricFlow
query. When that query ERRORS (binder error, unknown metric, no valid join path) or
returns EMPTY (a relative-time filter over a static warehouse), a single mapper shot
cannot see *why* — only the executor's stderr/stdout knows. This module feeds that
concrete error back and returns a repaired spec for the caller to RE-EXECUTE.

It runs AFTER the human-in-the-loop answer is incorporated too: HITL fixes the user's
intent, execution-repair fixes the structural realisation of that intent.

Design mirrors ``SemanticMapperAgent.refine_with_feedback``:
  * deterministic first — a hand-written error->edit mapper recognises the common
    MetricFlow error classes and applies a targeted fix WITHOUT an LLM;
  * LLM fallback — when a client is provided and the deterministic mapper did nothing,
    a low-temperature repair prompt (NL request + spec + error + schema columns) asks
    for a corrected spec JSON, parsed defensively;
  * no client + unrecognised error -> spec returned UNCHANGED (no-op).

Retries are NOT here. This module performs ONE repair attempt for ONE error; the
orchestrator owns the bounded retry loop.

Public entry point:
    repair_spec_from_execution(spec, mf_error, schema_context, nl_request="", llm_client=None) -> dict
"""
from __future__ import annotations

import json
import re
from typing import Any

# Default data span for the jaffle_shop_duckdb warehouse (static 2018 data). Used when
# schema_context carries no explicit ``data_span`` and a relative-time filter returns
# empty. Override per-warehouse by putting {"data_span": {"min": "...", "max": "..."}}
# (or {"start"/"end"}) into schema_context.
_DEFAULT_DATA_SPAN = {"min": "2018-01-01", "max": "2018-04-09"}

# Tokens that mark a filter as RELATIVE to "now" (meaningless over a static warehouse).
_RELATIVE_TIME_TOKENS = (
    "current_date",
    "current_timestamp",
    "now()",
    "getdate()",
    "interval",
    "last_",
    "trailing",
    "date_trunc",
    "dateadd",
    "date_sub",
)


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------
def classify_mf_error(mf_error: str) -> str:
    """Map a raw MetricFlow error/stdout string to one of a small set of error classes.

    Returns one of:
      "unknown_metric"        - "does not exactly match any known metrics" (+ Suggestions)
      "fanout_join"           - "No valid join paths exist (fan-out join support is pending)"
                                or "does not match any of the available group-by-items"
      "binder_current_date"   - "Binder Error: ... does not have a column named 'current_date'"
                                (relative-time SQL leaked into the compiled query)
      "empty_relative_time"   - empty result attributable to a relative-time filter
      "none"                  - unrecognised / nothing actionable
    """
    text = (mf_error or "").lower()
    if not text.strip():
        return "none"

    # Unknown metric — usually accompanied by "Suggestions: [...]".
    if "does not exactly match any known metrics" in text or (
        "unknown metric" in text and "suggestion" in text
    ):
        return "unknown_metric"

    # Fan-out join / unjoinable group-by dimension.
    if "no valid join paths exist" in text or "fan-out join" in text:
        return "fanout_join"
    if "does not match any of the available group-by-items" in text:
        return "fanout_join"

    # Binder error from a relative-time expression compiled into SQL against a backend
    # (e.g. duckdb) that has no such bare column.
    if "binder error" in text and "current_date" in text:
        return "binder_current_date"
    if "does not have a column named 'current_date'" in text:
        return "binder_current_date"

    return "none"


def _is_empty_result(mf_error: str) -> bool:
    """Heuristic: did the executor report an EMPTY result rather than an error?

    The caller normally passes the stderr+stdout; an empty query has no error text but
    may carry an explicit empty marker. We also treat a literal empty/whitespace error
    as "empty" so the relative-time->absolute repair can fire on a clean empty run.
    """
    text = (mf_error or "").strip().lower()
    if not text:
        return True
    return any(
        marker in text
        for marker in ("no rows", "empty result", "0 rows", "returned no data", "result is empty")
    )


# ---------------------------------------------------------------------------
# Helpers shared by the deterministic edits
# ---------------------------------------------------------------------------
def _extract_suggestions(mf_error: str) -> list[str]:
    """Pull the suggested metric names out of a 'Suggestions: [a, b, c]' tail."""
    m = re.search(r"suggestions?\s*:?\s*\[([^\]]*)\]", mf_error or "", flags=re.IGNORECASE)
    if not m:
        return []
    inner = m.group(1)
    parts = re.split(r",", inner)
    out: list[str] = []
    for p in parts:
        name = p.strip().strip("'\"` ")
        if name:
            out.append(name)
    return out


def _all_columns(schema_context: dict[str, Any]) -> dict[str, list[str]]:
    return (schema_context or {}).get("columns", {}) or {}


def _data_span(schema_context: dict[str, Any]) -> tuple[str, str]:
    span = (schema_context or {}).get("data_span") or {}
    lo = span.get("min") or span.get("start") or _DEFAULT_DATA_SPAN["min"]
    hi = span.get("max") or span.get("end") or _DEFAULT_DATA_SPAN["max"]
    return str(lo), str(hi)


def _looks_like_time_filter(filt: Any) -> bool:
    f = str(filt or "").lower()
    return any(tok in f for tok in _RELATIVE_TIME_TOKENS)


def _time_column(schema_context: dict[str, Any], source_model: str | None) -> str | None:
    """Pick a physical time column for an absolute-range filter. Prefer the spec's own
    source model, then any model, matching common date/time naming."""
    cols_by_model = _all_columns(schema_context)

    def pick(cols: list[str]) -> str | None:
        for c in cols:
            cl = c.lower()
            if cl.endswith(("_date", "_at", "_ts")) or cl in {"date", "order_date"} or "date" in cl:
                return c
        return None

    if source_model and source_model in cols_by_model:
        hit = pick(cols_by_model[source_model])
        if hit:
            return hit
    for cols in cols_by_model.values():
        hit = pick(cols)
        if hit:
            return hit
    return None


def _model_with_columns(
    schema_context: dict[str, Any], needed: list[str]
) -> str | None:
    """Return the model whose columns cover ALL of ``needed`` (co-located), if any."""
    needed_l = {n.lower() for n in needed if n}
    if not needed_l:
        return None
    for model, cols in _all_columns(schema_context).items():
        cols_l = {c.lower() for c in cols}
        if needed_l <= cols_l:
            return model
    return None


# ---------------------------------------------------------------------------
# Deterministic edits, one per error class
# ---------------------------------------------------------------------------
def _repair_unknown_metric(spec: dict[str, Any], mf_error: str) -> tuple[dict[str, Any], bool]:
    """'does not exactly match any known metrics' + Suggestions -> adopt the suggestion.

    Prefer a suggestion that is a near-match of the current metric_name; else take the
    first suggestion."""
    suggestions = _extract_suggestions(mf_error)
    if not suggestions:
        return spec, False
    current = str(spec.get("metric_name") or "").lower()

    chosen = None
    for s in suggestions:
        sl = s.lower()
        if current and (sl in current or current in sl):
            chosen = s
            break
    if chosen is None:
        chosen = suggestions[0]

    if chosen == spec.get("metric_name"):
        return spec, False
    out = dict(spec)
    out["metric_name"] = chosen
    return out, True


def _repair_fanout_join(
    spec: dict[str, Any], schema_context: dict[str, Any]
) -> tuple[dict[str, Any], bool]:
    """Fan-out join / unjoinable group-by: the measure and the requested group-by
    dimensions live in different models with no valid join path. If a SINGLE model holds
    both the measure columns AND the group-by columns, swap source_model to it
    (co-located, no join needed)."""
    measures = list(spec.get("base_measures") or [])
    group_by = list(spec.get("group_by") or [])
    needed = [c for c in (measures + group_by) if c and c != "metric_time"]
    target = _model_with_columns(schema_context, needed)
    if not target or target == spec.get("source_model"):
        return spec, False
    out = dict(spec)
    out["source_model"] = target
    return out, True


def _repair_relative_time(
    spec: dict[str, Any], schema_context: dict[str, Any]
) -> tuple[dict[str, Any], bool]:
    """Convert relative-time filters (current_date - interval, last_N_days, ...) into an
    absolute date range that lies within the warehouse's data span. Used both for the
    binder-error case (relative SQL the backend rejects) and the empty-result-over-static-
    data case."""
    filters = list(spec.get("filters") or [])
    if not filters:
        return spec, False
    lo, hi = _data_span(schema_context)
    time_col = _time_column(schema_context, spec.get("source_model"))

    new_filters: list[str] = []
    changed = False
    has_time = False
    for f in filters:
        if _looks_like_time_filter(f):
            changed = True
            has_time = True
            continue  # drop the relative clause; replace with an absolute range below
        new_filters.append(f)

    if not changed:
        return spec, False

    # Express the absolute range against metric_time when we have no concrete column, or
    # against the discovered physical time column. The executor's _filters_to_where maps
    # bare `col >= 'date'` onto the right TimeDimension.
    if has_time:
        col = time_col or "metric_time"
        new_filters.append(f"{col} >= '{lo}'")
        new_filters.append(f"{col} <= '{hi}'")

    out = dict(spec)
    out["filters"] = new_filters
    return out, True


# ---------------------------------------------------------------------------
# LLM fallback
# ---------------------------------------------------------------------------
def _extract_json(raw: str) -> dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise ValueError("LLM did not return JSON")
        return json.loads(match.group(0))


def _build_repair_prompt(
    spec: dict[str, Any],
    mf_error: str,
    schema_context: dict[str, Any],
    nl_request: str,
) -> str:
    cols_by_model = _all_columns(schema_context)
    cols_lines = "\n".join(f"  {m}: {c}" for m, c in cols_by_model.items()) or "  (none)"
    models = ", ".join((schema_context or {}).get("models", []) or []) or "(none)"
    lo, hi = _data_span(schema_context)
    return (
        "A MetricFlow query built from the semantic spec below FAILED or returned EMPTY. "
        "Produce a CORRECTED semantic spec as JSON that fixes the specific error.\n\n"
        f"Original natural-language request:\n{nl_request or '(not provided)'}\n\n"
        f"Current semantic spec:\n{json.dumps(spec, default=str)}\n\n"
        f"MetricFlow error / empty-result detail:\n{mf_error}\n\n"
        f"Available models: {models}\n"
        "Available columns by model:\n"
        f"{cols_lines}\n\n"
        f"Warehouse data span: {lo} .. {hi} (data is STATIC; relative-time filters such as "
        "'current_date - interval' return nothing — use an ABSOLUTE range inside this span).\n\n"
        "RULES:\n"
        "- metric_name: if the error lists Suggestions, use the closest suggested name.\n"
        "- base_measures / group_by: only REAL column names from the schema above, no "
        "table prefixes, no SQL functions.\n"
        "- source_model: a model that actually contains the measure and group-by columns "
        "(avoid cross-model fan-out joins).\n"
        "- filters: convert any relative-time filter to an absolute date range within the "
        "data span.\n"
        "Return ONLY the corrected JSON spec with the same fields as the current spec "
        "(metric_name, source_model, base_measures, aggregation, group_by, "
        "time_granularity, filters)."
    )


def _repair_with_llm(
    spec: dict[str, Any],
    mf_error: str,
    schema_context: dict[str, Any],
    nl_request: str,
    llm_client: Any,
) -> dict[str, Any]:
    prompt = _build_repair_prompt(spec, mf_error, schema_context, nl_request)
    try:
        raw = llm_client.complete(
            "You repair MetricFlow semantic specs from concrete execution errors. "
            "Return only JSON.",
            prompt,
            temperature=0.0,
            json_mode=True,
        )
    except Exception:
        return spec
    try:
        payload = _extract_json(raw)
    except Exception:
        return spec
    if not isinstance(payload, dict) or not payload:
        return spec

    # Merge defensively over the original spec so the LLM dropping a field never erases
    # it; only overwrite with non-empty corrected values.
    out = dict(spec)
    for key in (
        "metric_name",
        "source_model",
        "base_measures",
        "aggregation",
        "group_by",
        "time_granularity",
        "filters",
    ):
        if key not in payload:
            continue
        value = payload[key]
        if value is None:
            continue
        if isinstance(value, (list, str)) and len(value) == 0:
            # allow an explicit empty filters/group_by clearing only for those slots
            if key in ("filters", "group_by"):
                out[key] = list(value) if isinstance(value, list) else []
            continue
        if key == "filters":
            # Repair LLMs (frontier-tier especially) return structured filter dicts;
            # downstream consumers require canonical strings.
            from semanticflow.dbt_integration.filter_utils import coerce_filters_to_strings

            value = coerce_filters_to_strings(value)
            if not value:
                continue
        out[key] = value
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def repair_spec_from_execution(
    spec: dict[str, Any],
    mf_error: str,
    schema_context: dict[str, Any],
    nl_request: str = "",
    llm_client: Any | None = None,
) -> dict[str, Any]:
    """Repair a semantic spec from ONE concrete MetricFlow execution failure.

    Args:
        spec: the current semantic spec dict (metric_name, base_measures, aggregation,
            group_by, filters, time_granularity, source_model, ...).
        mf_error: the MetricFlow error text (stderr+stdout). Pass an empty/whitespace
            string to signal a clean EMPTY result.
        schema_context: {"models": [...], "columns": {model: [cols]},
            optional "data_span": {"min"/"start": ..., "max"/"end": ...}}.
        nl_request: the original natural-language request (used only by the LLM path).
        llm_client: optional BaseLLMClient-style client (must expose
            ``complete(system, user, temperature, json_mode)``). When None, only the
            deterministic mapper runs.

    Returns:
        A repaired spec dict, or the spec UNCHANGED if no repair applies.

    Performs exactly ONE repair attempt. Retries belong to the caller.
    """
    if not isinstance(spec, dict):
        return spec
    schema_context = schema_context or {}

    error_class = classify_mf_error(mf_error)

    # Deterministic mapper first (mirrors refine_with_feedback's deterministic-first
    # structure). Each branch is a targeted, schema-grounded edit.
    repaired = spec
    changed = False

    if error_class == "unknown_metric":
        repaired, changed = _repair_unknown_metric(spec, mf_error)
    elif error_class == "fanout_join":
        repaired, changed = _repair_fanout_join(spec, schema_context)
    elif error_class == "binder_current_date":
        repaired, changed = _repair_relative_time(spec, schema_context)

    # Empty result attributable to a relative-time filter -> absolute range. This is not
    # an "error class" (no error text), so it's checked independently.
    if not changed and error_class == "none" and _is_empty_result(mf_error):
        if any(_looks_like_time_filter(f) for f in (spec.get("filters") or [])):
            repaired, changed = _repair_relative_time(spec, schema_context)

    if changed:
        return repaired

    # LLM fallback only when the deterministic mapper found nothing AND a client exists.
    if llm_client is not None:
        return _repair_with_llm(spec, mf_error, schema_context, nl_request, llm_client)

    # No client, unrecognised error -> no-op.
    return spec


def clarifying_questions_from_error(mf_error: str) -> list[str]:
    """Deterministic (zero-LLM) clarifying question(s) for a query that STILL fails or
    returns empty after all repair loops — the post-execution HITL touchpoint.

    Pre-execution HITL asks about intent the mapper was unsure of; this asks about the
    one thing only execution could reveal (the concrete failure). Each question targets
    a single spec slot so a slot-scoped human can answer it."""
    error_class = classify_mf_error(mf_error)
    if error_class == "unknown_metric":
        return ["Which measure should be used to calculate this result?"]
    if error_class == "fanout_join":
        return [
            "Which dimension should the results be grouped by, given the measure's "
            "source table?"
        ]
    if error_class == "binder_current_date" or _is_empty_result(mf_error):
        return [
            "The query returned no rows. What absolute date range should the results "
            "cover?"
        ]
    # Unrecognised hard error: confirm the two slots that most often sink execution.
    return [
        "Which measure should be used to calculate this result?",
        "What date range should the results cover, if any?",
    ]
