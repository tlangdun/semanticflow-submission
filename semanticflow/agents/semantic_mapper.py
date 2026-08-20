from __future__ import annotations

import asyncio
import json
import math
import re
import time
from itertools import combinations
from typing import Any, Literal

from pydantic import BaseModel, Field

from semanticflow.agents.types import Agent, AgentMessage
from semanticflow.config import Settings
from semanticflow.llm import BaseLLMClient, heuristic_semantic_spec


class SemanticSpec(BaseModel):
    metric_name: str
    base_measures: list[str] = Field(default_factory=list)
    group_by: list[str] = Field(default_factory=list)
    time_granularity: str | None = None
    filters: list[str] = Field(default_factory=list)
    definition_notes: str = ""
    confidence: float | None = None
    uncertainty_score: float = 0.0
    ambiguity_points: list[str] = Field(default_factory=list)
    clarifying_questions: list[str] = Field(default_factory=list)
    order_by: list[dict[str, str]] | None = None
    limit: int | None = None


class SemanticParsingResult(BaseModel):
    semantic_spec: dict[str, Any]
    model_outputs: dict[str, dict[str, Any]]
    model_confidences: dict[str, float]
    epistemic_uncertainty: float
    semantic_agreement_score: float
    ambiguity: Literal["low", "high"]
    ambiguity_reason: str
    clarifying_questions: list[str]
    consensus_method: str


def _flatten_tokens(spec: dict[str, Any]) -> list[str]:
    tokens: list[str] = []
    metric = spec.get("metric_name")
    if metric:
        tokens.append(f"metric:{metric}")
    for measure in spec.get("base_measures", []) or []:
        tokens.append(f"measure:{measure}")
    for dim in spec.get("group_by", []) or []:
        tokens.append(f"dimension:{dim}")
    for filt in spec.get("filters", []) or []:
        tokens.append(f"filter:{filt}")
    return tokens


def _build_vocab(specs: list[dict[str, Any]], schema_context: dict[str, Any] | None = None) -> list[str]:
    vocab: set[str] = set()
    for spec in specs:
        vocab.update(_flatten_tokens(spec))
    if schema_context:
        for columns in (schema_context.get("columns") or {}).values():
            for col in columns:
                vocab.add(f"dimension:{col}")
                # Dynamically generate measure vocabulary from column names
                # No hardcoded measure names - infer from schema
                col_lower = col.lower()
                if any(p in col_lower for p in ["count", "num", "number", "total", "sum", "amount", "value", "revenue"]):
                    vocab.add(f"measure:{col}")
                # Also add generic measure patterns based on column
                vocab.add(f"measure:{col}_sum")
                vocab.add(f"measure:{col}_count")
    if not vocab:
        vocab.add("metric:unknown")
    return sorted(vocab)


def _spec_to_distribution(spec: dict[str, Any], vocab: list[str]) -> list[float]:
    counts = {token: 0.0 for token in vocab}
    for token in _flatten_tokens(spec):
        if token in counts:
            counts[token] += 1.0
    total = sum(counts.values())
    if total == 0:
        return [1.0 / len(vocab)] * len(vocab)
    return [counts[token] / total for token in vocab]


def _kl_divergence(p: list[float], q: list[float]) -> float:
    total = 0.0
    for pi, qi in zip(p, q):
        if pi == 0:
            continue
        total += pi * math.log(pi / max(qi, 1e-12))
    return total


def _jsd(p: list[float], q: list[float]) -> float:
    m = [(pi + qi) / 2 for pi, qi in zip(p, q)]
    return 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)


def _average_jsd(specs: list[dict[str, Any]], schema_context: dict[str, Any] | None = None) -> float:
    if len(specs) < 2:
        return 0.0
    vocab = _build_vocab(specs, schema_context)
    distributions = [_spec_to_distribution(spec, vocab) for spec in specs]
    pairs = list(combinations(range(len(distributions)), 2))
    if not pairs:
        return 0.0
    values = []
    for i, j in pairs:
        values.append(_jsd(distributions[i], distributions[j]))
    return sum(values) / len(values)


def _spec_fieldsets(spec: dict[str, Any]) -> dict[str, set[str]]:
    def norm(items: Any) -> set[str]:
        if not items:
            return set()
        if not isinstance(items, (list, tuple, set)):
            items = [items]
        return {str(i).strip().lower() for i in items if str(i).strip()}

    gran = spec.get("time_granularity")
    return {
        "measures": norm(spec.get("base_measures")),
        "group_by": norm(spec.get("group_by")),
        "filters": norm(spec.get("filters")),
        "granularity": {str(gran).strip().lower()} if gran else set(),
    }


def _jaccard_distance(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    return 1.0 - (len(a & b) / len(union)) if union else 0.0


def structural_disagreement(specs: list[dict[str, Any]]) -> float:
    """Mean pairwise distance over semantic spec FIELDS (measures, group-by, filters,
    granularity), each via Jaccard distance.

    Unlike bag-of-tokens JSD, this is field-aware: a differing critical filter counts
    as much as a differing measure, and lexical noise outside these slots is ignored.
    Returned in [0, 1]; 0 = identical structure, 1 = fully disjoint."""
    if len(specs) < 2:
        return 0.0
    fields = [_spec_fieldsets(s) for s in specs]
    keys = ("measures", "group_by", "filters", "granularity")
    pair_scores = []
    for i, j in combinations(range(len(fields)), 2):
        per_field = [_jaccard_distance(fields[i][k], fields[j][k]) for k in keys]
        pair_scores.append(sum(per_field) / len(per_field))
    return sum(pair_scores) / len(pair_scores) if pair_scores else 0.0


def _normalized_agreement(epistemic_uncertainty: float) -> float:
    max_jsd = math.log(2)
    normalized = min(epistemic_uncertainty / max_jsd, 1.0)
    return max(0.0, 1.0 - normalized)


def _vote_single_value(values: list[str | None], confidences: list[float]) -> str | None:
    scores: dict[str, float] = {}
    for value, weight in zip(values, confidences):
        if not value:
            continue
        scores[value] = scores.get(value, 0.0) + weight
    if not scores:
        return None
    return max(scores.items(), key=lambda item: item[1])[0]


def _vote_list(values: list[list[str]], confidences: list[float]) -> list[str]:
    scores: dict[str, float] = {}
    for items, weight in zip(values, confidences):
        for item in items:
            scores[item] = scores.get(item, 0.0) + weight
    if not scores:
        return []
    max_score = max(scores.values())
    threshold = max_score * 0.7
    selected = [item for item, score in scores.items() if score >= threshold]
    return sorted(set(selected))


def _looks_like_time_col(column: str) -> bool:
    """Heuristic time-column check by whole tokens/suffix (avoids 'at' in 'status')."""
    c = (column or "").lower()
    tokens = {t for t in re.split(r"[^a-z0-9]+", c) if t}
    return bool(tokens & {"date", "time", "day", "month", "year", "week", "quarter"}) or \
        c.endswith(("_at", "_date", "_time", "_ts"))


def consensus_vote(specs: list[dict[str, Any]], confidences: list[float]) -> dict[str, Any]:
    metric_name = _vote_single_value([spec.get("metric_name") for spec in specs], confidences)
    aggregation = _vote_single_value([spec.get("aggregation") for spec in specs], confidences)
    base_measures = _vote_list([spec.get("base_measures", []) or [] for spec in specs], confidences)
    group_by = _vote_list([spec.get("group_by", []) or [] for spec in specs], confidences)
    time_granularity = _vote_single_value([spec.get("time_granularity") for spec in specs], confidences)
    top_index = max(range(len(confidences)), key=lambda idx: confidences[idx]) if confidences else 0
    # Filters are ANDed at query time, so unioning them via _vote_list piles up
    # semantically-equivalent variants from different providers/samples (raw-SQL vs
    # Jinja, '2018-02' vs date literal) into one over-restrictive conjunction. Take the
    # top-confidence provider's filter list instead, like order_by/limit below.
    filters = (specs[top_index].get("filters", []) or []) if specs else []
    order_by = specs[top_index].get("order_by") if specs else None
    limit = specs[top_index].get("limit") if specs else None

    ambiguity_points: list[str] = []
    clarifying_questions: list[str] = []
    for spec in specs:
        ambiguity_points.extend(spec.get("ambiguity_points", []) or [])
        clarifying_questions.extend(spec.get("clarifying_questions", []) or [])

    return {
        "metric_name": metric_name or "metric",
        "aggregation": aggregation,
        "base_measures": base_measures,
        "group_by": group_by,
        "time_granularity": time_granularity,
        "filters": filters,
        "order_by": order_by,
        "limit": limit,
        "definition_notes": "",
        "ambiguity_points": sorted(set(ambiguity_points)),
        "clarifying_questions": sorted(set(clarifying_questions)),
    }


def _normalize_slot_value(value: Any) -> str | tuple[str, ...]:
    """Canonical form of a slot value for equality / emptiness checks.

    None -> "" ; lists/tuples/sets -> tuple(sorted of non-empty lowercased stripped
    strings) ; scalars -> lowercased stripped str. "Empty" is "" or ()."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return tuple(sorted(str(x).lower().strip() for x in value if str(x).strip()))
    return str(value).lower().strip()


def _slot_is_empty(value: Any) -> bool:
    norm = _normalize_slot_value(value)
    return norm == "" or norm == ()


# Slots whose already-correct consensus value we protect from HITL overwrite.
GUARDED_SLOTS = ["base_measures", "group_by", "filters", "aggregation", "time_granularity"]


def guard_agreed_slots(
    consensus_spec: dict,
    refined_spec: dict,
    model_outputs: dict | None,
    clarified_slots: set[str] | None = None,
) -> tuple[dict, list[str]]:
    """Revert HITL refinements that OVERWRITE a slot all providers already agreed on.

    HITL clarification refines the consensus spec, but it sometimes harms an
    already-correct task by overwriting a slot every ensemble provider agreed on (and was
    right about). It helps when it fills an empty slot or resolves a slot the providers
    disagreed on. So for each guarded slot we revert the refined value back to the
    consensus value IFF it is a "protected overwrite": the consensus value is NON-EMPTY
    AND all per-provider specs agree on it (identical after normalization). Otherwise the
    refined value is kept.

    No-op (returns refined_spec unchanged) when model_outputs is missing or has < 2
    providers, preserving single-provider / baseline behavior.

    Returns (guarded_spec, reverted_slots)."""
    guarded = dict(refined_spec)
    reverted: list[str] = []
    if not model_outputs or len(model_outputs) < 2:
        return guarded, reverted

    provider_specs = list(model_outputs.values())
    for slot in GUARDED_SLOTS:
        # The human's explicit clarification of a slot is ground truth and overrides the
        # providers' agreement: never revert a slot the human directly answered (otherwise a
        # unanimous-but-wrong default, e.g. measure='amount' for "show me the trend", clobbers
        # the human's "number of orders" correction).
        if clarified_slots and slot in clarified_slots:
            continue
        consensus_val = consensus_spec.get(slot)
        # Only protect a NON-EMPTY consensus slot.
        if _slot_is_empty(consensus_val):
            continue
        # All providers must agree on the slot (identical after normalization).
        per_provider = [_normalize_slot_value(s.get(slot)) for s in provider_specs]
        if len(set(per_provider)) != 1:
            continue
        # Providers agreed; revert only if the refinement actually changed the slot.
        if _normalize_slot_value(guarded.get(slot)) != _normalize_slot_value(consensus_val):
            guarded[slot] = consensus_val
            reverted.append(slot)
    return guarded, reverted


class SemanticMapperAgent(Agent):
    name: str = "semantic_mapper"
    settings: Settings
    llm_clients: dict[str, BaseLLMClient]

    def _contains_json_object(self, raw: str) -> bool:
        return re.search(r"\{.*\}", raw, flags=re.DOTALL) is not None

    async def _complete_with_timeout(
        self,
        client: BaseLLMClient,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None,
        json_mode: bool,
        timeout_override: float | None = None,
    ) -> str:
        call = asyncio.to_thread(client.complete, system_prompt, user_prompt, temperature, json_mode)
        timeout = timeout_override if timeout_override is not None else self.settings.llm_timeout_sec
        if timeout and timeout > 0:
            return await asyncio.wait_for(call, timeout=timeout)
        return await call

    def _extract_json(self, raw: str) -> dict[str, Any]:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if not match:
                raise ValueError("LLM did not return JSON")
            return json.loads(match.group(0))

    def _normalize_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            items = value
        else:
            items = [value]
        normalized = []
        for item in items:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                normalized.append(text)
        return normalized

    def _normalize_time_granularity(self, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, list):
            value = value[0] if value else None
        if value is None:
            return None
        key = str(value).strip().lower()
        if not key:
            return None
        aliases = {
            "daily": "day",
            "day": "day",
            "d": "day",
            "weekly": "week",
            "week": "week",
            "w": "week",
            "monthly": "month",
            "month": "month",
            "m": "month",
            "quarterly": "quarter",
            "quarter": "quarter",
            "q": "quarter",
            "yearly": "year",
            "year": "year",
            "y": "year",
            "hourly": "hour",
            "hour": "hour",
            "h": "hour",
            "minute": "minute",
            "min": "minute",
        }
        return aliases.get(key, key)

    def _normalize_direction(self, value: Any) -> str:
        if value is None:
            return "asc"
        key = str(value).strip().lower()
        if key in {"desc", "descending", "down", "-1"}:
            return "desc"
        return "asc"

    def _normalize_filters(self, value: Any) -> list[str]:
        # Shared coercion: handles plural "values" and `between` expansion, which the
        # old inline dict branch did not (frontier mappers emit exactly that shape and
        # got "col between" garbage strings here).
        from semanticflow.dbt_integration.filter_utils import coerce_filters_to_strings

        return coerce_filters_to_strings(value)

    def _normalize_order_by(self, value: Any) -> list[dict[str, str]] | None:
        if value is None:
            return None
        if isinstance(value, list):
            items = value
        else:
            items = [value]
        normalized: list[dict[str, str]] = []
        for item in items:
            if isinstance(item, dict):
                field = item.get("field") or item.get("column") or item.get("dimension")
                if not field:
                    continue
                direction = self._normalize_direction(item.get("direction") or item.get("dir"))
                normalized.append({"field": str(field), "direction": direction})
                continue
            text = str(item).strip()
            if not text:
                continue
            parts = text.split()
            field = parts[0]
            direction = self._normalize_direction(parts[1] if len(parts) > 1 else None)
            normalized.append({"field": field, "direction": direction})
        return normalized or None

    def _normalize_spec_payload(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        normalized = dict(payload)
        metric_name = (
            normalized.get("metric_name")
            or normalized.get("metric")
            or normalized.get("name")
        )
        if metric_name is not None:
            normalized["metric_name"] = str(metric_name).strip()
        aggregation = normalized.get("aggregation") or normalized.get("agg")
        normalized["aggregation"] = (
            str(aggregation).strip().lower() if aggregation else None
        )
        base_measures = normalized.get("base_measures")
        if base_measures is None:
            base_measures = (
                normalized.get("base_measure")
                or normalized.get("measure")
                or normalized.get("measures")
            )
        normalized["base_measures"] = self._normalize_list(base_measures)
        group_by = normalized.get("group_by")
        if group_by is None:
            group_by = normalized.get("dimensions")
        normalized["group_by"] = self._normalize_list(group_by)
        filters = normalized.get("filters")
        if filters is None:
            filters = normalized.get("filter")
        normalized["filters"] = self._normalize_filters(filters)
        normalized["time_granularity"] = self._normalize_time_granularity(
            normalized.get("time_granularity")
        )
        normalized["definition_notes"] = str(normalized.get("definition_notes") or "")
        confidence = normalized.get("confidence")
        if confidence is not None:
            try:
                normalized["confidence"] = float(confidence)
            except (TypeError, ValueError):
                normalized["confidence"] = None
        uncertainty = normalized.get("uncertainty_score")
        if uncertainty is not None:
            try:
                normalized["uncertainty_score"] = float(uncertainty)
            except (TypeError, ValueError):
                normalized["uncertainty_score"] = 0.0
        normalized["ambiguity_points"] = self._normalize_list(normalized.get("ambiguity_points"))
        normalized["clarifying_questions"] = self._normalize_list(
            normalized.get("clarifying_questions")
        )
        normalized["order_by"] = self._normalize_order_by(normalized.get("order_by"))
        limit = normalized.get("limit")
        if limit is not None and limit != "":
            try:
                normalized["limit"] = int(limit)
            except (TypeError, ValueError):
                normalized["limit"] = None
        return normalized

    def _repair_output(self, client: BaseLLMClient, raw: str) -> str:
        repair_prompt = (
            "Fix the following response so it is valid JSON matching the schema exactly. "
            "Return only JSON.\n\n"
            f"Response:\n{raw}\n\n"
            "Schema fields: metric_name (string), aggregation (string), base_measures (list), "
            "group_by (list), "
            "time_granularity (string or null), filters (list), definition_notes (string), "
            "confidence (0-1), uncertainty_score (0-1), ambiguity_points (list), "
            "clarifying_questions (list), order_by (list or null), limit (int or null)."
        )
        return client.complete(
            "You repair JSON for semantic specs.",
            repair_prompt,
            temperature=0.0,
            json_mode=True,
        )

    async def _call_model(
        self, provider: str, client: BaseLLMClient, msg: AgentMessage
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        system_prompt = (
            "You map natural language analytics requests into a JSON semantic spec. "
            "Return only JSON matching the schema fields."
        )
        # Extract available columns for explicit listing
        schema_ctx = msg.task.schema_context
        columns_info = schema_ctx.get("columns", {})
        columns_list = []
        for model_name, cols in columns_info.items():
            columns_list.append(f"  {model_name}: {cols}")
        columns_str = "\n".join(columns_list) if columns_list else "  (no columns info)"
        
        # Get available models list
        models_list = schema_ctx.get("models", [])
        models_str = ", ".join(models_list) if models_list else "(no models)"
        
        user_prompt = (
            "Request:\n"
            f"{msg.task.nl_request}\n\n"
            "Schema context:\n"
            f"{schema_ctx}\n\n"
            f"AVAILABLE MODELS: {models_str}\n\n"
            "AVAILABLE COLUMNS BY MODEL:\n"
            f"{columns_str}\n\n"
            "Fields: metric_name (string), source_model (string), base_measures (list), "
            "aggregation (string), group_by (list), "
            "time_granularity (string or null), filters (list), definition_notes (string), "
            "confidence (0-1), uncertainty_score (0-1), ambiguity_points (list), "
            "clarifying_questions (list), order_by (list or null), limit (int or null).\n\n"
            "CRITICAL RULES (MUST FOLLOW):\n"
            "0. aggregation: derive from the REQUEST VERB, not the column name. One of: "
            "sum, average, count, count_distinct, min, max, median.\n"
            "   'average'/'avg'/'mean' -> average;  'how many'/'number of'/'count' -> count;\n"
            "   'total'/'sum'/'how much' -> sum;  'distinct'/'unique' -> count_distinct;\n"
            "   'highest'/'max' -> max;  'lowest'/'min' -> min;  'median' -> median.\n"
            "   Example: 'average order value' -> aggregation='average' even though the "
            "column is 'amount' (do NOT default to sum).\n"
            "1. source_model: MUST be one of the available models above.\n"
            "   Choose the model that contains the column(s) you need to aggregate.\n"
            "   Example: If aggregating 'amount', and 'amount' is in 'orders', use 'orders'.\n"
            "2. base_measures: MUST be actual column names that exist in source_model.\n"
            "   Use the exact column name that will be aggregated.\n"
            "   CORRECT: ['amount'], ['customer_lifetime_value'], ['order_id']\n"
            "   WRONG: ['revenue_sum'], ['total_revenue'] (these don't exist as columns)\n"
            "   WRONG: ['COUNT(order_id)'], ['SUM(amount)'] (no SQL syntax)\n"
            "3. group_by: ONLY column names from schema, NO table prefixes.\n"
            "   CORRECT: ['order_date'], ['customer_id']\n"
            "   WRONG: ['orders.order_date'], ['customers.customer_id']\n"
            "4. filters: Use MetricFlow Jinja syntax for time filters.\n"
            "   CORRECT: \"{{ TimeDimension('metric_time', 'day') }} >= current_date - interval '30 days'\"\n"
            "   Non-time filters: \"status = 'completed'\" (OK as-is)\n\n"
            "Return valid JSON only."
        )
        debug: dict[str, Any] = {
            "provider": provider,
            "client_type": type(client).__name__,
            "model": getattr(client, "_model", None),
            "status": "ok",
            "repaired": False,
            "attempts": 0,
            "used_heuristic": False,
            "error": None,
        }
        deadline: float | None = None
        if self.settings.llm_timeout_sec and self.settings.llm_timeout_sec > 0:
            deadline = time.monotonic() + self.settings.llm_timeout_sec

        def _time_left() -> float | None:
            if deadline is None:
                return None
            remaining = deadline - time.monotonic()
            return max(0.0, remaining)

        try:
            time_left = _time_left()
            if time_left is not None and time_left <= 0:
                raise TimeoutError
            raw = await self._complete_with_timeout(
                client, system_prompt, user_prompt, 0.7, True, timeout_override=time_left
            )
        except TimeoutError:
            debug.update(
                {
                    "status": "timeout",
                    "used_heuristic": True,
                    "error": "timeout",
                }
            )
            heuristic = heuristic_semantic_spec(msg.task.nl_request, msg.task.schema_context)
            return provider, heuristic, debug
        except Exception as exc:  # pragma: no cover - network/API dependent
            debug.update(
                {
                    "status": "error",
                    "used_heuristic": True,
                    "error": str(exc),
                }
            )
            heuristic = heuristic_semantic_spec(msg.task.nl_request, msg.task.schema_context)
            return provider, heuristic, debug
        if self.settings.log_llm_raw:
            debug["raw_response"] = raw
        attempts = 0
        payload: dict[str, Any] | None = None
        try:
            payload = self._extract_json(raw)
        except Exception:
            payload = None
        if payload is None:
            if not self._contains_json_object(raw):
                debug.update(
                    {
                        "status": "fallback",
                        "used_heuristic": True,
                        "error": "no_json_in_response",
                    }
                )
                heuristic = heuristic_semantic_spec(msg.task.nl_request, msg.task.schema_context)
                return provider, heuristic, debug
            attempts = 0
            while attempts < self.settings.max_llm_repair_attempts:
                attempts += 1
                debug["attempts"] = attempts
                try:
                    repair_call = asyncio.to_thread(self._repair_output, client, raw)
                    time_left = _time_left()
                    if time_left is not None and time_left <= 0:
                        raise TimeoutError
                    repair_timeout = self.settings.llm_repair_timeout_sec
                    if repair_timeout and repair_timeout > 0:
                        if time_left is not None:
                            repair_timeout = min(repair_timeout, time_left)
                    else:
                        repair_timeout = time_left
                    if repair_timeout and repair_timeout > 0:
                        raw = await asyncio.wait_for(repair_call, timeout=repair_timeout)
                    else:
                        raw = await repair_call
                    payload = self._extract_json(raw)
                    break
                except TimeoutError:
                    debug.update(
                        {
                            "status": "fallback",
                            "used_heuristic": True,
                            "error": "repair_timeout",
                        }
                    )
                    heuristic = heuristic_semantic_spec(msg.task.nl_request, msg.task.schema_context)
                    return provider, heuristic, debug
                except Exception:
                    payload = None
            if payload is None:
                debug.update(
                    {
                        "status": "fallback",
                        "used_heuristic": True,
                        "error": "json_repair_failed",
                    }
                )
                heuristic = heuristic_semantic_spec(msg.task.nl_request, msg.task.schema_context)
                return provider, heuristic, debug
        payload = self._normalize_spec_payload(payload)
        try:
            spec = SemanticSpec.model_validate(payload)
            debug["attempts"] = attempts
            debug["repaired"] = attempts > 0
            return provider, spec.model_dump(), debug
        except Exception as exc:
            debug.update(
                {
                    "status": "fallback",
                    "used_heuristic": True,
                    "error": "schema_validation_failed",
                }
            )
            if self.settings.log_llm_raw:
                debug["validation_error"] = str(exc)
            heuristic = heuristic_semantic_spec(msg.task.nl_request, msg.task.schema_context)
            return provider, heuristic, debug

    async def run(self, msg: AgentMessage) -> AgentMessage:
        # Iterate the clients actually configured on this agent rather than a fixed
        # provider list, so the single-provider baseline mapper uses its real client.
        model_specs = dict(self.llm_clients)
        # k samples per provider for ALEATORIC (within-model) uncertainty. k=1 reproduces
        # the prior single-call behaviour exactly (per-provider aleatoric is then 0).
        k = max(1, getattr(self.settings, "self_consistency_k", 1))
        ordered_providers = [p for p, c in model_specs.items() if c is not None]
        tasks = [
            self._call_model(provider, model_specs[provider], msg)
            for provider in ordered_providers
            for _ in range(k)
        ]
        if not tasks:
            heuristic = heuristic_semantic_spec(msg.task.nl_request, msg.task.schema_context)
            tasks = [asyncio.sleep(0, result=("mock", heuristic, {"provider": "mock", "used_heuristic": True}))]

        results = await asyncio.gather(*tasks)

        # Group the flat result list back by provider.
        from collections import defaultdict
        samples_by_provider: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
        for provider, output, debug in results:
            samples_by_provider[provider].append((output, debug))

        # Per provider: a representative spec (consensus of its k samples) and its
        # within-provider disagreement (aleatoric component).
        model_outputs: dict[str, dict[str, Any]] = {}
        debug_map: dict[str, dict[str, Any]] = {}
        provider_aleatoric: dict[str, float | None] = {}
        for provider, samples in samples_by_provider.items():
            outs = [o for o, _ in samples]
            debs = [d for _, d in samples]
            if len(outs) == 1:
                rep = outs[0]
            else:
                rep = consensus_vote(outs, [float(o.get("confidence") or 0.5) for o in outs])
            model_outputs[provider] = rep
            # Within-model (aleatoric) disagreement is only meaningful if the k samples
            # came from the stochastic model. If every sample fell back to the
            # DETERMINISTIC heuristic (timeout/parse failure), the 0 disagreement is an
            # artifact — not confidence — so exclude this provider from the aleatoric mean.
            n_heuristic = sum(1 for d in debs if d.get("used_heuristic"))
            all_heuristic = n_heuristic == len(debs)
            provider_aleatoric[provider] = (
                None if all_heuristic else structural_disagreement(outs)
            )
            rep_debug = dict(debs[0])
            if len(debs) > 1:
                rep_debug["k"] = len(debs)
                rep_debug["n_heuristic_samples"] = n_heuristic
                rep_debug["aleatoric"] = provider_aleatoric[provider]
            debug_map[provider] = rep_debug

        raw_outputs: dict[str, str | None] | None = None
        if self.settings.log_llm_raw:
            raw_outputs = {
                provider: debug_map[provider].pop("raw_response", None)
                for provider in debug_map
            }

        confidences = []
        for output in model_outputs.values():
            confidence = output.get("confidence")
            if confidence is None:
                confidence = 0.5
            confidences.append(float(confidence))

        specs = list(model_outputs.values())  # one representative per provider
        # EPISTEMIC (cross-model) over representatives; ALEATORIC averaged over providers.
        # Two epistemic signals are logged: bag-of-tokens JSD (epistemic_uncertainty, in
        # [0, ln2] — also normalized to [0,1]) and the field-aware Jaccard distance
        # (structural_uncertainty, in [0,1]). Pick the better one empirically via
        # `validate-uncertainty`.
        epistemic_uncertainty = _average_jsd(specs, msg.task.schema_context)
        epistemic_uncertainty_norm = min(epistemic_uncertainty / math.log(2), 1.0)
        structural_uncertainty = structural_disagreement(specs)
        # Average aleatoric over providers with a REAL (non-heuristic) within-model signal.
        real_aleatoric = [v for v in provider_aleatoric.values() if v is not None]
        aleatoric_uncertainty = (
            sum(real_aleatoric) / len(real_aleatoric) if real_aleatoric else 0.0
        )
        aleatoric_degraded = len(real_aleatoric) < len(provider_aleatoric)
        # Total predictive uncertainty = within-model (aleatoric) + cross-model epistemic.
        # Both terms are field-set Jaccard distances in [0,1], so the sum lives in [0,2]
        # and is dimensionally consistent (unlike mixing in the JSD, which is [0, ln2]).
        total_uncertainty = aleatoric_uncertainty + structural_uncertainty
        agreement = _normalized_agreement(epistemic_uncertainty)
        ambiguity = "high" if agreement < self.settings.agreement_threshold else "low"

        consensus = consensus_vote(specs, confidences)
        consensus["uncertainty_score"] = (
            max(output.get("uncertainty_score", 0.0) for output in specs) if specs else 0.0
        )
        consensus["ambiguity_points"] = sorted(set(consensus.get("ambiguity_points", [])))
        consensus["clarifying_questions"] = sorted(set(consensus.get("clarifying_questions", [])))

        parsing = SemanticParsingResult(
            semantic_spec=consensus,
            model_outputs=model_outputs,
            model_confidences={
                provider: float(model_outputs[provider].get("confidence", 0.5))
                for provider in model_outputs
            },
            epistemic_uncertainty=epistemic_uncertainty,
            semantic_agreement_score=agreement,
            ambiguity=ambiguity,
            ambiguity_reason="Model disagreement" if ambiguity == "high" else "",
            clarifying_questions=consensus.get("clarifying_questions", []),
            consensus_method="jsd_weighted_voting",
        )

        state = dict(msg.state)
        state.update(
            {
                "semantic_spec": parsing.semantic_spec,
                "model_outputs": parsing.model_outputs,
                "model_confidences": parsing.model_confidences,
                "epistemic_uncertainty": parsing.epistemic_uncertainty,
                "epistemic_uncertainty_norm": epistemic_uncertainty_norm,
                "structural_uncertainty": structural_uncertainty,
                "aleatoric_uncertainty": aleatoric_uncertainty,
                "aleatoric_degraded": aleatoric_degraded,
                "total_uncertainty": total_uncertainty,
                "semantic_agreement_score": parsing.semantic_agreement_score,
                "ambiguity": parsing.ambiguity,
                "ambiguity_reason": parsing.ambiguity_reason,
                "clarifying_questions": parsing.clarifying_questions,
                "consensus_method": parsing.consensus_method,
                "uncertainty_score": parsing.semantic_spec.get("uncertainty_score", 0.0),
                "llm_debug": {"providers": debug_map},
            }
        )
        if raw_outputs is not None:
            state["llm_raw_outputs"] = raw_outputs
        return AgentMessage(task=msg.task, state=state)

    def refine_with_feedback(
        self,
        original_spec: dict[str, Any],
        nl_request: str,
        user_answers: dict[str, str],
        schema_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Refine semantic spec based on user's answers to clarifying questions."""
        
        # Build a detailed prompt that guides the LLM to make specific changes
        prompt = (
            "You are refining a semantic analytics specification based on user feedback.\n\n"
            "TASK: Update the semantic spec to incorporate the user's answers.\n\n"
            f"Original user request: {nl_request}\n\n"
            "User answered these clarifying questions:\n"
        )
        
        for question, answer in user_answers.items():
            prompt += f"  Q: {question}\n  A: {answer}\n\n"
        
        prompt += (
            f"Current semantic spec:\n{original_spec}\n\n"
            f"Available schema:\n"
            f"  Models: {schema_context.get('models', [])}\n"
            f"  Columns: {schema_context.get('columns', {})}\n\n"
            "CRITICAL NAMING RULES (MUST FOLLOW):\n"
            "- base_measures: Use simple snake_case names like 'order_count', 'revenue_sum'\n"
            "  NEVER use SQL syntax like 'COUNT(order_id)' or 'SUM(amount)'\n"
            "  NEVER use table prefixes like 'orders.order_id'\n"
            "- group_by: Use simple column names like 'order_date', 'customer_id'\n"
            "  NEVER use table prefixes like 'orders.order_date'\n"
            "  For time dimensions, prefer 'metric_time' over physical column names\n"
            "- filters: Use MetricFlow Jinja syntax for time filters:\n"
            "  CORRECT: \"{{ TimeDimension('metric_time', 'day') }} >= current_date - interval '30 days'\"\n"
            "  WRONG: \"orders.order_date >= current_date - interval '30 days'\"\n"
            "  WRONG: \"order_date >= current_date - interval '30 days'\"\n\n"
            "INSTRUCTIONS:\n"
            "1. Parse each user answer and update the relevant field in the spec\n"
            "2. If user mentions 'count' or 'number of', set base_measures to ['order_count']\n"
            "3. If user mentions 'sum' or 'total', set base_measures to ['revenue_sum'] or similar\n"
            "4. If user specifies a time period, use MetricFlow filter syntax\n"
            "5. If user specifies granularity (e.g., 'daily'), set time_granularity to 'day'\n"
            "6. Keep fields unchanged if not addressed by user answers\n"
            "7. When the user asks to break results down 'by X' (e.g. 'by customer'), group by "
            "the SINGLE identifying column for X (e.g. 'customer_id'). Do NOT add descriptive "
            "columns such as names/labels unless the user explicitly asks to see them.\n"
            "8. A request to sort/rank ('highest to lowest') affects ordering only, not which "
            "columns are grouped or selected — do not add columns for it.\n\n"
            "Return the refined JSON spec with fields: metric_name, base_measures, group_by, "
            "time_granularity, filters, definition_notes, confidence, uncertainty_score."
        )
        
        # Deterministic first: with the keyword (canonical-phrasing) human the answers are
        # structured ("Group by order_date.", "Use the measure amount.") and name real
        # columns, so parsing them directly is far more reliable than re-prompting an LLM.
        refined_rules, changed = self._apply_answers_deterministic(
            original_spec, user_answers, schema_context
        )

        # LLM refiner: for free-form, natural-language answers ("count from the orders table
        # directly", "broken down by customer") the deterministic parser mis-handles the
        # phrasing. When enabled, interpret the answers with an LLM (grounded in the real
        # schema); the downstream designer maps any loose names to real columns. The
        # deterministic result is the fallback if the LLM call fails or returns nothing useful.
        if self.settings.llm_refiner and any((a or "").strip() for a in user_answers.values()):
            client = self.llm_clients.get("openai") or self.llm_clients.get("anthropic")
            if client:
                try:
                    spec = client.complete_json(
                        "You are an expert at refining analytics specifications from a "
                        "business user's plain-English answers. Incorporate every answer that "
                        "carries intent; use only the listed schema columns; keep slots the "
                        "answers do not address unchanged.",
                        prompt,
                        SemanticSpec,
                        temperature=0.1,
                    )
                    refined = spec.model_dump()
                    if refined.get("base_measures") or refined.get("group_by") or refined.get("filters"):
                        # Hybrid backstop: re-apply the deterministic content parser ON TOP of
                        # the LLM result to GUARANTEE clearly-stated slots the LLM sometimes
                        # drops or mis-sets actually land (e.g. "daily" -> day grain,
                        # "by customer" -> grouping, "number of orders" -> count). The LLM
                        # supplies the nuanced reading; this enforces the unambiguous keywords.
                        refined, _ = self._apply_answers_deterministic(
                            refined, user_answers, schema_context
                        )
                        refined["confidence"] = min(1.0, float(refined.get("confidence") or 0.5) + 0.2)
                        refined["uncertainty_score"] = max(0.0, float(refined.get("uncertainty_score") or 0.5) - 0.2)
                        return refined
                except Exception:
                    pass  # fall through to the deterministic / rule-based result

        if changed:
            refined_rules["confidence"] = min(1.0, float(refined_rules.get("confidence") or 0.5) + 0.2)
            refined_rules["uncertainty_score"] = max(
                0.0, float(refined_rules.get("uncertainty_score") or 0.5) - 0.2
            )
            return refined_rules

        client = self.llm_clients.get("openai") or self.llm_clients.get("anthropic")
        if not client:
            # Apply rule-based refinement if no LLM available
            return self._rule_based_refinement(original_spec, user_answers, schema_context)

        try:
            spec = client.complete_json(
                "You are an expert at refining analytics specifications based on user feedback.",
                prompt,
                SemanticSpec,
                temperature=0.2,  # Low temperature for consistent refinement
            )
            refined = spec.model_dump()
            # Increase confidence since user provided clarification
            refined["confidence"] = min(1.0, refined.get("confidence", 0.5) + 0.2)
            refined["uncertainty_score"] = max(0.0, refined.get("uncertainty_score", 0.5) - 0.2)
            return refined
        except Exception:
            return self._rule_based_refinement(original_spec, user_answers, schema_context)
    
    def _apply_answers_deterministic(
        self,
        spec: dict[str, Any],
        user_answers: dict[str, str],
        schema_context: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        """Apply clarifying answers to the spec by parsing answer CONTENT (not the question).

        Returns (refined_spec, changed). Only slots the answers clearly address are touched;
        everything else is preserved. Keyed on real schema columns so it never invents names.
        """
        refined = dict(spec)
        all_cols: set[str] = set()
        for cols in (schema_context.get("columns") or {}).values():
            all_cols.update(cols)
        cols_lower = {c.lower(): c for c in all_cols}
        changed = False

        def cols_in(text: str) -> list[str]:
            t = text.lower()
            return [orig for cl, orig in cols_lower.items() if re.search(rf"\b{re.escape(cl)}\b", t)]

        for answer in user_answers.values():
            if not answer or not answer.strip():
                continue
            a = answer.lower()
            # Parse CLAUSE BY CLAUSE: a column in the filter clause must not leak into the
            # measure/group_by slots ("Use the measure order_id. Group by status." once dumped
            # both columns into both slots and corrupted correct specs).
            clauses = [c.strip() for c in re.split(r"[.;]\s+|\n", answer) if c.strip()]

            for clause in clauses:
                cl = clause.lower()

                # Measure clause — columns in THIS clause, else the token after "measure"
                # (the sim-human may name a synthetic measure like 'order_count').
                if "measure" in cl:
                    mcols = cols_in(clause)
                    if mcols:
                        refined["base_measures"] = mcols
                        changed = True
                    else:
                        m = re.search(r"measure[s]?\s+(?:is\s+|are\s+|of\s+)?([a-z_][\w]*)", cl)
                        if m:
                            refined["base_measures"] = [m.group(1)]
                            changed = True

                # Grouping clause
                if any(p in cl for p in ("do not group", "single total", "single summary",
                                         "single number", "ungrouped", "no grouping")):
                    refined["group_by"] = []
                    changed = True
                elif any(p in cl for p in ("group by", "grouped by", "broken down by",
                                           "break down by", "breakdown by")):
                    gcols = cols_in(clause)
                    # The human speaks in entity/gold vocabulary ("group by customer"),
                    # not physical columns — resolve named tokens to schema columns
                    # (customer -> customer_id) so the answer is applied rather than
                    # silently dropped, mirroring the leniency the measure clause
                    # already has for non-column measure names. Unresolvable tokens are
                    # skipped, never inserted verbatim.
                    m = re.search(
                        r"(?:group(?:ed)? by|broken down by|break down by|breakdown by)"
                        r"\s+(.+)$", cl)
                    if m:
                        for tok in re.split(r",|\band\b", m.group(1)):
                            tok = re.sub(r"^(?:the|a|an)\s+", "", tok.strip())
                            tok = tok.strip().strip(".").replace(" ", "_")
                            if not tok:
                                continue
                            resolved = (
                                cols_lower.get(tok)
                                or cols_lower.get(f"{tok}_id")
                                or cols_lower.get(f"{tok.rstrip('s')}_id")
                            )
                            if resolved and resolved not in gcols:
                                gcols.append(resolved)
                    if gcols:
                        refined["group_by"] = gcols
                        changed = True

                # Filter clause
                if "no filter" in cl or "include all rows" in cl:
                    refined["filters"] = []
                    changed = True
                elif "filter" in cl:
                    m = re.search(r"filter\(s\)?:?\s*(.+)$", clause, re.IGNORECASE)
                    if m:
                        fs = [f.strip().rstrip(".").strip()
                              for f in re.split(r",(?![^()]*\))", m.group(1)) if f.strip()]
                        # Several answers often repeat the same reveal — appending each
                        # repeat stacked the identical predicate 5x deep on real_027.
                        fs = [f for f in fs if f and f not in (refined.get("filters") or [])]
                        if fs:
                            refined["filters"] = list(refined.get("filters") or []) + fs
                            changed = True
                    days = re.search(r"last\s+(\d+)\s*days?", cl)
                    if days:
                        token = f"last_{days.group(1)}_days"
                        if token not in (refined.get("filters") or []):
                            refined["filters"] = list(refined.get("filters") or []) + [token]
                            changed = True

            # Aggregation + granularity are keyword-only (no column conflation), so the whole
            # answer is safe. "single/grand/a total" describes the GROUPING (a one-row
            # result), not a SUM — stripping it stops "return a single total" from turning a
            # count measure into SUM(order_id)=4950 instead of COUNT=99.
            agg_text = re.sub(r"\b(?:a|one|single|grand)\s+total\b", " ", a)
            measure_names = [str(m).lower() for m in (refined.get("base_measures") or [])]
            measure_is_count = any("_count" in m or m == "count" for m in measure_names)
            agg = None
            if re.search(r"\bcount\b|number of", agg_text) or measure_is_count:
                agg = "count"
            elif re.search(r"\bsum\b|\btotal\b|revenue", agg_text):
                agg = "sum"
            elif re.search(r"\baverage\b|\bmean\b", agg_text):
                agg = "average"
            if agg:
                refined["aggregation"] = agg
                changed = True

            if not re.search(r"\bno[\w\s]*granularity|granularity[\w\s]*not needed", a):
                for token, gran in (("dai", "day"), ("day", "day"), ("week", "week"),
                                    ("month", "month"), ("quarter", "quarter"), ("year", "year")):
                    if token in a:
                        refined["time_granularity"] = gran
                        # A granularity implies grouping by the time dimension, else the spec
                        # carries a grain but collapses to a single total.
                        gb = list(refined.get("group_by") or [])
                        time_cols = [c for c in all_cols if _looks_like_time_col(c)]
                        if time_cols and not any(_looks_like_time_col(g) for g in gb):
                            gb.append(time_cols[0])
                            refined["group_by"] = gb
                        changed = True
                        break

        return refined, changed

    def _rule_based_refinement(
        self,
        spec: dict[str, Any],
        user_answers: dict[str, str],
        schema_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply rule-based refinement when LLM is unavailable."""
        refined = dict(spec)
        
        # Build schema-aware measure candidates (no hardcoded names)
        all_columns: set[str] = set()
        for cols in schema_context.get("columns", {}).values():
            all_columns.update(cols)
        
        # Find columns that look like counts or amounts
        count_columns = [c for c in all_columns if any(p in c.lower() for p in ["count", "num", "number", "qty"])]
        amount_columns = [c for c in all_columns if any(p in c.lower() for p in ["amount", "revenue", "total", "value", "price"])]
        id_columns = [c for c in all_columns if c.lower().endswith("_id") or c.lower() == "id"]
        
        for question, answer in user_answers.items():
            answer_lower = answer.lower()
            question_lower = question.lower()
            
            # Handle aggregation type answers - use schema columns, not hardcoded names
            if "count" in question_lower or "measure" in question_lower:
                if "count" in answer_lower or "number" in answer_lower:
                    # Use a pre-aggregated count column if available, otherwise use ID for counting
                    if count_columns:
                        refined["base_measures"] = [count_columns[0]]
                    elif id_columns:
                        refined["base_measures"] = [f"{id_columns[0].replace('_id', '')}_count"]
                    else:
                        refined["base_measures"] = ["row_count"]
                elif "sum" in answer_lower or "total" in answer_lower or "revenue" in answer_lower:
                    if amount_columns:
                        refined["base_measures"] = [amount_columns[0]]
                    else:
                        refined["base_measures"] = ["total_amount"]
            
            # Handle time granularity answers
            if "granularity" in question_lower or "time" in question_lower:
                if "day" in answer_lower or "daily" in answer_lower:
                    refined["time_granularity"] = "day"
                elif "week" in answer_lower or "weekly" in answer_lower:
                    refined["time_granularity"] = "week"
                elif "month" in answer_lower or "monthly" in answer_lower:
                    refined["time_granularity"] = "month"
            
            # Handle time period/filter answers
            if "period" in question_lower or "time" in question_lower or "days" in answer_lower:
                import re
                days_match = re.search(r"(\d+)\s*days?", answer_lower)
                if days_match:
                    days = days_match.group(1)
                    refined["filters"] = refined.get("filters", [])
                    refined["filters"].append(f"last_{days}_days")
            
            # Handle dimension answers
            if "dimension" in question_lower or "group" in question_lower:
                columns = schema_context.get("columns", {})
                all_cols = set()
                for cols in columns.values():
                    all_cols.update(cols)
                
                for col in all_cols:
                    if col in answer_lower:
                        refined["group_by"] = refined.get("group_by", [])
                        if col not in refined["group_by"]:
                            refined["group_by"].append(col)
        
        return refined
