from __future__ import annotations

import os
import re
from typing import Any

from semanticflow.agents.types import Agent, AgentMessage
from semanticflow.config import Settings
from semanticflow.evaluation import compare_results, run_gold_sql


def _norm_token(value: Any) -> str:
    """Canonicalize a spec token for diagnostic set comparison: lowercase, drop
    quotes/whitespace, collapse separators. Diagnostics only — never the headline."""
    text = re.sub(r"['\"`]", "", str(value).strip().lower())
    return re.sub(r"\s+", " ", text)


def _norm_set(items: Any) -> set[str]:
    if not items:
        return set()
    if not isinstance(items, (list, set, tuple)):
        items = [items]
    return {_norm_token(item) for item in items if str(item).strip()}


def _f1(pred: set[str], gold: set[str]) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _expected_field(expected: dict[str, Any], key: str, fallback: str | None = None) -> Any:
    if key in expected:
        return expected[key]
    if fallback and fallback in expected:
        return expected[fallback]
    return None


_AGG_SYNONYMS = {"avg": "average", "mean": "average", "count_distinct": "distinct",
                 "unique": "distinct"}


def _norm_agg(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = _norm_token(value)
    return _AGG_SYNONYMS.get(text, text)


def _score_against_spec(
    semantic_spec: dict[str, Any], expected: dict[str, Any]
) -> dict[str, Any]:
    """Spec-field agreement of ``semantic_spec`` against ONE reference ``expected``.

    Returns the per-field f1s, the metric-name similarity, and ``spec_field_score``
    (mean of measures/dimensions/filters f1). Pure function over two dicts — used both
    for the single-gold path and, taking the max, for the multi-gold acceptable set.
    metric_name is reported as a soft similarity only and never enters the score.
    """
    predicted_metric = _norm_set([semantic_spec.get("metric_name")])
    expected_metric = _norm_set([_expected_field(expected, "metric", "metric_name")])

    predicted_group_by = _norm_set(semantic_spec.get("group_by"))
    expected_group_by = _norm_set(_expected_field(expected, "group_by"))

    predicted_filters = _norm_set(semantic_spec.get("filters"))
    expected_filters = _norm_set(_expected_field(expected, "filters"))

    predicted_measures = _norm_set(semantic_spec.get("base_measures"))
    expected_measures = _norm_set(_expected_field(expected, "base_measures", "base_measure"))

    measures_f1 = _f1(predicted_measures, expected_measures)
    dimensions_f1 = _f1(predicted_group_by, expected_group_by)
    filters_f1 = _f1(predicted_filters, expected_filters)

    # Aggregation agreement: scored only when the gold spec pins an aggregation (a SUM
    # where the gold says AVERAGE was previously invisible at the spec level). A spec
    # that omits aggregation when the gold specifies one is underspecified — scored 0.
    expected_agg = _norm_agg(_expected_field(expected, "aggregation"))
    aggregation_match: float | None = None
    parts = [measures_f1, dimensions_f1, filters_f1]
    if expected_agg is not None:
        aggregation_match = 1.0 if _norm_agg(semantic_spec.get("aggregation")) == expected_agg else 0.0
        parts.append(aggregation_match)

    return {
        "metric_name_similarity": _jaccard(predicted_metric, expected_metric),
        "measures_f1": measures_f1,
        "dimensions_f1": dimensions_f1,
        "group_by_f1": dimensions_f1,  # back-compat alias
        "filters_f1": filters_f1,
        "aggregation_match": aggregation_match,
        "spec_field_score": sum(parts) / len(parts),
    }


class EvaluatorAgent(Agent):
    name: str = "evaluator"
    settings: Settings

    def _resolve_duckdb_path(self) -> str | None:
        if self.settings.duckdb_path:
            return self.settings.duckdb_path
        env_path = os.getenv("DBT_DUCKDB_PATH") or os.getenv("DUCKDB_PATH")
        if env_path:
            return env_path
        default_path = "jaffle_shop.duckdb"
        return default_path if os.path.exists(default_path) else None

    def _normalize_gold_result(self, gold: Any) -> list[dict[str, Any]]:
        if not gold:
            return []
        if isinstance(gold, list):
            return [row for row in gold if isinstance(row, dict)]
        if isinstance(gold, dict):
            rows = gold.get("rows") or gold.get("data")
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
            return [gold]
        return []

    async def run(self, msg: AgentMessage) -> AgentMessage:
        expected = msg.task.expected_semantic
        semantic_spec = msg.state.get("semantic_spec", {})
        dbt_results = msg.state.get("dbt_results", {})
        evaluation: dict[str, Any] = {
            "dbt_parse_success": bool(dbt_results.get("dbt_parse_success")),
            "dbt_build_success": bool(dbt_results.get("dbt_build_success")),
        }

        spec_field_score: float | None = None
        if expected:
            # Spec-field agreement — DIAGNOSTIC ONLY (not the headline).
            # metric_name is a freely-invented label, so it is reported as a soft
            # token similarity and deliberately excluded from any score.
            #
            # Multi-reference: an underspecified task may carry ``acceptable_semantics`` —
            # additional valid interpretations of an ambiguous request. The reported
            # spec-field metrics are those of the BEST-matching acceptable reading (max over
            # {expected_semantic} ∪ acceptable_semantics), so a correct-but-different
            # disambiguation is not penalised. With no acceptable_semantics this is exactly
            # the original single-gold score.
            references = [expected, *(msg.task.acceptable_semantics or [])]
            scored = [_score_against_spec(semantic_spec, ref) for ref in references]
            best = max(scored, key=lambda s: s["spec_field_score"])
            spec_field_score = best["spec_field_score"]
            evaluation.update(best)
            if len(references) > 1:
                evaluation["spec_field_score_n_refs"] = len(references)

        state = dict(msg.state)
        gold_sql_present = bool(msg.task.gold_sql or msg.task.gold_query_result)
        gold_sql_exec_success = False
        gold_result: list[dict[str, Any]] = []
        execution_match = False
        execution_accuracy = 0.0
        result_diff_summary: dict[str, Any] = {}

        if gold_sql_present:
            if msg.task.gold_query_result:
                gold_result = self._normalize_gold_result(msg.task.gold_query_result)
                gold_sql_exec_success = True
            elif msg.task.gold_sql:
                duckdb_path = self._resolve_duckdb_path()
                gold_sql_exec_success, gold_result, error = run_gold_sql(
                    msg.task.gold_sql,
                    duckdb_path=duckdb_path,
                )
                if error:
                    result_diff_summary["gold_sql_error"] = error

            actual_result = state.get("mf_query_result", []) or []
            mf_query_success = bool(state.get("mf_query_success"))
            # Multi-gold execution: a result is correct if it execution-matches the primary
            # gold OR any acceptable alternative table. We compare against each candidate and
            # keep the best (a full match short-circuits). With no acceptable_results this is
            # a single comparison against gold_result — identical to the original path.
            candidate_golds = [gold_result, *(msg.task.acceptable_results or [])]
            comparison = None
            for cand in candidate_golds:
                cmp = compare_results(
                    cand,
                    actual_result,
                    result_schema=msg.task.result_schema,
                    compare_rules=msg.task.compare_rules,
                )
                if comparison is None or cmp.accuracy > comparison.accuracy:
                    comparison = cmp
                if cmp.match:
                    break
            if len(candidate_golds) > 1:
                evaluation["execution_n_refs"] = len(candidate_golds)
            execution_match = comparison.match
            execution_accuracy = comparison.accuracy
            # Guard against two false-positive paths:
            #  1. The MetricFlow query errored — an empty/failed result can still
            #     "match" an empty gold. A query that did not run cannot be correct.
            #  2. Empty-gold vs empty-actual ([] == []) trivially matches. Only credit
            #     execution-equivalence when the gold actually returned rows.
            if not mf_query_success:
                execution_match = False
                execution_accuracy = 0.0
                result_diff_summary["execution_voided"] = "mf_query_failed"
            elif not gold_result:
                execution_match = False
                execution_accuracy = 0.0
                result_diff_summary["execution_voided"] = "empty_gold"
            result_diff_summary.update(comparison.diff)
            evaluation["execution_accuracy"] = execution_accuracy
            evaluation["execution_match"] = execution_match

        # A task is execution-scorable when the gold ran and produced rows. A failed
        # MetricFlow query against such a gold is an execution *failure* (score 0, voided
        # above), NOT a reason to fall back to spec-fields — that would inflate the
        # headline. Fall back only when the gold itself is unusable (errored or empty).
        gold_execution_scorable = (
            gold_sql_present and gold_sql_exec_success and bool(gold_result)
        )

        # Headline accuracy: prefer execution-equivalence (the trustworthy signal);
        # fall back to spec-field agreement only when no executable gold exists.
        if gold_execution_scorable:
            evaluation["overall_score"] = execution_accuracy
            evaluation["accuracy_basis"] = "execution"
        elif spec_field_score is not None:
            evaluation["overall_score"] = spec_field_score
            evaluation["accuracy_basis"] = "spec_fields"
        else:
            evaluation["overall_score"] = None
            evaluation["accuracy_basis"] = "none"
        evaluation["reasoning"] = ""

        state["gold_sql_present"] = gold_sql_present
        state["gold_sql_exec_success"] = gold_sql_exec_success
        state["gold_result"] = gold_result
        state["execution_accuracy"] = execution_accuracy
        state["execution_match"] = execution_match
        state["result_diff_summary"] = result_diff_summary
        state["evaluation"] = evaluation
        return AgentMessage(task=msg.task, state=state)
