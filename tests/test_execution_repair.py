"""Tests for the execution-feedback repair step (execution_repair.py).

When a generated/refined spec ERRORS or returns EMPTY at MetricFlow execution, the
concrete mf error is fed back and ONE repaired spec is produced for the caller to
re-execute. These pin each deterministic error->edit mapping, the no-op case, and the
mocked-LLM fallback path. No live API.
"""
from __future__ import annotations

from typing import Any

from semanticflow.agents.execution_repair import (
    classify_mf_error,
    repair_spec_from_execution,
)

# Single-model schema (co-located) — jaffle_shop orders.
_SCHEMA = {
    "models": ["orders"],
    "columns": {"orders": ["order_id", "customer_id", "order_date", "amount", "status"]},
}

# Two-model schema where measure (amount on payments) and a requested dimension
# (status) are split — until 'orders' co-locates both for the fan-out repair.
_SCHEMA_SPLIT = {
    "models": ["stg_payments", "orders"],
    "columns": {
        "stg_payments": ["payment_id", "order_id", "amount_cents"],
        "orders": ["order_id", "customer_id", "order_date", "amount", "status"],
    },
}


# --- Fake LLM client (no network) -------------------------------------------
class FakeLLMClient:
    """Records the prompt and returns a canned JSON corrected spec."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.last_user_prompt: str | None = None

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> str:
        self.last_user_prompt = user_prompt
        return self.response


class RaisingLLMClient:
    def complete(self, *args: Any, **kwargs: Any) -> str:
        raise RuntimeError("boom")


# --- Error classification ----------------------------------------------------
class TestClassify:
    def test_unknown_metric(self):
        err = "The given input does not exactly match any known metrics. Suggestions: ['orders_per_day']"
        assert classify_mf_error(err) == "unknown_metric"

    def test_fanout(self):
        err = "The given input does not match any of the available group-by-items ... No valid join paths exist (fan-out join support is pending)"
        assert classify_mf_error(err) == "fanout_join"

    def test_binder_current_date(self):
        err = "Binder Error: Referenced column \"current_date\" not found ... does not have a column named 'current_date'"
        assert classify_mf_error(err) == "binder_current_date"

    def test_unrecognised(self):
        assert classify_mf_error("some totally unrelated traceback") == "none"

    def test_empty(self):
        assert classify_mf_error("") == "none"


# --- Deterministic: metric-name suggestion ----------------------------------
class TestMetricNameSuggestion:
    def test_adopts_suggestion(self):
        spec = {"metric_name": "orders_perday", "base_measures": ["order_id"]}
        err = "does not exactly match any known metrics. Suggestions: ['orders_per_day', 'order_count']"
        out = repair_spec_from_execution(spec, err, _SCHEMA)
        assert out["metric_name"] == "orders_per_day"  # near-match preferred

    def test_first_suggestion_when_no_near_match(self):
        spec = {"metric_name": "zzz", "base_measures": ["amount"]}
        err = "does not exactly match any known metrics. Suggestions: ['total_revenue', 'order_count']"
        out = repair_spec_from_execution(spec, err, _SCHEMA)
        assert out["metric_name"] == "total_revenue"

    def test_no_suggestions_is_noop(self):
        spec = {"metric_name": "zzz"}
        err = "does not exactly match any known metrics."
        out = repair_spec_from_execution(spec, err, _SCHEMA)
        assert out == spec


# --- Deterministic: relative-time -> absolute -------------------------------
class TestRelativeTimeToAbsolute:
    def test_binder_error_converts_relative_filter(self):
        spec = {
            "metric_name": "order_count",
            "source_model": "orders",
            "base_measures": ["order_id"],
            "filters": ["{{ TimeDimension('metric_time', 'day') }} >= current_date - interval '30 days'"],
        }
        err = "Binder Error: ... does not have a column named 'current_date'"
        out = repair_spec_from_execution(spec, err, _SCHEMA)
        # relative clause dropped, absolute range within 2018 span added
        assert not any("current_date" in f for f in out["filters"])
        assert any(">= '2018-01-01'" in f for f in out["filters"])
        assert any("<= '2018-04-09'" in f for f in out["filters"])
        # uses the discovered physical time column
        assert any("order_date" in f for f in out["filters"])

    def test_empty_result_with_relative_filter(self):
        spec = {
            "metric_name": "order_count",
            "source_model": "orders",
            "base_measures": ["order_id"],
            "filters": ["last_30_days"],
        }
        # empty error text signals an EMPTY result
        out = repair_spec_from_execution(spec, "", _SCHEMA)
        assert not any("last_30_days" in f for f in out["filters"])
        assert any("'2018-" in f for f in out["filters"])

    def test_uses_data_span_override(self):
        schema = dict(_SCHEMA, data_span={"min": "2020-01-01", "max": "2020-12-31"})
        spec = {"source_model": "orders", "filters": ["order_date >= current_date - interval '7 days'"]}
        out = repair_spec_from_execution(spec, "binder error current_date", schema)
        assert any("2020-01-01" in f for f in out["filters"])

    def test_empty_result_no_relative_filter_is_noop(self):
        spec = {"source_model": "orders", "filters": ["status = 'completed'"]}
        out = repair_spec_from_execution(spec, "", _SCHEMA)
        assert out == spec


# --- Deterministic: fan-out -> co-located source_model ----------------------
class TestFanoutRepair:
    def test_swaps_to_colocated_model(self):
        # measure 'amount' + group_by 'status' both live in 'orders', not the current
        # 'stg_payments' source -> swap source_model to 'orders'.
        spec = {
            "metric_name": "revenue",
            "source_model": "stg_payments",
            "base_measures": ["amount"],
            "group_by": ["status"],
        }
        err = "No valid join paths exist (fan-out join support is pending)"
        out = repair_spec_from_execution(spec, err, _SCHEMA_SPLIT)
        assert out["source_model"] == "orders"

    def test_noop_when_no_colocated_model(self):
        # measure column not present in any single model alongside the dimension
        spec = {
            "source_model": "stg_payments",
            "base_measures": ["nonexistent_col"],
            "group_by": ["status"],
        }
        err = "No valid join paths exist (fan-out join support is pending)"
        out = repair_spec_from_execution(spec, err, _SCHEMA_SPLIT)
        assert out == spec


# --- No-op cases -------------------------------------------------------------
class TestNoOp:
    def test_unrecognised_error_no_client_unchanged(self):
        spec = {"metric_name": "x", "base_measures": ["amount"]}
        out = repair_spec_from_execution(spec, "weird unrelated error", _SCHEMA)
        assert out == spec

    def test_non_dict_spec_returned_as_is(self):
        assert repair_spec_from_execution(None, "err", _SCHEMA) is None  # type: ignore[arg-type]


# --- Mocked-LLM fallback path ------------------------------------------------
class TestLLMFallback:
    def test_llm_called_only_when_deterministic_noop(self):
        # unrecognised error -> deterministic no-op -> LLM consulted
        spec = {"metric_name": "x", "source_model": "orders", "base_measures": ["amount"]}
        client = FakeLLMClient(
            '{"metric_name": "fixed_metric", "source_model": "orders", '
            '"base_measures": ["amount"], "group_by": ["status"]}'
        )
        out = repair_spec_from_execution(
            spec, "some opaque error", _SCHEMA, nl_request="revenue by status", llm_client=client
        )
        assert out["metric_name"] == "fixed_metric"
        assert out["group_by"] == ["status"]
        assert client.last_user_prompt is not None
        assert "revenue by status" in client.last_user_prompt

    def test_deterministic_wins_over_llm(self):
        # recognised error -> deterministic fix taken, LLM NOT consulted
        spec = {"metric_name": "orders_perday", "base_measures": ["order_id"]}
        err = "does not exactly match any known metrics. Suggestions: ['orders_per_day']"
        client = FakeLLMClient('{"metric_name": "should_not_be_used"}')
        out = repair_spec_from_execution(spec, err, _SCHEMA, llm_client=client)
        assert out["metric_name"] == "orders_per_day"
        assert client.last_user_prompt is None  # LLM never called

    def test_llm_merge_defensive_keeps_original_on_dropped_field(self):
        spec = {"metric_name": "x", "source_model": "orders", "base_measures": ["amount"], "aggregation": "sum"}
        # LLM omits aggregation -> original preserved
        client = FakeLLMClient('{"metric_name": "y"}')
        out = repair_spec_from_execution(spec, "opaque", _SCHEMA, llm_client=client)
        assert out["metric_name"] == "y"
        assert out["aggregation"] == "sum"

    def test_llm_raising_returns_spec_unchanged(self):
        spec = {"metric_name": "x", "base_measures": ["amount"]}
        out = repair_spec_from_execution(spec, "opaque", _SCHEMA, llm_client=RaisingLLMClient())
        assert out == spec

    def test_llm_bad_json_returns_spec_unchanged(self):
        spec = {"metric_name": "x", "base_measures": ["amount"]}
        client = FakeLLMClient("not json at all")
        out = repair_spec_from_execution(spec, "opaque", _SCHEMA, llm_client=client)
        assert out == spec
