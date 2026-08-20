"""Tests for verifier-guided repair (semanticflow/evaluation/verification_repair.py).

Each test feeds REAL ``verify_spec`` violations into ``auto_repair_from_verification`` so
the repairer's dispatch stays pinned to the verifier's actual output contract (no API,
synthetic fixtures only). One test per auto-fixable class, one for an un-fixable/ambiguous
violation left untouched, and the empty-violations no-op.
"""
from __future__ import annotations

from semanticflow.evaluation.spec_verification import verify_spec
from semanticflow.evaluation.verification_repair import (
    WAREHOUSE_END,
    auto_repair_from_verification,
)

_ORDERS_SCHEMA = {
    "models": ["orders"],
    "columns": {"orders": ["order_id", "customer_id", "order_date", "amount", "status"]},
}


def _violations(nl: str, spec: dict) -> list[str]:
    return verify_spec(nl, spec, _ORDERS_SCHEMA).violations


# --- auto-fixable: missing_grain --------------------------------------------------------
def test_missing_grain_sets_time_granularity():
    nl = "Show revenue per month."
    spec = {"metric_name": "gross_revenue_per_month", "base_measures": ["amount"],
            "time_granularity": None}
    violations = _violations(nl, spec)
    assert violations  # verifier flagged it

    repaired, applied = auto_repair_from_verification(spec, violations, _ORDERS_SCHEMA)

    assert repaired["time_granularity"] == "month"
    assert any("time_granularity" in f and "month" in f for f in applied)
    assert spec["time_granularity"] is None  # input not mutated


def test_missing_grain_recovers_weekly_grain():
    nl = "Weekly order counts."
    spec = {"metric_name": "weekly_order_count", "base_measures": ["order_id"],
            "time_granularity": None}
    violations = _violations(nl, spec)
    repaired, applied = auto_repair_from_verification(spec, violations, _ORDERS_SCHEMA)
    assert repaired["time_granularity"] == "week"
    assert applied


# --- auto-fixable: stale_relative_time --------------------------------------------------
def test_stale_relative_time_becomes_absolute_range():
    nl = "How are we doing this month?"
    spec = {"metric_name": "gross_revenue_this_month", "base_measures": ["amount"],
            "filters": []}
    violations = _violations(nl, spec)
    assert violations

    repaired, applied = auto_repair_from_verification(spec, violations, _ORDERS_SCHEMA)

    # 'this month' on the static warehouse anchors to April 2018.
    assert "order_date >= '2018-04-01'" in repaired["filters"]
    assert f"order_date <= '{WAREHOUSE_END}'" in repaired["filters"]
    assert any("absolute range" in f for f in applied)
    assert spec["filters"] == []  # input not mutated


def test_stale_last_month_window():
    nl = "Total revenue last month."
    spec = {"metric_name": "gross_revenue_last_month", "base_measures": ["amount"],
            "filters": []}
    violations = _violations(nl, spec)
    repaired, applied = auto_repair_from_verification(spec, violations, _ORDERS_SCHEMA)
    assert "order_date >= '2018-03-01'" in repaired["filters"]
    assert "order_date <= '2018-03-31'" in repaired["filters"]
    assert applied


def test_stale_relative_time_skipped_when_absolute_bound_present():
    # An existing absolute order_date bound means the query is already constrained;
    # don't double-add a window.
    nl = "Revenue this month."
    spec = {"metric_name": "gross_revenue_this_month", "base_measures": ["amount"],
            "filters": ["order_date >= '2018-04-01'"]}
    violations = _violations(nl, spec)
    repaired, applied = auto_repair_from_verification(spec, violations, _ORDERS_SCHEMA)
    assert repaired["filters"] == ["order_date >= '2018-04-01'"]
    assert not any("absolute range" in f for f in applied)


# --- auto-fixable: unknown_col close match ----------------------------------------------
def test_unknown_measure_corrected_to_close_schema_column():
    nl = "Total revenue."
    spec = {"metric_name": "revenue", "base_measures": ["amountt"]}  # typo of 'amount'
    violations = _violations(nl, spec)
    assert violations

    repaired, applied = auto_repair_from_verification(spec, violations, _ORDERS_SCHEMA)

    assert repaired["base_measures"] == ["amount"]
    assert any("amountt" in f and "amount'" in f for f in applied)


def test_unknown_group_by_corrected_to_close_schema_column():
    nl = "Orders by status."
    spec = {"metric_name": "order_count", "base_measures": ["order_id"],
            "group_by": ["statuss"]}  # typo of 'status'
    violations = _violations(nl, spec)
    repaired, applied = auto_repair_from_verification(spec, violations, _ORDERS_SCHEMA)
    assert repaired["group_by"] == ["status"]
    assert applied


# --- un-fixable / ambiguous left untouched ----------------------------------------------
def test_ratio_violation_left_for_human():
    nl = "What percentage of orders use coupons?"
    spec = {"metric_name": "coupon_share", "base_measures": ["amount"]}
    violations = _violations(nl, spec)
    assert violations  # ratio defect fired

    repaired, applied = auto_repair_from_verification(spec, violations, _ORDERS_SCHEMA)

    assert repaired == spec  # nothing safely fixable
    assert applied == []


def test_unknown_column_with_no_close_match_left_for_human():
    # 'region' has no close column in the orders schema -> a fix would be a guess.
    nl = "Revenue by region."
    spec = {"metric_name": "revenue", "base_measures": ["amount"], "group_by": ["region"]}
    violations = _violations(nl, spec)
    assert violations

    repaired, applied = auto_repair_from_verification(spec, violations, _ORDERS_SCHEMA)

    assert repaired["group_by"] == ["region"]  # untouched
    assert applied == []


def test_missing_grain_with_unrecoverable_word_left_for_human():
    # Verifier fires on the NL grain word, but the spec text carries no grain word for the
    # repairer to recover deterministically -> leave for a human, no guess.
    nl = "Show revenue over time."
    spec = {"metric_name": "gross_revenue", "base_measures": ["amount"],
            "time_granularity": None, "definition_notes": ""}
    violations = _violations(nl, spec)
    assert violations  # 'over time' fired missing_grain

    repaired, applied = auto_repair_from_verification(spec, violations, _ORDERS_SCHEMA)
    assert repaired["time_granularity"] is None
    assert applied == []


# --- no-op cases ------------------------------------------------------------------------
def test_empty_violations_is_noop():
    spec = {"metric_name": "gross_revenue", "base_measures": ["amount"]}
    repaired, applied = auto_repair_from_verification(spec, [], _ORDERS_SCHEMA)
    assert repaired == spec
    assert applied == []


def test_clean_spec_roundtrip_is_noop():
    nl = "Total revenue across all orders."
    spec = {"metric_name": "gross_revenue", "base_measures": ["amount"]}
    violations = _violations(nl, spec)
    assert violations == []  # nothing fired
    repaired, applied = auto_repair_from_verification(spec, violations, _ORDERS_SCHEMA)
    assert repaired == spec
    assert applied == []


def test_multiple_violations_mixed_fix_and_leave():
    # Missing grain (fixable) + ratio (left). Repairer applies one, leaves the other.
    nl = "What percentage of orders complete per month?"
    spec = {"metric_name": "completion_rate_per_month", "base_measures": ["amount"],
            "time_granularity": None}
    violations = _violations(nl, spec)
    assert len(violations) >= 2

    repaired, applied = auto_repair_from_verification(spec, violations, _ORDERS_SCHEMA)
    assert repaired["time_granularity"] == "month"  # grain fixed
    assert len(applied) == 1  # ratio not reported
