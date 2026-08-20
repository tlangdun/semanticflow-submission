"""Tests for verification-grounded confidence (semanticflow/evaluation/spec_verification.py).

These pin the behaviour demonstrated in the 2026-06-08 diagnostic: the deterministic
schema checks fire on real failure modes (avg->sum, ratio, stale relative time, unknown
column, missing grain) and stay silent on a well-grounded spec.
"""
from __future__ import annotations

from semanticflow.evaluation.spec_verification import verify_spec

_ORDERS_SCHEMA = {"models": ["orders"],
                  "columns": {"orders": ["order_id", "customer_id", "order_date", "amount", "status"]}}


def test_average_request_no_longer_fires_after_path_a():
    # Post-Path-A the designer derives 'average' from the NL verb, so avg is handled and
    # must NOT be flagged as a defect (was a true positive before; now a false positive).
    spec = {"metric_name": "average_order_value_per_month", "base_measures": ["amount"],
            "group_by": ["order_date"], "time_granularity": "month"}
    assert verify_spec("Show average order value per month.", spec, _ORDERS_SCHEMA).fired["agg_verb"] is False


def test_percentage_request_fires_ratio():
    # Ratio/percentage stays unexpressible (needs numerator/denominator) -> still flagged.
    spec = {"metric_name": "coupon_orders", "base_measures": ["coupon_amount"]}
    assert verify_spec("What percentage of orders use coupons?", spec, _ORDERS_SCHEMA).fired["agg_verb"]


def test_count_request_does_not_fire():
    spec = {"metric_name": "order_count", "base_measures": ["order_id"]}
    assert verify_spec("How many orders are there?", spec, _ORDERS_SCHEMA).fired["agg_verb"] is False


def test_unknown_column_fires():
    spec = {"metric_name": "revenue", "base_measures": ["revenue_total"]}  # not a real column
    res = verify_spec("total revenue", spec, _ORDERS_SCHEMA)
    assert res.fired["unknown_col"] is True


def test_missing_grain_fires():
    spec = {"metric_name": "orders", "base_measures": ["amount"], "time_granularity": None}
    assert verify_spec("show revenue per month", spec, _ORDERS_SCHEMA).fired["missing_grain"]


def test_stale_relative_time_fires():
    spec = {"metric_name": "rev", "base_measures": ["amount"], "filters": []}
    assert verify_spec("How are we doing this month vs last month?", spec, _ORDERS_SCHEMA).fired["stale_relative_time"]


def test_clean_spec_is_silent():
    # "How much revenue" -> sum of amount, no grain/relative-time: nothing should fire.
    spec = {"metric_name": "total_revenue", "base_measures": ["amount"], "group_by": []}
    res = verify_spec("How much total revenue did we make?", spec, _ORDERS_SCHEMA)
    assert res.score == 0
    assert res.violations == []


def test_violations_are_human_readable_for_hitl():
    # Relative-time on static data yields a plain-language, answerable clarification.
    spec = {"metric_name": "rev", "base_measures": ["amount"], "filters": []}
    res = verify_spec("How are we doing this month vs last month?", spec, _ORDERS_SCHEMA)
    assert res.violations
    assert any("date range" in v or "relative time" in v for v in res.violations)
