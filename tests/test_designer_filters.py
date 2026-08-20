"""Tests for categorical-filter handling (the WHERE status='completed' binder-error fix).

Three coordinated pieces let a categorical filter execute in MetricFlow:
1. _is_time_column must NOT misclassify 'status' (contains 'at') as a time dimension.
2. _extract_filter_columns surfaces filter-only columns so they become dimensions.
3. _sanitize_filter rewrites bare SQL into the Jinja Dimension form MetricFlow requires.
"""
from __future__ import annotations

from semanticflow.agents.designer import (
    _extract_filter_columns,
    _is_time_column,
    _normalize_enum_literals,
    _sanitize_filter,
)


class TestIsTimeColumn:
    def test_status_is_not_time(self):
        # Regression: substring set included "at", so "st-at-us" was a time column.
        assert _is_time_column("status", None) is False

    def test_substring_false_positives_fixed(self):
        for col in ("category", "birthday", "latitude", "format"):
            assert _is_time_column(col, None) is False, col

    def test_genuine_time_columns(self):
        for col in ("order_date", "created_at", "order_ts", "metric_time", "signup_day"):
            assert _is_time_column(col, None) is True, col

    def test_data_type_is_authoritative(self):
        # A column typed varchar is never time, even if oddly named.
        assert _is_time_column("day_label", {"day_label": "varchar"}) is False
        assert _is_time_column("x", {"x": "timestamp"}) is True

    def test_date_typed_column_without_time_token_is_time(self):
        # Regression real_020: customers.first_order is a DATE but carries no time
        # token, so without types it was declared categorical and the month grain
        # could never be applied at query time.
        assert _is_time_column("first_order", {"first_order": "DATE"}) is True
        assert _is_time_column("first_order", None) is False  # name alone can't tell


class TestDimensionForTypedDate:
    def test_first_order_becomes_time_dimension(self):
        from semanticflow.agents.designer import _dimension_for

        dim = _dimension_for(
            "first_order",
            {"customer_id", "first_order", "number_of_orders"},
            {"first_order": "DATE"},
        )
        assert dim is not None and dim.type == "time"


class TestDuckdbDataTypes:
    def test_warehouse_types_for_jaffle_shop(self):
        from semanticflow.dbt_integration.project_inspect import duckdb_data_types

        types = duckdb_data_types("third_party/jaffle_shop_duckdb")
        assert types.get("customers", {}).get("first_order") == "DATE"
        assert types.get("orders", {}).get("order_date") == "DATE"

    def test_missing_project_dir_returns_empty(self):
        from semanticflow.dbt_integration.project_inspect import duckdb_data_types

        assert duckdb_data_types("/nonexistent/project/dir") == {}


class TestExtractFilterColumns:
    def test_bare_sql_filters(self):
        cols = _extract_filter_columns(["status = 'completed'", "amount > 30"])
        assert cols == {"status", "amount"}

    def test_in_clause(self):
        assert "status" in _extract_filter_columns(["status IN ('a','b')"])

    def test_jinja_dimension(self):
        assert _extract_filter_columns(["{{ Dimension('order__status') }} = 'x'"]) == {"status"}


class TestSanitizeCategoricalFilter:
    _DIMS = {"status", "order_date"}

    def test_categorical_to_jinja(self):
        out = _sanitize_filter("status = 'completed'", "orders", "order", self._DIMS)
        assert out == "{{ Dimension('order__status') }} = 'completed'"

    def test_inequality_and_in(self):
        assert _sanitize_filter("status != 'returned'", "orders", "order", self._DIMS) == \
            "{{ Dimension('order__status') }} != 'returned'"
        assert _sanitize_filter("status IN ('a','b')", "orders", "order", self._DIMS) == \
            "{{ Dimension('order__status') }} IN ('a','b')"

    def test_non_dimension_left_alone(self):
        # A measure threshold is not a categorical dimension filter — leave untouched.
        assert _sanitize_filter("amount > 30", "orders", "order", self._DIMS) == "amount > 30"

    def test_numeric_threshold_on_registered_dimension(self):
        # A numeric column registered as a dimension (e.g. a precomputed customer column the
        # model put in base_measures but uses as a row-level threshold) must be rewritten to
        # the Jinja Dimension form, with the numeric literal left UNquoted.
        dims = {"customer_lifetime_value", "number_of_orders"}
        assert _sanitize_filter("customer_lifetime_value > 30", "customers", "customer", dims) == \
            "{{ Dimension('customer__customer_lifetime_value') }} > 30"
        assert _sanitize_filter("number_of_orders >= 2", "customers", "customer", dims) == \
            "{{ Dimension('customer__number_of_orders') }} >= 2"
        assert _sanitize_filter("coupon_amount > 0", "orders", "order", {"coupon_amount"}) == \
            "{{ Dimension('order__coupon_amount') }} > 0"

    def test_no_entity_or_dims_no_conversion(self):
        assert _sanitize_filter("status = 'x'", "orders", None, None) == "status = 'x'"

    def test_subquery_filter_is_flattened(self):
        out = _sanitize_filter(
            "order_id IN (SELECT order_id FROM orders WHERE status = 'completed')",
            "orders", "order", self._DIMS,
        )
        assert out == "{{ Dimension('order__status') }} = 'completed'"

    def test_enum_value_hyphen_normalized(self):
        out = _sanitize_filter("status IN ('returned', 'return-pending')", "orders", "order", self._DIMS)
        assert out == "{{ Dimension('order__status') }} IN ('returned', 'return_pending')"


class TestNormalizeEnumLiterals:
    def test_hyphen_to_underscore(self):
        assert _normalize_enum_literals("'return-pending'") == "'return_pending'"
        assert _normalize_enum_literals("('a-b', 'c d')") == "('a_b', 'c_d')"

    def test_date_literal_preserved(self):
        # Critical: must NOT turn a date bound into '2018_04_01'.
        assert _normalize_enum_literals("'2018-04-01'") == "'2018-04-01'"

    def test_unquoted_untouched(self):
        assert _normalize_enum_literals("30") == "30"
