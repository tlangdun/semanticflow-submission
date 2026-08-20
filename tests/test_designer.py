"""Tests for the designer agent and related utilities."""
from __future__ import annotations

import sys


class TestEntityInference:
    """Test generic entity inference from schema context."""

    def test_infer_primary_entity_from_orders(self):
        """Should infer order as primary entity from orders table."""
        from semanticflow.agents.designer import _build_entities

        model_columns = {"order_id", "customer_id", "order_date", "amount", "status"}
        entities = _build_entities("orders", model_columns)

        primary = [e for e in entities if e.type == "primary"]
        assert len(primary) == 1
        assert primary[0].name == "order"
        assert primary[0].expr == "order_id"

    def test_infer_foreign_entities(self):
        """Should infer foreign entities from *_id columns."""
        from semanticflow.agents.designer import _build_entities

        model_columns = {"order_id", "customer_id", "product_id", "amount"}
        entities = _build_entities("orders", model_columns)

        foreign = [e for e in entities if e.type == "foreign"]
        assert len(foreign) == 2
        foreign_names = {e.name for e in foreign}
        assert "customer" in foreign_names
        assert "product" in foreign_names

    def test_infer_primary_entity_from_customers(self):
        """Should infer customer as primary entity from customers table."""
        from semanticflow.agents.designer import _build_entities

        model_columns = {"customer_id", "name", "email", "created_at"}
        entities = _build_entities("customers", model_columns)

        primary = [e for e in entities if e.type == "primary"]
        assert len(primary) == 1
        assert primary[0].name == "customer"
        assert primary[0].expr == "customer_id"

    def test_handle_id_column(self):
        """Should use 'id' as primary key if no {model}_id exists."""
        from semanticflow.agents.designer import _build_entities

        model_columns = {"id", "name", "value"}
        entities = _build_entities("items", model_columns)

        primary = [e for e in entities if e.type == "primary"]
        assert len(primary) == 1
        assert primary[0].expr == "id"


class TestAggregationInference:
    """Test aggregation inference from column names and types."""

    def test_count_for_id_columns(self):
        """Should infer COUNT for *_id columns."""
        from semanticflow.agents.designer import _infer_aggregation

        assert _infer_aggregation("order_id", None) == "count"
        assert _infer_aggregation("id", None) == "count"

    def test_sum_for_amount_columns(self):
        """Should infer SUM for amount/price columns."""
        from semanticflow.agents.designer import _infer_aggregation

        assert _infer_aggregation("amount", None) == "sum"
        assert _infer_aggregation("total_price", None) == "sum"
        assert _infer_aggregation("revenue", None) == "sum"

    def test_sum_for_numeric_types(self):
        """Should infer SUM for numeric data types."""
        from semanticflow.agents.designer import _infer_aggregation

        data_types = {"quantity": "integer", "rating": "decimal"}
        assert _infer_aggregation("quantity", data_types) == "sum"
        assert _infer_aggregation("rating", data_types) == "sum"


class TestTimeColumnDetection:
    """Test detection of time columns."""

    def test_detect_date_columns(self):
        """Should detect date columns by name."""
        from semanticflow.agents.designer import _is_time_column

        assert _is_time_column("order_date", None) is True
        assert _is_time_column("created_at", None) is True
        assert _is_time_column("updated_timestamp", None) is True

    def test_detect_time_by_type(self):
        """Should detect time columns by data type."""
        from semanticflow.agents.designer import _is_time_column

        data_types = {"event_ts": "timestamp", "name": "varchar"}
        assert _is_time_column("event_ts", data_types) is True
        assert _is_time_column("name", data_types) is False

    def test_non_time_columns(self):
        """Should not detect non-time columns."""
        from semanticflow.agents.designer import _is_time_column

        assert _is_time_column("customer_id", None) is False
        assert _is_time_column("amount", None) is False


class TestNamingUtilities:
    """Test metric naming utilities."""

    def test_to_snake_case(self):
        """Should convert various formats to snake_case."""
        from semanticflow.utils.naming import to_snake_case

        assert to_snake_case("OrdersPerDay") == "orders_per_day"
        assert to_snake_case("orders-per-day") == "orders_per_day"
        assert to_snake_case("Orders Per Day") == "orders_per_day"
        assert to_snake_case("orders_per_day") == "orders_per_day"

    def test_normalize_metric_name(self):
        """Should normalize metric names."""
        from semanticflow.utils.naming import normalize_metric_name

        assert normalize_metric_name("Orders Per Day") == "orders_per_day"
        assert normalize_metric_name("revenue", prefix="daily") == "daily_revenue"

    def test_suggest_metric_name(self):
        """Should suggest metric names from components."""
        from semanticflow.utils.naming import suggest_metric_name

        name = suggest_metric_name(["order_count"], ["order_date"])
        assert "order_count" in name
        assert "order_date" in name

        name = suggest_metric_name(["revenue"], ["customer"], ["last_30_days"])
        assert "revenue" in name
        assert "last_30_days" in name


def run_tests():
    """Run all tests without pytest."""
    import traceback

    test_classes = [
        TestEntityInference,
        TestAggregationInference,
        TestTimeColumnDetection,
        TestNamingUtilities,
    ]

    passed = 0
    failed = 0

    for test_class in test_classes:
        print(f"\n=== {test_class.__name__} ===")
        instance = test_class()
        for method_name in dir(instance):
            if method_name.startswith("test_"):
                try:
                    getattr(instance, method_name)()
                    print(f"  ✅ {method_name}")
                    passed += 1
                except Exception as e:
                    print(f"  ❌ {method_name}: {e}")
                    traceback.print_exc()
                    failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
