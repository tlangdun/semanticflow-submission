"""Tests for MetricFlow filter translation.

TDD: These tests define the expected behavior for translating abstract filter
concepts (like 'last_30_days') into valid MetricFlow Jinja filter syntax.
"""

import sys


class TestFilterTranslation:
    """Test cases for translating abstract filters to MetricFlow syntax."""

    def test_translate_last_n_days(self):
        """Abstract 'last_30_days' should become valid MetricFlow Jinja syntax."""
        from semanticflow.dbt_integration.filter_utils import translate_filter
        
        result = translate_filter("last_30_days")
        
        # Should produce valid MetricFlow Jinja syntax
        assert "TimeDimension" in result or "metric_time" in result
        assert "30" in result
        assert "day" in result.lower()

    def test_translate_last_n_days_variations(self):
        """Various last_N_days patterns should all be translated."""
        from semanticflow.dbt_integration.filter_utils import translate_filter
        
        test_cases = [
            ("last_7_days", "7"),
            ("last_30_days", "30"),
            ("last_90_days", "90"),
            ("last_365_days", "365"),
        ]
        
        for abstract_filter, expected_num in test_cases:
            result = translate_filter(abstract_filter)
            assert expected_num in result, f"Expected {expected_num} in result for {abstract_filter}"
            assert "{{" in result, f"Expected Jinja syntax for {abstract_filter}"

    def test_translate_last_n_days_inclusive(self):
        """Variations with 'inclusive_of_today' should be handled."""
        from semanticflow.dbt_integration.filter_utils import translate_filter
        
        result = translate_filter("last_30_days_inclusive_of_today")
        
        assert "TimeDimension" in result or "metric_time" in result
        assert "30" in result

    def test_translate_last_month(self):
        """Abstract 'last_month' should become valid syntax."""
        from semanticflow.dbt_integration.filter_utils import translate_filter
        
        result = translate_filter("last_month")
        
        assert "{{" in result
        assert "month" in result.lower() or "30" in result

    def test_translate_last_quarter(self):
        """Abstract 'last_quarter' should become valid syntax."""
        from semanticflow.dbt_integration.filter_utils import translate_filter
        
        result = translate_filter("last_quarter")
        
        assert "{{" in result
        # Quarter is converted to 3 months
        assert "3 months" in result or "quarter" in result.lower()

    def test_passthrough_valid_sql_filter(self):
        """Already valid SQL/Jinja filters should pass through unchanged."""
        from semanticflow.dbt_integration.filter_utils import translate_filter
        
        valid_filter = "{{ TimeDimension('metric_time', 'day') }} >= current_date - interval '30 days'"
        result = translate_filter(valid_filter)
        
        assert result == valid_filter

    def test_passthrough_column_comparison(self):
        """Simple column comparisons should pass through."""
        from semanticflow.dbt_integration.filter_utils import translate_filter
        
        sql_filter = "status = 'completed'"
        result = translate_filter(sql_filter)
        
        assert result == sql_filter

    def test_translate_returns_none_for_empty(self):
        """Empty or None filters should return None."""
        from semanticflow.dbt_integration.filter_utils import translate_filter
        
        assert translate_filter("") is None
        assert translate_filter(None) is None

    def test_is_abstract_filter(self):
        """Test detection of abstract vs concrete filters."""
        from semanticflow.dbt_integration.filter_utils import is_abstract_filter
        
        # Abstract filters
        assert is_abstract_filter("last_30_days") is True
        assert is_abstract_filter("last_7_days") is True
        assert is_abstract_filter("last_month") is True
        assert is_abstract_filter("last_quarter") is True
        assert is_abstract_filter("last_30_days_inclusive_of_today") is True
        
        # Concrete filters
        assert is_abstract_filter("status = 'completed'") is False
        assert is_abstract_filter("{{ TimeDimension('metric_time', 'day') }} >= current_date") is False
        assert is_abstract_filter("order_date >= '2024-01-01'") is False


class TestFilterTranslationWithDimension:
    """Test filter translation with explicit dimension names."""

    def test_translate_with_custom_dimension(self):
        """Should use provided dimension name instead of default."""
        from semanticflow.dbt_integration.filter_utils import translate_filter
        
        result = translate_filter("last_30_days", time_dimension="order_date")
        
        assert "order_date" in result
        assert "30" in result

    def test_translate_with_granularity(self):
        """Should use provided granularity."""
        from semanticflow.dbt_integration.filter_utils import translate_filter
        
        result = translate_filter("last_12_months", time_dimension="order_date", granularity="month")
        
        assert "order_date" in result
        assert "month" in result.lower()


class TestTranslateFilterList:
    """Test translating a list of filters."""

    def test_translate_filter_list(self):
        """Should translate all abstract filters in a list."""
        from semanticflow.dbt_integration.filter_utils import translate_filters
        
        filters = [
            "last_30_days",
            "status = 'completed'",
        ]
        
        result = translate_filters(filters)
        
        assert len(result) == 2
        assert "{{" in result[0]  # Translated
        assert result[1] == "status = 'completed'"  # Passed through

    def test_translate_filter_list_removes_none(self):
        """Should remove None/empty filters from list."""
        from semanticflow.dbt_integration.filter_utils import translate_filters
        
        filters = ["last_30_days", "", None, "status = 'active'"]
        
        result = translate_filters(filters)
        
        assert len(result) == 2
        assert all(f is not None and f != "" for f in result)


class TestMetricFlowFilterSyntax:
    """Test that generated filters are valid MetricFlow syntax."""

    def test_generated_filter_has_jinja_delimiters(self):
        """MetricFlow filters must use Jinja {{ }} syntax."""
        from semanticflow.dbt_integration.filter_utils import translate_filter
        
        result = translate_filter("last_30_days")
        
        assert "{{" in result
        assert "}}" in result

    def test_generated_filter_uses_time_dimension(self):
        """MetricFlow time filters should use TimeDimension macro."""
        from semanticflow.dbt_integration.filter_utils import translate_filter
        
        result = translate_filter("last_30_days")
        
        # Should use TimeDimension or Dimension macro
        assert "TimeDimension" in result or "Dimension" in result

    def test_generated_filter_is_valid_sql_predicate(self):
        """The generated filter should be a valid SQL WHERE predicate."""
        from semanticflow.dbt_integration.filter_utils import translate_filter
        
        result = translate_filter("last_30_days")
        
        # Should contain a comparison operator
        assert any(op in result for op in [">=", "<=", ">", "<", "=", "BETWEEN"])


class TestDimensionNormalization:
    """Test normalization of bare column names to metric_time."""

    def test_normalize_bare_column_name(self):
        """Bare column names like 'order_date' should become 'metric_time'."""
        from semanticflow.dbt_integration.filter_utils import translate_filter
        
        llm_filter = "{{ TimeDimension('order_date', 'day') }} >= current_date - interval '30 days'"
        result = translate_filter(llm_filter)
        
        assert "metric_time" in result
        assert "order_date" not in result

    def test_metric_time_unchanged(self):
        """metric_time should stay unchanged."""
        from semanticflow.dbt_integration.filter_utils import translate_filter
        
        filter_str = "{{ TimeDimension('metric_time', 'day') }} >= current_date"
        result = translate_filter(filter_str)
        
        assert result == filter_str

    def test_entity_dimension_unchanged(self):
        """entity__dimension format should stay unchanged."""
        from semanticflow.dbt_integration.filter_utils import translate_filter
        
        filter_str = "{{ TimeDimension('order__order_date', 'day') }} >= current_date"
        result = translate_filter(filter_str)
        
        assert result == filter_str


def run_tests():
    """Run all tests without pytest."""
    import traceback
    
    test_classes = [
        TestFilterTranslation,
        TestFilterTranslationWithDimension,
        TestTranslateFilterList,
        TestMetricFlowFilterSyntax,
        TestDimensionNormalization,
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
