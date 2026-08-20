"""Tests for new agent features: error recovery, result validation, and caching."""
from __future__ import annotations

import sys


class TestErrorRecovery:
    """Test error recovery and auto-fix capabilities."""

    def test_analyze_missing_column_error(self):
        """Should detect missing column errors."""
        from semanticflow.agents.error_recovery import analyze_errors, ErrorType

        stderr = "Error: column 'order_dates' does not exist"
        analysis = analyze_errors(stderr)

        assert len(analysis.errors) > 0
        assert ErrorType.MISSING_COLUMN in analysis.error_types

    def test_analyze_invalid_dimension_name(self):
        """Should detect invalid dimension name format errors."""
        from semanticflow.agents.error_recovery import analyze_errors, ErrorType

        stderr = "Name is in an incorrect format: 'order_date'. It should be of the form: <primary entity name>__<dimension_name>"
        analysis = analyze_errors(stderr)

        assert len(analysis.errors) > 0
        assert ErrorType.INVALID_DIMENSION_NAME in analysis.error_types

    def test_suggest_similar_column(self):
        """Should suggest similar column names."""
        from semanticflow.agents.error_recovery import analyze_errors

        stderr = "Error: column 'order_dates' does not exist"
        schema_context = {"columns": {"orders": ["order_id", "order_date", "customer_id"]}}
        
        analysis = analyze_errors(stderr, schema_context=schema_context)
        
        # Should have a recovery action with suggestion
        assert len(analysis.recovery_actions) > 0
        action = analysis.recovery_actions[0]
        assert action.fix_details.get("with") == "order_date"

    def test_apply_recovery_replaces_column(self):
        """Should apply column replacement fix."""
        from semanticflow.agents.error_recovery import apply_recovery, ErrorAnalysis, RecoveryAction, ErrorType

        analysis = ErrorAnalysis(
            errors=["Column not found"],
            error_types=[ErrorType.MISSING_COLUMN],
            recovery_actions=[
                RecoveryAction(
                    error_type=ErrorType.MISSING_COLUMN,
                    description="Replace column",
                    auto_fixable=True,
                    fix_details={"replace": "order_dates", "with": "order_date"},
                    confidence=0.8,
                )
            ],
            can_auto_recover=True,
        )
        
        spec = {"group_by": ["order_dates", "customer_id"]}
        updated_spec, _, fixes = apply_recovery(analysis, spec, {})
        
        assert "order_date" in updated_spec["group_by"]
        assert "order_dates" not in updated_spec["group_by"]
        assert len(fixes) > 0


class TestResultValidation:
    """Test query result validation."""

    def test_validate_empty_results(self):
        """Should flag empty results."""
        from semanticflow.agents.result_validator import validate_query_result

        result = validate_query_result([])
        
        assert result.row_count == 0
        assert len(result.issues) > 0
        assert any(i.check_name == "empty_results" for i in result.issues)

    def test_validate_few_results(self):
        """Should note very few results."""
        from semanticflow.agents.result_validator import validate_query_result

        rows = [{"metric": i} for i in range(3)]
        result = validate_query_result(rows)
        
        assert result.row_count == 3
        assert any(i.check_name == "few_results" for i in result.issues)

    def test_validate_null_values(self):
        """Should detect null values in columns."""
        from semanticflow.agents.result_validator import validate_query_result

        rows = [
            {"order_count": 10, "date": "2024-01-01"},
            {"order_count": None, "date": "2024-01-02"},
            {"order_count": 15, "date": None},
        ]
        result = validate_query_result(rows)
        
        assert any(i.check_name == "null_values" for i in result.issues)

    def test_validate_duplicate_rows(self):
        """Should detect duplicate rows."""
        from semanticflow.agents.result_validator import validate_query_result

        rows = [
            {"order_count": 10, "date": "2024-01-01"},
            {"order_count": 10, "date": "2024-01-01"},  # Duplicate
            {"order_count": 15, "date": "2024-01-02"},
        ]
        result = validate_query_result(rows)
        
        assert any(i.check_name == "duplicate_rows" for i in result.issues)

    def test_validate_negative_counts(self):
        """Should flag negative values in count columns."""
        from semanticflow.agents.result_validator import validate_query_result

        rows = [
            {"order_count": 10},
            {"order_count": -5},  # Negative count
            {"order_count": 15},
        ]
        result = validate_query_result(rows)
        
        assert any(i.check_name == "negative_values" for i in result.issues)


class TestLLMCache:
    """Test LLM response caching."""

    def test_cache_miss_then_hit(self):
        """Should cache and retrieve responses."""
        from semanticflow.llm.cache import LLMCache
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = LLMCache(cache_dir=tmpdir, enabled=True)
            
            # First call - cache miss
            result = cache.get("openai", "gpt-4", "system", "user")
            assert result is None
            
            # Store response
            cache.set("openai", "gpt-4", "system", "user", '{"answer": "test"}')
            
            # Second call - cache hit
            result = cache.get("openai", "gpt-4", "system", "user")
            assert result == '{"answer": "test"}'

    def test_cache_stats(self):
        """Should track cache statistics."""
        from semanticflow.llm.cache import LLMCache
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = LLMCache(cache_dir=tmpdir, enabled=True)
            
            # Generate some activity
            cache.get("test", "model", "sys", "user")  # Miss
            cache.set("test", "model", "sys", "user", "response")
            cache.get("test", "model", "sys", "user")  # Hit
            
            stats = cache.stats()
            assert stats["hits"] == 1
            assert stats["misses"] == 1
            assert stats["hit_rate"] == 50.0

    def test_cache_disabled(self):
        """Should not cache when disabled."""
        from semanticflow.llm.cache import LLMCache

        cache = LLMCache(enabled=False)
        
        cache.set("test", "model", "sys", "user", "response")
        result = cache.get("test", "model", "sys", "user")
        
        assert result is None  # Cache disabled


def run_tests():
    """Run all tests without pytest."""
    import traceback

    test_classes = [
        TestErrorRecovery,
        TestResultValidation,
        TestLLMCache,
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
