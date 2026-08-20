"""
Tests for the LLM-powered measure/column mapper.

Tests cover:
- Successful mapping with valid LLM responses
- Validation of LLM output against schema
- Error handling for malformed responses
- Edge cases (empty inputs, invalid columns, low confidence)
- Mock LLM client behavior
"""
import pytest
import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add parent to path to import llm_mapper directly without triggering other imports
sys.path.insert(0, str(Path(__file__).parent.parent / "semanticflow" / "agents"))

# Import directly to avoid orchestrator imports
import importlib.util
spec = importlib.util.spec_from_file_location(
    "llm_mapper",
    Path(__file__).parent.parent / "semanticflow" / "agents" / "llm_mapper.py"
)
llm_mapper = importlib.util.module_from_spec(spec)

# Mock the dependencies ONLY while loading llm_mapper as a standalone module, then
# restore sys.modules so other test files still import the real packages (otherwise the
# MagicMock leaks for the whole pytest session and breaks e.g. semanticflow.llm.cache).
_saved_modules = {k: sys.modules.get(k) for k in ("semanticflow.llm", "semanticflow.log")}
sys.modules["semanticflow.llm"] = MagicMock()
sys.modules["semanticflow.log"] = MagicMock()
sys.modules["semanticflow.log"].get_logger = MagicMock(return_value=MagicMock())

spec.loader.exec_module(llm_mapper)

for _name, _mod in _saved_modules.items():
    if _mod is None:
        sys.modules.pop(_name, None)
    else:
        sys.modules[_name] = _mod

MeasureMapping = llm_mapper.MeasureMapping
MeasureMappingResult = llm_mapper.MeasureMappingResult
EntityMapping = llm_mapper.EntityMapping
MockLLMClient = llm_mapper.MockLLMClient
llm_map_measure = llm_mapper.llm_map_measure
sync_map_measure = llm_mapper.sync_map_measure
_validate_mapping = llm_mapper._validate_mapping
VALID_AGGREGATIONS = llm_mapper.VALID_AGGREGATIONS
MIN_CONFIDENCE_THRESHOLD = llm_mapper.MIN_CONFIDENCE_THRESHOLD


# ============================================================================
# Test MeasureMapping model
# ============================================================================

class TestMeasureMapping:
    """Tests for MeasureMapping pydantic model."""
    
    def test_valid_mapping(self):
        """Test creating a valid mapping."""
        mapping = MeasureMapping(
            column="amount",
            aggregation="sum",
            confidence=0.9,
            reasoning="Amount is a numeric column suitable for summing",
        )
        assert mapping.column == "amount"
        assert mapping.aggregation == "sum"
        assert mapping.confidence == 0.9
    
    def test_aggregation_normalization(self):
        """Test that aggregation types are normalized."""
        # avg -> average
        mapping = MeasureMapping(column="x", aggregation="avg")
        assert mapping.aggregation == "average"
        
        # AVG (uppercase) -> average
        mapping = MeasureMapping(column="x", aggregation="AVG")
        assert mapping.aggregation == "average"
        
        # mean -> average
        mapping = MeasureMapping(column="x", aggregation="mean")
        assert mapping.aggregation == "average"
    
    def test_invalid_aggregation_defaults_to_sum(self):
        """Test that unknown aggregations default to sum."""
        mapping = MeasureMapping(column="x", aggregation="unknown_agg")
        assert mapping.aggregation == "sum"
    
    def test_confidence_bounds(self):
        """Test confidence is bounded 0-1."""
        with pytest.raises(ValueError):
            MeasureMapping(column="x", aggregation="sum", confidence=1.5)
        
        with pytest.raises(ValueError):
            MeasureMapping(column="x", aggregation="sum", confidence=-0.1)


class TestMeasureMappingResult:
    """Tests for MeasureMappingResult wrapper."""
    
    def test_success_result_is_truthy(self):
        """Test that successful results are truthy."""
        result = MeasureMappingResult(
            mapping=MeasureMapping(column="x", aggregation="sum"),
            success=True,
        )
        assert bool(result) is True
    
    def test_failed_result_is_falsy(self):
        """Test that failed results are falsy."""
        result = MeasureMappingResult(error="No LLM client")
        assert bool(result) is False
    
    def test_result_with_no_mapping_is_falsy(self):
        """Test that result without mapping is falsy even if success=True."""
        result = MeasureMappingResult(success=True, mapping=None)
        assert bool(result) is False


# ============================================================================
# Test validation
# ============================================================================

class TestValidateMapping:
    """Tests for _validate_mapping function."""
    
    def test_valid_mapping(self):
        """Test validation passes for valid mapping."""
        mapping = MeasureMapping(column="amount", aggregation="sum", confidence=0.9)
        columns = ["id", "amount", "status"]
        
        is_valid, error = _validate_mapping(mapping, columns, "revenue")
        
        assert is_valid is True
        assert error == ""
    
    def test_column_not_in_schema(self):
        """Test validation fails when column not in schema."""
        mapping = MeasureMapping(column="nonexistent", aggregation="sum", confidence=0.9)
        columns = ["id", "amount", "status"]
        
        is_valid, error = _validate_mapping(mapping, columns, "revenue")
        
        assert is_valid is False
        assert "not found in schema" in error
    
    def test_case_insensitive_column_match(self):
        """Test that column matching is case-insensitive."""
        mapping = MeasureMapping(column="AMOUNT", aggregation="sum", confidence=0.9)
        columns = ["id", "amount", "status"]
        
        is_valid, error = _validate_mapping(mapping, columns, "revenue")
        
        assert is_valid is True
        # Column name should be normalized to actual case
        assert mapping.column == "amount"
    
    def test_low_confidence_fails(self):
        """Test validation fails for low confidence."""
        mapping = MeasureMapping(column="amount", aggregation="sum", confidence=0.1)
        columns = ["id", "amount", "status"]
        
        is_valid, error = _validate_mapping(mapping, columns, "revenue")
        
        assert is_valid is False
        assert "below threshold" in error


# ============================================================================
# Test MockLLMClient
# ============================================================================

class TestMockLLMClient:
    """Tests for MockLLMClient."""
    
    def test_returns_configured_response(self):
        """Test mock returns configured response for measure concept."""
        mock = MockLLMClient(responses={
            "order_count": MeasureMapping(
                column="number_of_orders",
                aggregation="sum",
                confidence=0.95,
                reasoning="Pre-aggregated count column",
            )
        })
        
        result = mock.complete_json(
            system="test",
            prompt='Measure concept: "order_count"\nModel: customers',
            response_model=MeasureMapping,
        )
        
        assert result.column == "number_of_orders"
        assert result.aggregation == "sum"
    
    def test_returns_default_for_unknown_concept(self):
        """Test mock returns default response for unknown concept."""
        mock = MockLLMClient()
        
        result = mock.complete_json(
            system="test",
            prompt='Measure concept: "unknown_measure"\nModel: test',
            response_model=MeasureMapping,
        )
        
        assert result.column == "id"
        assert result.aggregation == "count"
    
    def test_tracks_calls(self):
        """Test mock tracks all calls made."""
        mock = MockLLMClient()
        
        mock.complete_json("sys1", "prompt1", MeasureMapping)
        mock.complete_json("sys2", "prompt2", MeasureMapping)
        
        assert len(mock.calls) == 2
        assert mock.calls[0]["system"] == "sys1"
        assert mock.calls[1]["prompt"] == "prompt2"


# ============================================================================
# Test async llm_map_measure
# ============================================================================

class TestLlmMapMeasure:
    """Tests for llm_map_measure async function."""
    
    def _run_async(self, coro):
        """Helper to run async functions in sync tests."""
        # asyncio.run creates a fresh loop per call; the old get_event_loop() reused a
        # (possibly closed) shared loop and failed when the suite ran multiple async tests.
        return asyncio.run(coro)
    
    def test_returns_error_for_no_client(self):
        """Test returns error when no LLM client provided."""
        result = self._run_async(llm_map_measure(
            measure_concept="revenue",
            columns=["amount"],
            model_name="orders",
            llm_client=None,
        ))
        
        assert result.success is False
        assert "No LLM client" in result.error
    
    def test_returns_error_for_empty_concept(self):
        """Test returns error for empty measure concept."""
        mock = MockLLMClient()
        
        result = self._run_async(llm_map_measure(
            measure_concept="",
            columns=["amount"],
            model_name="orders",
            llm_client=mock,
        ))
        
        assert result.success is False
        assert "Empty measure concept" in result.error
    
    def test_returns_error_for_no_columns(self):
        """Test returns error when no columns provided."""
        mock = MockLLMClient()
        
        result = self._run_async(llm_map_measure(
            measure_concept="revenue",
            columns=[],
            model_name="orders",
            llm_client=mock,
        ))
        
        assert result.success is False
        assert "No columns" in result.error
    
    def test_successful_mapping(self):
        """Test successful measure mapping."""
        mock = MockLLMClient(responses={
            "revenue": MeasureMapping(
                column="amount",
                aggregation="sum",
                confidence=0.95,
                reasoning="Amount column for revenue",
            )
        })
        
        result = self._run_async(llm_map_measure(
            measure_concept="revenue",
            columns=["id", "amount", "status"],
            model_name="orders",
            llm_client=mock,
        ))
        
        assert result.success is True
        assert result.mapping is not None
        assert result.mapping.column == "amount"
        assert result.mapping.aggregation == "sum"
        assert result.latency_ms > 0
    
    def test_validates_column_exists(self):
        """Test that result is validated against schema."""
        mock = MockLLMClient(responses={
            "revenue": MeasureMapping(
                column="nonexistent_column",
                aggregation="sum",
                confidence=0.9,
            )
        })
        
        result = self._run_async(llm_map_measure(
            measure_concept="revenue",
            columns=["id", "amount"],
            model_name="orders",
            llm_client=mock,
        ))
        
        assert result.success is False
        assert "not found in schema" in result.error
    
    def test_rejects_low_confidence(self):
        """Test that low confidence results are rejected."""
        mock = MockLLMClient(responses={
            "revenue": MeasureMapping(
                column="amount",
                aggregation="sum",
                confidence=0.2,  # Below threshold
            )
        })
        
        result = self._run_async(llm_map_measure(
            measure_concept="revenue",
            columns=["id", "amount"],
            model_name="orders",
            llm_client=mock,
        ))
        
        assert result.success is False
        assert "below threshold" in result.error
    
    def test_handles_llm_exception(self):
        """Test graceful handling of LLM exceptions."""
        mock = MagicMock()
        mock.complete_json.side_effect = Exception("API timeout")
        
        result = self._run_async(llm_map_measure(
            measure_concept="revenue",
            columns=["amount"],
            model_name="orders",
            llm_client=mock,
        ))
        
        assert result.success is False
        assert "API timeout" in result.error


# ============================================================================
# Test sync_map_measure
# ============================================================================

class TestSyncMapMeasure:
    """Tests for sync_map_measure wrapper."""
    
    def test_returns_mapping_on_success(self):
        """Test returns MeasureMapping on success."""
        mock = MockLLMClient(responses={
            "revenue": MeasureMapping(
                column="amount",
                aggregation="sum",
                confidence=0.9,
            )
        })
        
        result = sync_map_measure(
            measure_concept="revenue",
            columns=["id", "amount"],
            model_name="orders",
            llm_client=mock,
        )
        
        assert result is not None
        assert result.column == "amount"
        assert result.aggregation == "sum"
    
    def test_returns_none_on_failure(self):
        """Test returns None when mapping fails."""
        result = sync_map_measure(
            measure_concept="revenue",
            columns=["amount"],
            model_name="orders",
            llm_client=None,  # No client
        )
        
        assert result is None
    
    def test_returns_none_for_invalid_column(self):
        """Test returns None when LLM returns invalid column."""
        mock = MockLLMClient(responses={
            "revenue": MeasureMapping(
                column="nonexistent",
                aggregation="sum",
                confidence=0.9,
            )
        })
        
        result = sync_map_measure(
            measure_concept="revenue",
            columns=["id", "amount"],
            model_name="orders",
            llm_client=mock,
        )
        
        assert result is None


# ============================================================================
# Integration tests with designer
# ============================================================================
# Note: These tests are skipped if designer module cannot be imported
# Run with full environment to test integration

@pytest.mark.skipif(
    "langgraph" not in sys.modules and True,  # Skip if deps not available
    reason="Designer integration tests require full environment"
)
class TestDesignerIntegration:
    """Test integration with designer's _measure_for function.
    
    These tests require the full environment with langgraph installed.
    Run with: pytest tests/test_llm_mapper.py -v -k "not Integration"
    to skip these if deps are not available.
    """
    pass  # Integration tests moved to separate file or run with full env
