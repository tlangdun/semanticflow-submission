"""Tests for deterministic order-by tie-breaking under LIMIT.

When a query has a LIMIT, ties at the cutoff (e.g. two customers with the same
lifetime value at rank 10) make the returned row set non-deterministic unless a
secondary sort key is emitted. The executor must append the resolved group-by
dimensions as secondary ascending order keys whenever a limit is set, matching
the conventional gold tie-break (e.g. ORDER BY value DESC, customer_id ASC).
Regression test for task real_002 (top 10 customers by lifetime value).
"""

import sys
from pathlib import Path
from unittest.mock import patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from semanticflow.agents.executor import ExecutorAgent
from semanticflow.config import Settings
from semanticflow.dbt_integration.metricflow_runner import MfCommandResult, run_mf_query


def _make_executor() -> ExecutorAgent:
    return ExecutorAgent(settings=Settings())


# Synthetic design mirroring the real_002 shape: one semantic model whose
# primary entity is `customer`, grouped by the entity itself.
DESIGN = {
    "semantic_models": [
        {
            "name": "customers",
            "primary_entity": "customer",
            "entities": [{"name": "customer", "type": "primary"}],
            "dimensions": [],
        }
    ],
    "metrics": [{"name": "customer_lifetime_value"}],
}


class TestResolveOrderByTiebreak:
    """_resolve_order_by must append group-by dims as secondary asc keys under LIMIT."""

    def test_limit_appends_dims_as_secondary_asc_keys(self):
        """real_002 shape: metric desc + limit -> entity appended as secondary asc key."""
        executor = _make_executor()
        spec = {
            "metric_name": "customer_lifetime_value",
            "group_by": ["customer"],
            "order_by": [{"field": "customer_lifetime_value", "direction": "desc"}],
            "limit": 10,
        }
        result = executor._resolve_order_by(
            spec["order_by"], spec, DESIGN, ["customer"], "customer_lifetime_value",
            limit=10,
        )
        assert result == [
            {"field": "customer_lifetime_value", "direction": "desc"},
            {"field": "customer", "direction": "asc"},
        ]

    def test_no_limit_leaves_order_by_unchanged(self):
        """Without a limit there is no cutoff, so no secondary keys are appended."""
        executor = _make_executor()
        spec = {
            "metric_name": "customer_lifetime_value",
            "group_by": ["customer"],
            "order_by": [{"field": "customer_lifetime_value", "direction": "desc"}],
        }
        result = executor._resolve_order_by(
            spec["order_by"], spec, DESIGN, ["customer"], "customer_lifetime_value",
        )
        assert result == [{"field": "customer_lifetime_value", "direction": "desc"}]

    def test_limit_does_not_duplicate_existing_order_key(self):
        """A dimension already in the order-by must not be appended twice."""
        executor = _make_executor()
        spec = {
            "metric_name": "customer_lifetime_value",
            "group_by": ["customer"],
            "order_by": [{"field": "customer", "direction": "desc"}],
            "limit": 5,
        }
        result = executor._resolve_order_by(
            spec["order_by"], spec, DESIGN, ["customer"], "customer_lifetime_value",
            limit=5,
        )
        assert result == [{"field": "customer", "direction": "desc"}]

    def test_limit_appends_multiple_dims_in_order(self):
        """All resolved group-by dims not already present are appended, in order."""
        executor = _make_executor()
        spec = {
            "metric_name": "revenue",
            "group_by": ["customer", "status"],
            "order_by": [{"field": "revenue", "direction": "desc"}],
            "limit": 3,
        }
        dims = ["customer", "customer__status"]
        result = executor._resolve_order_by(
            spec["order_by"], spec, DESIGN, dims, "revenue", limit=3,
        )
        assert result == [
            {"field": "revenue", "direction": "desc"},
            {"field": "customer", "direction": "asc"},
            {"field": "customer__status", "direction": "asc"},
        ]

    def test_limit_without_order_by_emits_dims_ascending(self):
        """LIMIT with no requested order still gets a deterministic ascending order."""
        executor = _make_executor()
        spec = {
            "metric_name": "customer_lifetime_value",
            "group_by": ["customer"],
            "limit": 10,
        }
        result = executor._resolve_order_by(
            None, spec, DESIGN, ["customer"], "customer_lifetime_value", limit=10,
        )
        assert result == [{"field": "customer", "direction": "asc"}]

    def test_no_order_by_and_no_limit_returns_none(self):
        """Existing behaviour preserved: nothing to order by, nothing emitted."""
        executor = _make_executor()
        spec = {"metric_name": "customer_lifetime_value", "group_by": ["customer"]}
        result = executor._resolve_order_by(
            None, spec, DESIGN, ["customer"], "customer_lifetime_value",
        )
        assert result is None


class TestMfQueryOrderFlagFormat:
    """run_mf_query must emit '-metric' for desc and plain names for asc keys."""

    def test_secondary_asc_key_emitted_in_mf_order_format(self):
        captured_command: list[str] = []

        def mock_run(command, env=None, cwd=None, timeout=None):
            captured_command.extend(command)
            return MfCommandResult(
                command=command, stdout="", stderr="", returncode=0, issues=[],
            )

        settings = Settings(mf_query_output="")
        with patch(
            "semanticflow.dbt_integration.metricflow_runner._run", side_effect=mock_run
        ):
            run_mf_query(
                ".",
                settings,
                "customer_lifetime_value",
                dimensions=["customer"],
                limit=10,
                order_by=[
                    {"field": "customer_lifetime_value", "direction": "desc"},
                    {"field": "customer", "direction": "asc"},
                ],
            )

        assert "--order" in captured_command
        order_value = captured_command[captured_command.index("--order") + 1]
        assert order_value == "-customer_lifetime_value,customer"
