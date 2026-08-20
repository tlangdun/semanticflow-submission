"""Tests for deterministic HITL answer incorporation (refine_with_feedback).

The HITL loop asked clarifying questions, the simulated human answered, but the refined
spec ignored the answers (esp. grouping) and invented fake measure names. These pin the
deterministic parser that applies answer CONTENT to the spec against real schema columns.
"""
from __future__ import annotations

from semanticflow.agents.semantic_mapper import SemanticMapperAgent
from semanticflow.config import Settings

_SCHEMA = {"models": ["orders"],
           "columns": {"orders": ["order_id", "customer_id", "order_date", "amount", "status"]}}


def _agent():
    return SemanticMapperAgent(settings=Settings(), llm_clients={})


def _refine(answers, spec=None):
    spec = spec or {"base_measures": ["amount"], "group_by": [], "time_granularity": None, "filters": []}
    return _agent()._apply_answers_deterministic(spec, answers, _SCHEMA)


class TestAnswerIncorporation:
    def test_grouping_answer_applied(self):
        out, changed = _refine({"q": "Group by order_date."})
        assert changed and out["group_by"] == ["order_date"]

    def test_measure_answer_uses_real_column(self):
        out, _ = _refine({"q": "Use the measure amount."})
        assert out["base_measures"] == ["amount"]  # never a fake 'revenue_sum'

    def test_single_total_clears_grouping(self):
        out, _ = _refine({"q": "Do not group by any dimension; return a single total."},
                         spec={"group_by": ["order_date"], "base_measures": ["amount"]})
        assert out["group_by"] == []

    def test_granularity_implies_time_grouping(self):
        # "Use day granularity" with no explicit group_by must still group by the time column.
        out, _ = _refine({"q": "Use day granularity."})
        assert out["time_granularity"] == "day"
        assert "order_date" in out["group_by"]

    def test_mixed_answer_not_tripped_by_no_filters(self):
        # Regression: "No filters" must not suppress the day-granularity in the same answer.
        out, _ = _refine({"q": "Use day granularity. No filters; include all rows."})
        assert out["time_granularity"] == "day"
        assert out["group_by"] == ["order_date"]
        assert out["filters"] == []

    def test_explicit_no_granularity_is_respected(self):
        out, _ = _refine({"q": "No specific time granularity is needed."})
        assert out["time_granularity"] is None
        assert out["group_by"] == []

    def test_count_and_sum_intent(self):
        assert _refine({"q": "Count the number of orders."})[0]["aggregation"] == "count"
        assert _refine({"q": "Use the total / sum."})[0]["aggregation"] == "sum"

    def test_empty_answers_change_nothing(self):
        out, changed = _refine({"q": ""})
        assert changed is False

    def test_clause_scoping_no_cross_leak(self):
        # Regression: a column in the group-by clause must NOT also become a measure, and
        # vice-versa. ("Use the measure order_id. Group by status." once dumped both into both.)
        out, _ = _refine({"q": "Use the measure order_id. Group by status. No filters; include all rows."},
                         spec={"base_measures": ["order_id"], "group_by": ["status"]})
        assert out["base_measures"] == ["order_id"]
        assert out["group_by"] == ["status"]
        assert out["filters"] == []

    def test_entity_name_resolves_to_key_column(self):
        # Regression real_034: "Group by customer." was silently dropped because
        # 'customer' is not a physical column; it must resolve to customer_id.
        out, changed = _refine({"q": "Use the measure gross_revenue. Group by customer."})
        assert changed
        assert out["group_by"] == ["customer_id"]
        assert out["base_measures"] == ["gross_revenue"]

    def test_mixed_entity_and_column_grouping(self):
        out, _ = _refine({"q": "Group by customer and order_date."})
        assert set(out["group_by"]) == {"customer_id", "order_date"}

    def test_unresolvable_group_token_is_skipped(self):
        out, changed = _refine({"q": "Group by region."})
        assert out["group_by"] == []  # never inserted verbatim

    def test_filter_clause_does_not_pollute_measure(self):
        # Regression real_001: order_date in the FILTER clause must not become the measure.
        out, _ = _refine({"q": "Use the measure order_count. Apply the filter(s): order_date >= '2018-03-01', order_date <= '2018-03-31'."})
        assert out["base_measures"] == ["order_count"]   # not order_date
        assert "order_date >= '2018-03-01'" in out["filters"]
        assert all(not f.endswith(".") for f in out["filters"])  # trailing period stripped

    def test_repeated_filter_reveal_not_stacked(self):
        # Regression real_027: the same filter pair revealed in 4 answers was appended
        # 4x; identical predicates must be applied once.
        ans = "Apply the filter(s): order_date >= '2018-04-01', order_date <= '2018-04-09'."
        out, _ = _refine({"q1": ans, "q2": ans, "q3": ans})
        assert out["filters"] == ["order_date >= '2018-04-01'", "order_date <= '2018-04-09'"]


class TestSlotCoverageBackstop:
    def test_appends_grouping_and_grain_when_uncovered(self):
        from semanticflow.agents.ambiguity_analyzer import (
            GRAIN_BACKSTOP_QUESTION,
            GROUPING_BACKSTOP_QUESTION,
            ensure_slot_coverage,
        )

        qs = ensure_slot_coverage(["Which metric matters most to you?",
                                   "Should I include all orders regardless of status?"])
        assert GROUPING_BACKSTOP_QUESTION in qs
        assert GRAIN_BACKSTOP_QUESTION in qs

    def test_no_duplicates_when_already_covered(self):
        from semanticflow.agents.ambiguity_analyzer import ensure_slot_coverage

        qs = ["Should this be broken down by status, or a single total?",
              "Do you want daily or monthly granularity?"]
        assert ensure_slot_coverage(qs) == qs

    def test_backstops_answerable_by_keyword_human(self):
        from semanticflow.agents.ambiguity_analyzer import (
            GRAIN_BACKSTOP_QUESTION,
            GROUPING_BACKSTOP_QUESTION,
        )
        from semanticflow.evaluation.sim_human import scoped_human_answer

        gold = {"base_measure": "gross_revenue", "group_by": ["order_date"],
                "time_granularity": "day"}
        g = scoped_human_answer(GROUPING_BACKSTOP_QUESTION, gold)
        assert "order_date" in g and "day" in g  # grouping reveal carries the grain
        assert "day" in scoped_human_answer(GRAIN_BACKSTOP_QUESTION, gold)
