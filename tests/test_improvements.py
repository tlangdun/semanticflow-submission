"""Tests for the executability/HITL improvement set: violation->question templates,
post-execution clarifying questions, the LLM sim human, aggregation scoring, design
pre-flight, and the cheap-client builder."""
from __future__ import annotations


class TestViolationsToQuestions:
    def test_known_violations_become_slot_targeted_questions(self):
        from semanticflow.evaluation.spec_verification import violations_to_questions

        violations = [
            "request implies a ratio/percentage (numerator/denominator) but the spec "
            "encodes a single measure — a ratio metric is needed",
            "base_measure 'ordr_cnt' is not a known schema column",
            "group_by 'cust' is not a known schema column",
            "request names a time grain but time_granularity is empty",
            "request uses relative time on a static warehouse; the filter may return no "
            "rows (confirm the data's date range with the user)",
        ]
        qs = violations_to_questions(violations)
        assert len(qs) == 5
        assert all(q.endswith("?") for q in qs)
        assert "ordr_cnt" in qs[1] and "measure" in qs[1].lower()
        assert "cust" in qs[2] and "group" in qs[2].lower()
        assert "granularity" in qs[3].lower()
        assert "date range" in qs[4].lower()

    def test_questions_are_answerable_by_keyword_sim_human(self):
        """Every templated question must hit a sim-human slot keyword — the exact
        brittleness (diagnostic wording vs. slot matching) the templates exist to fix."""
        from semanticflow.evaluation.sim_human import _SLOT_KEYWORDS
        from semanticflow.evaluation.spec_verification import violations_to_questions

        violations = [
            "base_measure 'x' is not a known schema column",
            "group_by 'y' is not a known schema column",
            "request names a time grain but time_granularity is empty",
            "request uses relative time on a static warehouse; the filter may return no "
            "rows (confirm the data's date range with the user)",
        ]
        for q in violations_to_questions(violations):
            ql = q.lower()
            assert any(
                kw in ql for kws in _SLOT_KEYWORDS.values() for kw in kws
            ), f"question not answerable by keyword sim human: {q}"

    def test_duplicate_violations_dedup(self):
        from semanticflow.evaluation.spec_verification import violations_to_questions

        violations = [
            "request names a time grain but time_granularity is empty",
            "request names a time grain but time_granularity is empty",
        ]
        assert len(violations_to_questions(violations)) == 1

    def test_unknown_violation_falls_back_to_confirm(self):
        from semanticflow.evaluation.spec_verification import violations_to_questions

        qs = violations_to_questions(["something novel happened"])
        assert qs == ["Please confirm: something novel happened"]


class TestPostExecutionQuestions:
    def test_unknown_metric_asks_for_measure(self):
        from semanticflow.agents.execution_repair import clarifying_questions_from_error

        qs = clarifying_questions_from_error(
            "x does not exactly match any known metrics. Suggestions: [revenue]"
        )
        assert len(qs) == 1 and "measure" in qs[0].lower()

    def test_empty_result_asks_for_date_range(self):
        from semanticflow.agents.execution_repair import clarifying_questions_from_error

        qs = clarifying_questions_from_error("")
        assert len(qs) == 1 and "date range" in qs[0].lower()

    def test_generic_error_asks_measure_and_range(self):
        from semanticflow.agents.execution_repair import clarifying_questions_from_error

        qs = clarifying_questions_from_error("Some unrecognised hard failure occurred")
        assert len(qs) == 2

    def test_questions_answerable_by_keyword_sim_human(self):
        from semanticflow.agents.execution_repair import clarifying_questions_from_error
        from semanticflow.evaluation.sim_human import _SLOT_KEYWORDS

        for err in (
            "x does not exactly match any known metrics. Suggestions: [a]",
            "No valid join paths exist (fan-out join support is pending)",
            "",
            "weird failure",
        ):
            for q in clarifying_questions_from_error(err):
                ql = q.lower()
                assert any(kw in ql for kws in _SLOT_KEYWORDS.values() for kw in kws)


class TestSimHumanAggregationSlot:
    EXP = {"base_measure": "amount", "aggregation": "average", "group_by": ["order_date"]}

    def test_aggregation_question_reveals_aggregation(self):
        from semanticflow.evaluation.sim_human import scoped_human_answer

        a = scoped_human_answer("Should the amount be summed, or average per order?", self.EXP)
        assert "average" in a.lower()

    def test_no_aggregation_in_gold_reveals_nothing_extra(self):
        from semanticflow.evaluation.sim_human import scoped_human_answer

        exp = {"base_measure": "amount", "group_by": ["order_date"]}
        a = scoped_human_answer("sum or average?", exp)
        assert "aggregation" not in a.lower()


class TestSimHumanBudgetPersistence:
    EXP = {"base_measure": "order_count", "group_by": ["order_date"],
           "time_granularity": "day"}

    def test_budget_persists_across_calls(self):
        from semanticflow.evaluation.sim_human import make_simulated_human

        provider = make_simulated_human(self.EXP, budget=1)
        first = provider(["What measure should I use?"])
        assert any(v for v in first.values())
        # Second consultation (e.g. post-execution touchpoint): budget already spent.
        second = provider(["Which dimension should I group by?"])
        assert all(v == "" for v in second.values())

    def test_repeated_reveal_costs_no_budget(self):
        from semanticflow.evaluation.sim_human import make_simulated_human

        # Two questions routed to the same slot reveal the same fact; only the first
        # spends budget, so a later question about a NEW slot is still answered.
        provider = make_simulated_human(self.EXP, budget=2)
        answers = provider([
            "What measure should I use?",
            "Which measure do you want?",
            "Which dimension should I group by?",
        ])
        vals = list(answers.values())
        assert vals[0] == vals[1] != ""
        assert "order_date" in vals[2]


class _FakeClient:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def complete(self, system_prompt, user_prompt, temperature=None, json_mode=False):
        self.calls.append(user_prompt)
        return self.reply


class TestLLMSimHuman:
    EXP = {"base_measure": "order_count", "group_by": ["order_date"]}

    def test_llm_answer_counts_against_budget(self):
        from semanticflow.evaluation.sim_human import make_llm_simulated_human

        client = _FakeClient("Group by the order date.")
        provider = make_llm_simulated_human(self.EXP, client, budget=1, batch=False)
        first = provider(["Which dimension should I group by?"])
        assert list(first.values()) == ["Group by the order date."]
        second = provider(["What measure?"])
        assert all(v == "" for v in second.values())
        assert len(client.calls) == 1  # budget-exhausted questions never hit the API

    def test_verbatim_repeat_costs_no_budget(self):
        from semanticflow.evaluation.sim_human import make_llm_simulated_human

        client = _FakeClient("Group by the order date.")
        provider = make_llm_simulated_human(self.EXP, client, budget=2, batch=False)
        provider(["Which dimension should I group by?"])
        provider(["What should the breakdown be?"])  # identical reply: free
        client.reply = "Use the order_count measure."
        ans = provider(["What measure should I use?"])
        assert any(v for v in ans.values())  # second budget unit still available

    def test_idk_costs_no_budget(self):
        from semanticflow.evaluation.sim_human import make_llm_simulated_human

        client = _FakeClient("I don't know.")
        provider = make_llm_simulated_human(self.EXP, client, budget=1, batch=False)
        provider(["Something unintelligible?"])
        client.reply = "Use the order_count measure."
        ans = provider(["What measure should I use?"])
        assert any(v for v in ans.values())  # budget still available

    def test_client_failure_degrades_to_unanswered(self):
        from semanticflow.evaluation.sim_human import llm_human_answer

        class Boom:
            def complete(self, *a, **k):
                raise RuntimeError("api down")

        assert llm_human_answer("What measure?", self.EXP, Boom()) == ""

    def test_factory_falls_back_to_keyword_without_client(self):
        from semanticflow.evaluation.sim_human import make_sim_human

        provider = make_sim_human(self.EXP, budget=3, mode="llm", llm_client=None)
        ans = provider(["What measure should I use?"])
        assert "order_count" in next(iter(ans.values()))


class TestAggregationScoring:
    def test_wrong_aggregation_lowers_spec_score(self):
        from semanticflow.agents.evaluator import _score_against_spec

        expected = {"base_measure": "amount", "aggregation": "average"}
        right = _score_against_spec(
            {"base_measures": ["amount"], "aggregation": "average"}, expected
        )
        wrong = _score_against_spec(
            {"base_measures": ["amount"], "aggregation": "sum"}, expected
        )
        assert right["aggregation_match"] == 1.0
        assert wrong["aggregation_match"] == 0.0
        assert wrong["spec_field_score"] < right["spec_field_score"]

    def test_missing_predicted_aggregation_scores_zero(self):
        from semanticflow.agents.evaluator import _score_against_spec

        s = _score_against_spec(
            {"base_measures": ["amount"]}, {"base_measure": "amount", "aggregation": "sum"}
        )
        assert s["aggregation_match"] == 0.0

    def test_no_gold_aggregation_keeps_three_field_mean(self):
        from semanticflow.agents.evaluator import _score_against_spec

        s = _score_against_spec(
            {"base_measures": ["amount"], "aggregation": "sum"},
            {"base_measure": "amount"},
        )
        assert s["aggregation_match"] is None
        assert s["spec_field_score"] == (s["measures_f1"] + s["dimensions_f1"] + s["filters_f1"]) / 3.0

    def test_agg_synonyms_match(self):
        from semanticflow.agents.evaluator import _score_against_spec

        s = _score_against_spec(
            {"base_measures": ["amount"], "aggregation": "avg"},
            {"base_measure": "amount", "aggregation": "average"},
        )
        assert s["aggregation_match"] == 1.0


class TestPreflightDesign:
    SCHEMA = {"models": ["orders"], "columns": {"orders": ["order_id", "amount", "order_date"]}}

    def _design(self, **overrides):
        design = {
            "semantic_models": [
                {
                    "name": "orders",
                    "model": "orders",
                    "measures": [{"name": "amount", "agg": "sum", "expr": "amount"}],
                    "dimensions": [
                        {"name": "order_date", "type": "time", "expr": "order_date"}
                    ],
                }
            ],
            "metrics": [
                {"name": "revenue", "type": "simple", "type_params": {"measure": "amount"}}
            ],
        }
        design.update(overrides)
        return design

    def test_clean_design_passes(self):
        from semanticflow.evaluation.preflight import preflight_design

        assert preflight_design(self._design(), self.SCHEMA) == []

    def test_no_metric_flagged(self):
        from semanticflow.evaluation.preflight import preflight_design

        violations = preflight_design(self._design(metrics=[]), self.SCHEMA)
        assert any("no metric emitted" in v for v in violations)

    def test_unknown_measure_column_flagged(self):
        from semanticflow.evaluation.preflight import preflight_design

        design = self._design()
        design["semantic_models"][0]["measures"][0]["expr"] = "ghost_col"
        violations = preflight_design(design, self.SCHEMA)
        assert any("ghost_col" in v for v in violations)

    def test_metric_referencing_missing_measure_flagged(self):
        from semanticflow.evaluation.preflight import preflight_design

        design = self._design()
        design["metrics"][0]["type_params"]["measure"] = "nope"
        violations = preflight_design(design, self.SCHEMA)
        assert any("'nope'" in v for v in violations)

    def test_synthetic_ds_dimension_skipped(self):
        from semanticflow.evaluation.preflight import preflight_design

        design = self._design()
        design["semantic_models"][0]["dimensions"].append(
            {"name": "ds", "type": "time", "expr": "cast('2018-01-01' as date)"}
        )
        assert preflight_design(design, self.SCHEMA) == []


class TestCheapClient:
    def test_no_usable_provider_returns_none(self):
        from semanticflow.config import Settings
        from semanticflow.llm import build_cheap_client

        settings = Settings(llm_provider="mock", ensemble_providers=["mock"])
        settings.openai_api_key = None
        settings.anthropic_api_key = None
        settings.gemini_api_key = None
        assert build_cheap_client(settings) is None


class TestStructuredFilterCoercion:
    """Frontier-tier mappers and the repair LLM return filters as dicts; every dict
    shape must land on the canonical string form (the crash that aborted the first
    frontier run: designer .strip() on a `between` dict)."""

    def test_between_dict_expands_to_range(self):
        from semanticflow.dbt_integration.filter_utils import coerce_filters_to_strings

        out = coerce_filters_to_strings(
            [{"field": "order_date", "operator": "between",
              "values": ["2018-01-01", "2018-04-09"]}]
        )
        assert out == ["order_date >= '2018-01-01'", "order_date <= '2018-04-09'"]

    def test_eq_dict_with_singular_value(self):
        from semanticflow.dbt_integration.filter_utils import coerce_filters_to_strings

        out = coerce_filters_to_strings([{"field": "status", "operator": "eq", "value": "completed"}])
        assert out == ["status = 'completed'"]

    def test_in_dict_renders_value_list(self):
        from semanticflow.dbt_integration.filter_utils import coerce_filters_to_strings

        out = coerce_filters_to_strings(
            [{"field": "status", "operator": "in", "values": ["completed", "shipped"]}]
        )
        assert out == ["status in ('completed', 'shipped')"]

    def test_numbers_unquoted_strings_passthrough(self):
        from semanticflow.dbt_integration.filter_utils import coerce_filters_to_strings

        out = coerce_filters_to_strings(
            ["amount > 30", {"field": "amount", "operator": ">", "value": 30}]
        )
        assert out == ["amount > 30", "amount > 30"]

    def test_fieldless_dict_dropped(self):
        from semanticflow.dbt_integration.filter_utils import coerce_filters_to_strings

        assert coerce_filters_to_strings([{"operator": "=", "value": "x"}]) == []

    def test_llm_repair_merge_coerces_filters(self):
        from semanticflow.agents.execution_repair import _repair_with_llm

        class Client:
            def complete(self, *a, **k):
                return ('{"filters": [{"field": "order_date", "operator": "between", '
                        '"values": ["2018-01-01", "2018-04-09"]}]}')

        spec = {"metric_name": "m", "filters": ["old"]}
        out = _repair_with_llm(spec, "err", {}, "req", Client())
        assert all(isinstance(f, str) for f in out["filters"])
        assert "order_date >= '2018-01-01'" in out["filters"]

    def test_designer_survives_dict_filters(self):
        import asyncio
        from semanticflow.agents.designer import DesignerAgent
        from semanticflow.agents.types import AgentMessage
        from semanticflow.tasks.loader import Task

        task = Task(
            task_id="t", nl_request="orders in march",
            schema_context={"models": ["orders"],
                            "columns": {"orders": ["order_id", "order_date", "amount"]}},
            expected_semantic={},
        )
        spec = {"metric_name": "orders", "base_measures": ["order_id"],
                "aggregation": "count", "group_by": ["order_date"],
                "filters": [{"field": "order_date", "operator": "between",
                             "values": ["2018-03-01", "2018-03-31"]}]}
        agent = DesignerAgent(llm_client=None)
        msg = asyncio.run(agent.run(AgentMessage(task=task, state={"semantic_spec": spec})))
        assert msg.state["design_proposals"]["metrics"]  # no crash, metric emitted


class TestRealisticHumanFixes:
    """The interaction fixes that let a realistic free-form human recover the clarification
    gain: question prioritization, batched answering, and the human-aware slot guard."""

    def test_prioritize_questions_puts_structural_slots_first(self):
        from semanticflow.agents.ambiguity_analyzer import prioritize_questions

        qs = [
            "Should I include all orders regardless of status, or only completed ones?",
            "How should the results be sorted?",
            "Which metric would you like: total revenue or count of orders?",
            "Should the result be broken down by a dimension, or a single total?",
            "What time granularity: daily, weekly, monthly?",
        ]
        out = prioritize_questions(qs)
        assert "metric" in out[0].lower()            # measure first
        assert "broken down" in out[1].lower()       # then grouping
        assert "granularity" in out[2].lower()       # then grain
        assert "sorted" in out[-1].lower()           # sorting last

    def test_clarified_slots_detects_addressed_slots(self):
        from semanticflow.agents.ambiguity_analyzer import clarified_slots

        ua = {
            "Which metric?": "the number of orders",
            "Broken down by what dimension?": "by customer",
            "What granularity?": "daily",
            "Sort order?": "",  # empty answer -> not clarified
        }
        s = clarified_slots(ua)
        assert {"base_measures", "aggregation", "group_by", "time_granularity"} <= s

    def test_guard_keeps_human_clarified_slot(self):
        from semanticflow.agents.semantic_mapper import guard_agreed_slots

        consensus = {"base_measures": ["amount"]}
        refined = {"base_measures": ["order_count"]}  # human corrected to a count
        outputs = {p: {"base_measures": ["amount"]} for p in ("a", "b", "c")}
        # Without clarification the unanimous (wrong) default is protected -> reverted.
        g1, rev1 = guard_agreed_slots(consensus, refined, outputs)
        assert rev1 == ["base_measures"] and g1["base_measures"] == ["amount"]
        # When the human explicitly clarified the measure, it overrides agreement -> kept.
        g2, rev2 = guard_agreed_slots(consensus, refined, outputs,
                                      clarified_slots={"base_measures"})
        assert rev2 == [] and g2["base_measures"] == ["order_count"]

    def test_batched_human_answers_all_in_one_call(self):
        from semanticflow.evaluation.sim_human import make_llm_simulated_human

        class _Batch:
            def __init__(self):
                self.calls = 0

            def complete(self, system, user, temperature=None, json_mode=False):
                self.calls += 1
                return '{"1": "count of orders", "2": "by customer", "3": "daily"}'

        client = _Batch()
        provider = make_llm_simulated_human({"base_measure": "order_count"}, client, batch=True)
        ans = provider(["measure?", "grouping?", "grain?"])
        assert client.calls == 1                       # ONE call for all questions
        assert len([v for v in ans.values() if v]) == 3
