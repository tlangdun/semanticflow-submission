"""Tests for execution-grounded evaluation: column-agnostic result comparison and
the evaluator's execution-vs-spec headline logic (regression for the empty-result
false-positive trap)."""
from __future__ import annotations

import asyncio
import datetime
import sys


class TestColumnAgnosticCompare:
    """compare_results with column_agnostic mode bridges gold (pandas/duckdb) vs
    MetricFlow (text) representations."""

    def _cmp(self, gold, actual, **rules):
        from semanticflow.evaluation.result_compare import compare_results
        rules.setdefault("column_agnostic", True)
        return compare_results(gold, actual, compare_rules=rules)

    def test_matches_despite_renamed_reordered_columns(self):
        gold = [{"order_date": datetime.datetime(2018, 3, 1), "orders": 2}]
        actual = [{"metric_time__day": "2018-03-01", "orders_metric": "2"}]
        assert self._cmp(gold, actual).match

    def test_date_timestamp_equals_iso_string(self):
        gold = [{"d": datetime.datetime(2018, 3, 1, 0, 0, 0)}]
        actual = [{"x": "2018-03-01T00:00:00"}]
        assert self._cmp(gold, actual).match

    def test_numeric_tolerance(self):
        gold = [{"c": 1, "rev": 33.0}]
        actual = [{"a": "1", "b": "33.004"}]
        assert self._cmp(gold, actual, rel_tol=0.01).match

    def test_empty_does_not_match_nonempty(self):
        gold = [{"pm": "credit_card", "n": 55}]
        assert not self._cmp(gold, []).match

    def test_empty_matches_empty(self):
        assert self._cmp([], []).match

    def test_value_mismatch_detected(self):
        gold = [{"pm": "credit_card", "n": 55}]
        actual = [{"a": "credit_card", "b": "999"}]
        assert not self._cmp(gold, actual).match

    def test_ordered_topn(self):
        gold = [{"c": 30, "n": 2}, {"c": 5, "n": 1}]
        actual = [{"cid": "30", "cnt": "2"}, {"cid": "5", "cnt": "1"}]
        assert self._cmp(gold, actual, ordering="ordered").match

    def test_set_ignores_row_order(self):
        gold = [{"k": "a", "v": 1}, {"k": "b", "v": 2}]
        actual = [{"x": "b", "y": "2"}, {"x": "a", "y": "1"}]
        assert self._cmp(gold, actual).match

    def test_extra_descriptive_columns_projected(self):
        # MetricFlow adds first_name/last_name alongside the requested id + measure; the
        # gold columns are all present with correct values, so projecting onto gold matches.
        gold = [{"customer_id": 51, "clv": 99.0}, {"customer_id": 3, "clv": 65.0}]
        actual = [
            {"customer__customer_id": "51", "first": "Howard", "last": "R.", "clv": "99.0"},
            {"customer__customer_id": "3", "first": "Kathleen", "last": "P.", "clv": "65.0"},
        ]
        res = self._cmp(gold, actual)
        assert res.match
        assert res.diff["mode"] == "column_projected"

    def test_projection_does_not_rescue_wrong_values(self):
        # Right shape + extra columns, but the measure is wrong (sum 496 vs avg 17) — no
        # column subset can make a wrong number match.
        gold = [{"month": "2018-01-01", "avg_val": 17.1}]
        actual = [{"metric_time__month": "2018-01-01", "first": "x", "avg_val": "496.0"}]
        assert not self._cmp(gold, actual).match

    def test_projection_requires_exact_cardinality(self):
        # Extra columns AND extra rows (100 vs 62 grain error) — projection tolerates extra
        # columns but never extra/missing rows.
        gold = [{"cid": 1, "n": 2}, {"cid": 2, "n": 1}]
        actual = [
            {"customer__cid": "1", "name": "a", "n": "2"},
            {"customer__cid": "2", "name": "b", "n": "1"},
            {"customer__cid": "3", "name": "c", "n": "5"},
        ]
        assert not self._cmp(gold, actual).match

    def test_fewer_columns_than_gold_is_mismatch(self):
        # Generated dropped a gold column — genuine structural shortfall, not extra context.
        gold = [{"pm": "credit_card", "rev": 871.0}, {"pm": "other", "rev": 801.0}]
        actual = [{"rev": "411.0"}]
        res = self._cmp(gold, actual)
        assert not res.match
        assert res.diff["mode"] == "column_count_mismatch"

    # --- ordered comparison: tie tolerance (de-flaking top-N tasks) ----------------
    def test_ordered_tie_reorder_passes(self):
        # Two rows tied on the sort key (v=50) returned in swapped order — same result set,
        # must not flip pass/fail.
        gold = [{"c": 1, "v": 50}, {"c": 2, "v": 50}, {"c": 3, "v": 40}]
        actual = [{"x": 2, "y": "50"}, {"x": 1, "y": "50"}, {"x": 3, "y": "40"}]
        assert self._cmp(gold, actual, ordering="ordered").match

    def test_ordered_reversed_direction_still_fails(self):
        # Same rows, reversed ranking direction — ordering must still catch this.
        gold = [{"c": 1, "v": 100}, {"c": 2, "v": 90}, {"c": 3, "v": 80}]
        actual = [{"x": 3, "y": "80"}, {"x": 2, "y": "90"}, {"x": 1, "y": "100"}]
        assert not self._cmp(gold, actual, ordering="ordered").match

    def test_ordered_wrong_tie_member_still_fails(self):
        # A different row at the tie (cid 9 not in gold) — genuine wrong set, must fail.
        gold = [{"c": 1, "v": 50}, {"c": 2, "v": 50}, {"c": 3, "v": 40}]
        actual = [{"x": 1, "y": "50"}, {"x": 9, "y": "50"}, {"x": 3, "y": "40"}]
        assert not self._cmp(gold, actual, ordering="ordered").match

    def test_ordered_no_ties_unaffected(self):
        gold = [{"c": 1, "v": 30}, {"c": 2, "v": 20}, {"c": 3, "v": 10}]
        actual = [{"x": 1, "y": "30"}, {"x": 2, "y": "20"}, {"x": 3, "y": "10"}]
        assert self._cmp(gold, actual, ordering="ordered").match


class TestOrderedPrefixAcceptance:
    """Gold-as-ordered-prefix relaxation (synth_030): when the question fixes no N
    ("top customers"), gold's LIMIT is arbitrary — a produced top-10 whose first rows
    are EXACTLY gold's top-5 must count as a match, without opening any set-based or
    partial acceptance."""

    def _cmp(self, gold, actual, **rules):
        from semanticflow.evaluation.result_compare import compare_results
        rules.setdefault("column_agnostic", True)
        rules.setdefault("ordering", "ordered")
        return compare_results(gold, actual, compare_rules=rules)

    def test_gold_five_is_prefix_of_produced_ten(self):
        # The synth_030 shape: gold LIMIT 5, pipeline chose top 10, head identical.
        gold = [{"c": i, "v": 100 - 10 * i} for i in range(5)]
        actual = [{"x": str(i), "y": str(100 - 10 * i)} for i in range(10)]
        res = self._cmp(gold, actual)
        assert res.match
        assert res.diff["prefix_match"] is True
        assert res.diff["extra_rows"] == 5  # auditable: matched as prefix, 5 benign extras

    def test_produced_shorter_than_gold_still_fails(self):
        # The relaxation is one-directional: a truncated produced result misses gold rows.
        gold = [{"c": i, "v": 100 - 10 * i} for i in range(10)]
        actual = [{"x": str(i), "y": str(100 - 10 * i)} for i in range(5)]
        res = self._cmp(gold, actual)
        assert not res.match
        assert "prefix_match" not in res.diff

    def test_prefix_with_wrong_mid_row_fails(self):
        # Wrong entity at rank 3 — gold is not a strict prefix, no partial credit.
        gold = [{"c": i, "v": 100 - 10 * i} for i in range(5)]
        actual = [{"x": str(i), "y": str(100 - 10 * i)} for i in range(10)]
        actual[2] = {"x": "99", "y": "80"}
        assert not self._cmp(gold, actual).match

    def test_tie_straddling_cutoff_matches_fairly(self):
        # Gold (LIMIT 3) ends on v=80; produced has TWO rows tied at 80 with the other
        # member first. Gold's pick belongs to the straddling tie run, so the prefix holds.
        gold = [{"c": 1, "v": 95}, {"c": 2, "v": 90}, {"c": 3, "v": 80}]
        actual = [
            {"x": "1", "y": "95"}, {"x": "2", "y": "90"},
            {"x": "7", "y": "80"}, {"x": "3", "y": "80"},
            {"x": "5", "y": "70"},
        ]
        res = self._cmp(gold, actual)
        assert res.match
        assert res.diff["prefix_match"] is True

    def test_tie_at_cutoff_wrong_member_still_fails(self):
        # Gold's tail row (c=3) is NOT in produced's straddling tie run — genuine miss.
        gold = [{"c": 1, "v": 95}, {"c": 2, "v": 90}, {"c": 3, "v": 80}]
        actual = [
            {"x": "1", "y": "95"}, {"x": "2", "y": "90"},
            {"x": "7", "y": "80"}, {"x": "8", "y": "80"},
            {"x": "5", "y": "70"},
        ]
        assert not self._cmp(gold, actual).match

    def test_unordered_comparison_gets_no_prefix_forgiveness(self):
        # Set-mode extra rows are grain errors, never a benign continuation.
        gold = [{"c": 1, "v": 10}, {"c": 2, "v": 20}]
        actual = [{"x": "1", "y": "10"}, {"x": "2", "y": "20"}, {"x": "3", "y": "30"}]
        res = self._cmp(gold, actual, ordering="set")
        assert not res.match
        assert "prefix_match" not in res.diff


class TestEvaluatorHeadline:
    """The headline overall_score prefers execution equivalence, falling back to
    spec fields only when no executable gold exists."""

    def _eval(self, task_id, actual):
        from semanticflow.config import Settings
        from semanticflow.agents.evaluator import EvaluatorAgent
        from semanticflow.agents.types import AgentMessage
        from semanticflow.tasks.loader import load_tasks

        tasks = {t.task_id: t for t in load_tasks("data/real_tasks.jsonl")}
        ev = EvaluatorAgent(settings=Settings())
        # mf_query_success mirrors the real pipeline: a query that did not execute
        # successfully cannot be scored as execution-correct.
        state = {"mf_query_result": actual, "mf_query_success": bool(actual),
                 "semantic_spec": {"metric_name": "orders", "base_measures": ["order_count"]}}
        out = asyncio.run(ev.run(AgentMessage(task=tasks[task_id], state=state)))
        return out.state["evaluation"]

    def test_correct_execution_scores_one(self):
        actual = [{"x": "gift_card", "m": "12"}, {"x": "credit_card", "m": "55"},
                  {"x": "coupon", "m": "13"}, {"x": "bank_transfer", "m": "33"}]
        e = self._eval("real_005", actual)
        assert e["accuracy_basis"] == "execution"
        assert e["overall_score"] == 1.0

    def test_empty_result_is_not_correct(self):
        # Regression: stale-data relative-time tasks used to "match" on [] == [].
        e = self._eval("real_005", [])
        assert e["accuracy_basis"] == "execution"
        assert e["overall_score"] == 0.0

    def test_no_gold_falls_back_to_spec_fields(self):
        # All benchmark tasks now carry gold, so construct a gold-less task inline.
        import asyncio
        from semanticflow.config import Settings
        from semanticflow.agents.evaluator import EvaluatorAgent
        from semanticflow.agents.types import AgentMessage
        from semanticflow.tasks.loader import Task
        task = Task(task_id="nogold", nl_request="orders",
                    schema_context={"models": ["orders"], "columns": {"orders": ["order_id"]}},
                    expected_semantic={"metric": "orders", "base_measure": "order_count"})
        ev = EvaluatorAgent(settings=Settings())
        state = {"mf_query_result": [], "semantic_spec": {"metric_name": "orders", "base_measures": ["order_count"]}}
        e = asyncio.run(ev.run(AgentMessage(task=task, state=state))).state["evaluation"]
        assert e["accuracy_basis"] == "spec_fields"


class TestSimulatedHuman:
    """Scoped human reveals only the queried slot, never the whole gold spec."""

    EXP = {"metric": "orders_per_day", "base_measure": "order_count",
           "group_by": ["order_date"], "time_granularity": "day",
           "filters": ["status = 'completed'"], "limit": 10}

    def test_measure_question_reveals_only_measure(self):
        from semanticflow.evaluation.sim_human import scoped_human_answer
        a = scoped_human_answer("Should this be a COUNT or SUM aggregation?", self.EXP)
        assert "order_count" in a
        assert "order_date" not in a and "completed" not in a  # no leakage of other slots

    def test_grouping_question_reveals_only_grouping(self):
        from semanticflow.evaluation.sim_human import scoped_human_answer
        a = scoped_human_answer("Which dimension should I group by?", self.EXP)
        assert "order_date" in a
        assert "order_count" not in a

    def test_unrecognized_question_reveals_nothing(self):
        from semanticflow.evaluation.sim_human import scoped_human_answer
        assert scoped_human_answer("What do you mean exactly?", self.EXP) == ""

    def test_no_full_spec_leak(self):
        from semanticflow.evaluation.sim_human import scoped_human_answer
        a = scoped_human_answer("What measure?", self.EXP)
        assert "orders_per_day" not in a  # metric label never dumped wholesale

    def test_budget_caps_answers(self):
        from semanticflow.evaluation.sim_human import make_simulated_human
        provider = make_simulated_human(self.EXP, budget=1)
        ans = provider(["What measure should I use?", "Which dimension to group by?"])
        answered = [v for v in ans.values() if v]
        assert len(answered) == 1  # only one informative answer within budget


class TestThresholdSweep:
    """Offline effort-accuracy curve combines no-HITL signal with if-asked accuracy."""

    def _write(self, exp_id, rows):
        from pathlib import Path
        import json
        d = Path("experiments") / exp_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        return d

    def test_curve_monotone_endpoints(self):
        import shutil
        from semanticflow.experiment import threshold_sweep
        no_id, hi_id = "_test_sweep_nohitl", "_test_sweep_hitl"
        # task A: high uncertainty, HITL fixes it (0->1). task B: low uncertainty, already correct.
        no_rows = [
            {"task_id": "A", "epistemic_uncertainty": 0.8, "semantic_accuracy": 0.0, "accuracy_basis": "execution"},
            {"task_id": "B", "epistemic_uncertainty": 0.1, "semantic_accuracy": 1.0, "accuracy_basis": "execution"},
        ]
        hi_rows = [
            {"task_id": "A", "semantic_accuracy": 1.0, "accuracy_basis": "execution", "num_questions_asked": 2},
            {"task_id": "B", "semantic_accuracy": 1.0, "accuracy_basis": "execution", "num_questions_asked": 1},
        ]
        try:
            self._write(no_id, no_rows)
            self._write(hi_id, hi_rows)
            r = threshold_sweep(no_id, hi_id, steps=11)
            assert r["n_tasks"] == 2
            assert r["accuracy_never_ask"] == 0.5   # (0 + 1) / 2
            assert r["accuracy_always_ask"] == 1.0
            assert r["best_point"]["mean_accuracy"] == 1.0
            # best should trigger only A (1 task), not both
            assert r["best_point"]["num_triggered"] == 1
        finally:
            import shutil
            shutil.rmtree("experiments/" + no_id, ignore_errors=True)
            shutil.rmtree("experiments/" + hi_id, ignore_errors=True)


class TestStructuralDisagreement:
    def test_identical_specs_zero(self):
        from semanticflow.agents.semantic_mapper import structural_disagreement
        s = {"base_measures": ["amount"], "group_by": ["order_date"], "filters": [], "time_granularity": "day"}
        assert structural_disagreement([s, dict(s)]) == 0.0

    def test_disjoint_specs_high(self):
        from semanticflow.agents.semantic_mapper import structural_disagreement
        a = {"base_measures": ["amount"], "group_by": ["order_date"], "filters": ["x"], "time_granularity": "day"}
        b = {"base_measures": ["count"], "group_by": ["customer_id"], "filters": ["y"], "time_granularity": "month"}
        assert structural_disagreement([a, b]) == 1.0

    def test_partial_disagreement_between(self):
        from semanticflow.agents.semantic_mapper import structural_disagreement
        a = {"base_measures": ["amount"], "group_by": ["order_date"], "filters": [], "time_granularity": "day"}
        b = {"base_measures": ["amount"], "group_by": ["customer_id"], "filters": [], "time_granularity": "day"}
        d = structural_disagreement([a, b])
        assert 0.0 < d < 1.0


class TestUncertaintyValidation:
    def _write(self, exp_id, rows):
        import json
        from pathlib import Path
        d = Path("experiments") / exp_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    def test_auroc_perfect_separation(self):
        import shutil
        from semanticflow.evaluation.uncertainty_validation import validate_uncertainty
        exp = "_test_unc_val"
        # high uncertainty -> incorrect; low -> correct => AUROC should be 1.0
        rows = [
            {"task_id": "a", "epistemic_uncertainty": 0.9, "structural_uncertainty": 0.8,
             "accuracy_basis": "execution", "execution_match": False, "semantic_accuracy": 0.0},
            {"task_id": "b", "epistemic_uncertainty": 0.8, "structural_uncertainty": 0.7,
             "accuracy_basis": "execution", "execution_match": False, "semantic_accuracy": 0.0},
            {"task_id": "c", "epistemic_uncertainty": 0.1, "structural_uncertainty": 0.1,
             "accuracy_basis": "execution", "execution_match": True, "semantic_accuracy": 1.0},
            {"task_id": "d", "epistemic_uncertainty": 0.2, "structural_uncertainty": 0.0,
             "accuracy_basis": "execution", "execution_match": True, "semantic_accuracy": 1.0},
        ]
        try:
            self._write(exp, rows)
            r = validate_uncertainty(exp)
            assert r["n_correct"] == 2 and r["n_incorrect"] == 2
            # auroc_predicts_error is now {"auroc": ..., optionally "ci95"/"unstable"}
            assert r["signals"]["epistemic_uncertainty"]["auroc_predicts_error"]["auroc"] == 1.0
        finally:
            import shutil
            shutil.rmtree("experiments/" + exp, ignore_errors=True)


class TestSelfConsistency:
    """k-sampling produces aleatoric + total uncertainty; k=1 reproduces prior behaviour."""

    def _agent(self, varying: bool):
        import json as _json
        from semanticflow.llm.base import BaseLLMClient
        from semanticflow.agents.semantic_mapper import SemanticMapperAgent
        from semanticflow.config import Settings

        class StubClient(BaseLLMClient):
            def __init__(self, vary):
                self.n = 0
                self.vary = vary
                self._model = "stub"

            def complete(self, system_prompt, user_prompt, temperature=None, json_mode=False):
                self.n += 1
                gb = "customer_id" if (self.vary and self.n % 2 == 0) else "order_date"
                return _json.dumps({"metric_name": "m", "base_measures": ["amount"],
                                    "group_by": [gb], "filters": [], "time_granularity": "day",
                                    "confidence": 0.8})

        return SemanticMapperAgent(settings=Settings(), llm_clients={"openai": StubClient(varying)})

    def _run(self, agent):
        import asyncio
        from semanticflow.agents.types import AgentMessage
        from semanticflow.tasks.loader import Task
        task = Task(task_id="t", nl_request="orders by day",
                    schema_context={"models": ["orders"], "columns": {"orders": ["order_date", "amount", "customer_id"]}})
        return asyncio.run(agent.run(AgentMessage(task=task, state={}))).state

    def test_k1_aleatoric_zero(self):
        agent = self._agent(varying=True)  # k defaults to 1
        st = self._run(agent)
        assert st["aleatoric_uncertainty"] == 0.0
        assert "total_uncertainty" in st

    def test_k_varying_samples_positive_aleatoric(self):
        agent = self._agent(varying=True)
        agent.settings.self_consistency_k = 4
        st = self._run(agent)
        assert st["aleatoric_uncertainty"] > 0.0
        assert st["total_uncertainty"] >= st["aleatoric_uncertainty"]


class TestPairedStats:
    def test_mcnemar_exact_known_values(self):
        from semanticflow.experiment import _mcnemar_exact_p
        # 6 clean gains, 0 regressions -> p = 2*0.5^6 = 0.03125 (first significant)
        assert abs(_mcnemar_exact_p(0, 6) - 0.03125) < 1e-9
        assert _mcnemar_exact_p(0, 5) > 0.05   # 5 gains not significant (0.0625)
        assert _mcnemar_exact_p(0, 0) == 1.0

    def test_paired_stats_gains_and_ci(self):
        from semanticflow.experiment import paired_stats
        base = {f"t{i}": {"accuracy_basis": "execution", "execution_match": False} for i in range(8)}
        treat = {f"t{i}": {"accuracy_basis": "execution", "execution_match": True} for i in range(6)}
        # tasks 6,7: treatment also wrong (no change)
        treat["t6"] = {"accuracy_basis": "execution", "execution_match": False}
        treat["t7"] = {"accuracy_basis": "execution", "execution_match": False}
        ids = [f"t{i}" for i in range(8)]
        s = paired_stats(base, treat, ids, n_boot=500)
        assert s["n_pairs"] == 8
        assert s["gains"] == 6 and s["regressions"] == 0
        assert s["delta"] == 0.75   # 6/8
        assert s["significant_at_05"] is True
        assert s["delta_ci95"][0] <= s["delta"] <= s["delta_ci95"][1]

    def test_paired_stats_ignores_non_execution(self):
        from semanticflow.experiment import paired_stats
        base = {"a": {"accuracy_basis": "spec_fields", "semantic_accuracy": 0.0}}
        treat = {"a": {"accuracy_basis": "spec_fields", "semantic_accuracy": 1.0}}
        assert paired_stats(base, treat, ["a"])["n_pairs"] == 0


class TestRepeatAggregation:
    def _res(self, overall, match):
        from semanticflow.agents.orchestrator import OrchestrationResult
        return OrchestrationResult(
            task_id="t", mode="x",
            state={"evaluation": {"overall_score": overall, "execution_match": match,
                                  "accuracy_basis": "execution"}},
            clarifying_questions=[], runtime_sec=1.0)

    def test_mean_and_majority(self):
        from semanticflow.experiment import _aggregate_repeats
        agg = _aggregate_repeats([self._res(1.0, True), self._res(0.0, False), self._res(1.0, True)])
        e = agg.state["evaluation"]
        assert abs(e["overall_score"] - 2/3) < 1e-9   # mean
        assert e["execution_match"] is True            # majority (2/3)
        assert e["n_repeats"] == 3 and e["accuracy_std"] > 0

    def test_single_run_std_zero(self):
        from semanticflow.experiment import _aggregate_repeats
        e = _aggregate_repeats([self._res(1.0, True)]).state["evaluation"]
        assert e["n_repeats"] == 1 and e["accuracy_std"] == 0.0

    def test_usage_summed_across_repeats(self):
        from semanticflow.agents.orchestrator import OrchestrationResult
        from semanticflow.experiment import _aggregate_repeats

        def res(cost, calls):
            return OrchestrationResult(
                task_id="t", mode="x",
                state={"evaluation": {"overall_score": 1.0, "execution_match": True,
                                      "accuracy_basis": "execution"},
                       "llm_usage": {"anthropic": {"input_tokens": 100, "output_tokens": 50,
                                                   "total_tokens": 150, "cost_usd": cost, "calls": calls}}},
                clarifying_questions=[], runtime_sec=1.0)
        agg = _aggregate_repeats([res(0.001, 1), res(0.002, 1), res(0.003, 1)])
        u = agg.state["llm_usage"]["anthropic"]
        assert u["calls"] == 3 and u["total_tokens"] == 450
        assert abs(u["cost_usd"] - 0.006) < 1e-9   # summed, not just the last repeat


class TestUsageTracker:
    """Token usage aggregates per provider and cost is derived from the pricing table."""

    def test_accumulates_and_costs(self):
        from semanticflow.llm.usage import UsageTracker
        t = UsageTracker()
        t.add("anthropic", "claude-sonnet-4-6", 1000, 500)
        t.add("anthropic", "claude-sonnet-4-6", 2000, 200)
        a = t.snapshot()["anthropic"]
        assert a["input_tokens"] == 3000 and a["output_tokens"] == 700
        assert a["total_tokens"] == 3700 and a["calls"] == 2
        # 3000 * $3/M + 700 * $15/M = 0.0195
        assert abs(a["cost_usd"] - 0.0195) < 1e-9

    def test_unknown_model_zero_cost_but_counts_tokens(self):
        from semanticflow.llm.usage import UsageTracker
        t = UsageTracker()
        t.add("openai", "some-future-model", 100, 100)
        o = t.snapshot()["openai"]
        assert o["total_tokens"] == 200 and o["cost_usd"] == 0.0

    def test_reset_clears(self):
        from semanticflow.llm.usage import UsageTracker
        t = UsageTracker()
        t.add("gemini", "gemini-2.0-flash", 10, 10)
        t.reset()
        assert t.snapshot() == {}

    def test_experiment_usage_summary_reads_state(self):
        from semanticflow.experiment import _usage_summary
        state = {"llm_usage": {"anthropic": {"total_tokens": 3700.0, "cost_usd": 0.0195}}}
        s = _usage_summary(state)
        assert s["token_counts"]["anthropic"] == 3700.0
        assert s["estimated_cost"]["anthropic"] == 0.0195

    def test_summarize_cost_totals_across_rows(self):
        import json
        import shutil
        from pathlib import Path

        from semanticflow.experiment import summarize_cost
        exp = "_test_cost_summary"
        d = Path("experiments") / exp
        d.mkdir(parents=True, exist_ok=True)
        rows = [
            {"task_id": "a", "latency_sec": 10.0,
             "usage_detail": {"anthropic": {"input_tokens": 1000, "output_tokens": 500,
                                            "total_tokens": 1500, "cost_usd": 0.0195, "calls": 2}}},
            {"task_id": "b", "latency_sec": 12.0,
             "usage_detail": {"anthropic": {"input_tokens": 2000, "output_tokens": 300,
                                            "total_tokens": 2300, "cost_usd": 0.0105, "calls": 1},
                              "openai": {"input_tokens": 500, "output_tokens": 500,
                                         "total_tokens": 1000, "cost_usd": 0.00625, "calls": 1}}},
        ]
        try:
            (d / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            rep = summarize_cost(exp)
            assert rep["n_tasks"] == 2
            assert rep["total_tokens"] == 4800
            assert abs(rep["total_cost_usd"] - 0.03625) < 1e-6
            assert rep["by_provider"]["anthropic"]["calls"] == 3
            assert rep["total_latency_sec"] == 22.0
        finally:
            shutil.rmtree(d, ignore_errors=True)


def run_tests():
    import traceback

    test_classes = [TestColumnAgnosticCompare, TestEvaluatorHeadline,
                    TestSimulatedHuman, TestThresholdSweep,
                    TestStructuralDisagreement, TestUncertaintyValidation,
                    TestSelfConsistency, TestPairedStats, TestRepeatAggregation,
                    TestUsageTracker]
    passed = failed = 0
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
    sys.exit(0 if run_tests() else 1)
