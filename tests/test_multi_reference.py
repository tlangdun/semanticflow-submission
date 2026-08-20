"""Tests for multi-reference (multi-gold) scoring of ambiguous / underspecified tasks.

A genuinely ambiguous NL request ("How's business?") has several valid interpretations;
scoring against a single arbitrary gold penalises correct disambiguations (AmbigQA
EMNLP'20; AmbiQT EMNLP'23; Papicchio et al. DBML'24). A task may now carry
``acceptable_semantics`` (extra valid specs) and/or ``acceptable_results`` (extra valid
gold tables); the score is the MAX over the acceptable set. Single-gold behaviour is
unchanged. All offline — no API.
"""
from __future__ import annotations

import asyncio


def _eval(task, actual, semantic_spec):
    from semanticflow.agents.evaluator import EvaluatorAgent
    from semanticflow.agents.types import AgentMessage
    from semanticflow.config import Settings

    ev = EvaluatorAgent(settings=Settings())
    state = {"mf_query_result": actual, "mf_query_success": bool(actual),
             "semantic_spec": semantic_spec}
    out = asyncio.run(ev.run(AgentMessage(task=task, state=state)))
    return out.state["evaluation"]


def _task(**overrides):
    from semanticflow.tasks.loader import Task
    base = dict(
        task_id="t", nl_request="how's business?",
        schema_context={"models": ["orders"],
                        "columns": {"orders": ["order_id", "amount", "order_date"]}},
    )
    base.update(overrides)
    return Task(**base)


# --- spec-field multi-reference -------------------------------------------------------

# The intended (primary) reading: daily revenue trend.
PRIMARY = {"base_measures": ["amount"], "group_by": ["order_date"], "filters": []}
# A second, equally-valid reading: a single ungrouped total-revenue headline.
ALT = {"base_measures": ["amount"], "group_by": [], "filters": []}


class TestSpecFieldMultiReference:
    def test_single_gold_unchanged_when_spec_matches(self):
        # No acceptable_semantics: a spec matching the sole gold scores 1.0, exactly as before.
        spec = {"metric_name": "rev", "base_measures": ["amount"],
                "group_by": ["order_date"], "filters": []}
        e = _eval(_task(expected_semantic=PRIMARY), [], spec)
        assert e["spec_field_score"] == 1.0
        assert "spec_field_score_n_refs" not in e  # single-gold path leaves no marker

    def test_single_gold_unchanged_when_spec_differs(self):
        # A spec matching the *alternative* reading but with only the single PRIMARY gold
        # still scores below 1.0 — multi-gold must not silently rescue single-gold tasks.
        spec = {"metric_name": "rev", "base_measures": ["amount"],
                "group_by": [], "filters": []}  # ungrouped — differs from PRIMARY grain
        e = _eval(_task(expected_semantic=PRIMARY), [], spec)
        assert e["spec_field_score"] < 1.0

    def test_multi_gold_takes_the_max(self):
        # Same spec as above (matches ALT, not PRIMARY). With ALT in acceptable_semantics the
        # best-matching interpretation wins, so the score rises to 1.0.
        spec = {"metric_name": "rev", "base_measures": ["amount"],
                "group_by": [], "filters": []}
        single = _eval(_task(expected_semantic=PRIMARY), [], spec)["spec_field_score"]
        multi = _eval(
            _task(expected_semantic=PRIMARY, acceptable_semantics=[ALT]), [], spec
        )
        assert multi["spec_field_score"] == 1.0
        assert multi["spec_field_score"] > single          # max strictly helps here
        assert multi["spec_field_score_n_refs"] == 2

    def test_spec_matching_second_acceptable_interpretation_scores_high(self):
        # Primary gold is a count-by-status reading; the spec instead realises a
        # revenue-by-day reading that lives in the 2nd acceptable slot. It must score high.
        primary = {"base_measures": ["order_id"], "group_by": ["status"], "filters": []}
        accept = [{"base_measures": ["amount"], "group_by": ["customer_id"], "filters": []},
                  {"base_measures": ["amount"], "group_by": ["order_date"], "filters": []}]
        spec = {"metric_name": "rev", "base_measures": ["amount"],
                "group_by": ["order_date"], "filters": []}
        e = _eval(
            _task(expected_semantic=primary, acceptable_semantics=accept), [], spec
        )
        assert e["spec_field_score"] == 1.0
        assert e["spec_field_score_n_refs"] == 3

    def test_primary_still_wins_when_it_is_the_best(self):
        # A spec that perfectly matches the PRIMARY reading still scores 1.0 even when
        # weaker alternatives are present — max never lowers the primary score.
        spec = {"metric_name": "rev", "base_measures": ["amount"],
                "group_by": ["order_date"], "filters": []}
        e = _eval(
            _task(expected_semantic=PRIMARY, acceptable_semantics=[ALT]), [], spec
        )
        assert e["spec_field_score"] == 1.0


# --- execution multi-reference --------------------------------------------------------

class TestExecutionMultiReference:
    def test_single_gold_execution_unchanged(self):
        gold = [{"d": "2018-01-01", "rev": 100.0}]
        rules = {"column_agnostic": True}
        # Matching the sole gold -> execution_match True, no multi-gold marker.
        e = _eval(_task(gold_query_result=gold, compare_rules=rules),
                  [{"x": "2018-01-01", "y": "100.0"}], {"metric_name": "m"})
        assert e["execution_match"] is True
        assert "execution_n_refs" not in e
        # A non-matching result against a single gold stays a miss.
        e2 = _eval(_task(gold_query_result=gold, compare_rules=rules),
                   [{"x": "2018-01-01", "y": "999.0"}], {"metric_name": "m"})
        assert e2["execution_match"] is False

    def test_result_matching_acceptable_gold_is_correct(self):
        primary = [{"d": "2018-01-01", "rev": 100.0}]            # daily revenue
        alt = [{"total": 555.0}]                                   # single revenue headline
        rules = {"column_agnostic": True}
        task = _task(gold_query_result=primary, acceptable_results=[alt],
                     compare_rules=rules)
        # Actual matches the ALTERNATIVE gold, not the primary -> still correct.
        e = _eval(task, [{"t": "555.0"}], {"metric_name": "m"})
        assert e["execution_match"] is True
        assert e["execution_n_refs"] == 2

    def test_result_matching_neither_gold_fails(self):
        primary = [{"d": "2018-01-01", "rev": 100.0}]
        alt = [{"total": 555.0}]
        rules = {"column_agnostic": True}
        task = _task(gold_query_result=primary, acceptable_results=[alt],
                     compare_rules=rules)
        e = _eval(task, [{"t": "777.0"}], {"metric_name": "m"})
        assert e["execution_match"] is False


def run_tests():
    import traceback
    classes = [TestSpecFieldMultiReference, TestExecutionMultiReference]
    passed = failed = 0
    for cls in classes:
        print(f"\n=== {cls.__name__} ===")
        inst = cls()
        for name in dir(inst):
            if name.startswith("test_"):
                try:
                    getattr(inst, name)()
                    print(f"  PASS {name}")
                    passed += 1
                except Exception as e:  # noqa: BLE001
                    print(f"  FAIL {name}: {e}")
                    traceback.print_exc()
                    failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_tests() else 1)
