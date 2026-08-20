"""Tests for the improved uncertainty stack: execution-based consistency clustering,
conformal selective-HITL calibration, and the reframed threshold sweep (oracle bound +
questions-at-fixed-accuracy)."""
from __future__ import annotations


class TestExecutionConsistency:
    def _ec(self, tables, successes=None, **rules):
        from semanticflow.evaluation.execution_consistency import execution_consistency
        return execution_consistency(tables, successes, rules or None)

    def test_unanimous_agreement_is_zero_uncertainty(self):
        t = [{"d": "2018-03-01", "n": 2}]
        ec = self._ec([list(t), list(t), list(t)])
        assert ec.largest_cluster == 3
        assert ec.agreement == 1.0
        assert ec.execution_uncertainty == 0.0
        assert ec.n_clusters == 1

    def test_equivalent_despite_column_renames(self):
        # MetricFlow-style names vs gold aliases must NOT split the cluster.
        a = [{"order_date": "2018-03-01", "orders": 2}]
        b = [{"metric_time__day": "2018-03-01", "orders_metric": "2"}]
        ec = self._ec([a, b])
        assert ec.n_clusters == 1
        assert ec.execution_uncertainty == 0.0

    def test_split_answers_raise_uncertainty(self):
        a = [{"x": 1}]
        b = [{"x": 2}]
        c = [{"x": 1}]
        ec = self._ec([a, b, c])
        assert ec.largest_cluster == 2  # {a, c}
        assert round(ec.execution_uncertainty, 4) == round(1 - 2 / 3, 4)
        assert ec.modal_sample_index in (0, 2)

    def test_failed_samples_are_singletons(self):
        a = [{"x": 1}]
        ec = self._ec([a, None, a], successes=[True, False, True])
        assert ec.n_failed == 1
        assert ec.n_executed == 2
        assert ec.largest_cluster == 2
        assert ec.n_clusters == 2  # {a,a} + failed singleton

    def test_all_failed_is_max_uncertainty(self):
        ec = self._ec([None, None], successes=[False, False])
        assert ec.execution_uncertainty == 0.5  # two singleton failures
        assert ec.modal_sample_index is None

    def test_empty_input(self):
        ec = self._ec([])
        assert ec.n_samples == 0 and ec.execution_uncertainty == 0.0


class TestConformalCalibration:
    def test_perfect_signal_accepts_only_low_uncertainty(self):
        from semanticflow.evaluation.conformal import calibrate_threshold
        # Signal perfectly separates: low signal => correct, high => wrong. alpha=0.4 is
        # the smallest target the conservative Hoeffding slack can satisfy at n_acc=4
        # (slack ~= 0.29) — itself an honest illustration that the bound is loose at small n.
        signals = [0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9]
        correct = [True, True, True, True, False, False, False, False]
        out = calibrate_threshold(signals, correct, alpha=0.4, delta=0.5)
        assert out["tau"] is not None
        # Accepts only the correct (low-signal) tasks => zero empirical error, asks the rest.
        assert out["accepted_risk_empirical"] == 0.0
        assert out["n_accepted"] == 4
        assert out["ask_rate"] == 0.5

    def test_no_region_when_target_impossible(self):
        from semanticflow.evaluation.conformal import calibrate_threshold
        # Errors everywhere, even the lowest-signal task is wrong => can't hit alpha=0.
        signals = [0.2, 0.4, 0.6, 0.8]
        correct = [False, False, False, False]
        out = calibrate_threshold(signals, correct, alpha=0.0, delta=0.5)
        assert out["tau"] is None

    def test_evaluate_selective_accept_all(self):
        from semanticflow.evaluation.conformal import evaluate_selective
        ev = evaluate_selective([0.1, 0.2], [True, False], tau=None)
        assert ev["ask_rate"] == 0.0
        assert ev["accepted_accuracy"] == 0.5


class TestSweepReframe:
    def _rows_sweep(self, tmp_path, monkeypatch):
        """Build two tiny experiments on disk and run threshold_sweep over them."""
        import json

        import semanticflow.experiment as exp

        exproot = tmp_path / "experiments"

        def write(eid, rows):
            d = exproot / eid
            d.mkdir(parents=True)
            (d / "results.jsonl").write_text(
                "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
            )

        # 3 tasks. Asking helps t1 (0->1), is neutral on t2 (1->1), hurts t3 (1->0).
        no_hitl = [
            {"task_id": "t1", "total_uncertainty": 0.9, "semantic_accuracy": 0.0,
             "accuracy_basis": "execution", "execution_match": False},
            {"task_id": "t2", "total_uncertainty": 0.1, "semantic_accuracy": 1.0,
             "accuracy_basis": "execution", "execution_match": True},
            {"task_id": "t3", "total_uncertainty": 0.5, "semantic_accuracy": 1.0,
             "accuracy_basis": "execution", "execution_match": True},
        ]
        hitl = [
            {"task_id": "t1", "semantic_accuracy": 1.0, "accuracy_basis": "execution",
             "execution_match": True, "num_questions_asked": 2},
            {"task_id": "t2", "semantic_accuracy": 1.0, "accuracy_basis": "execution",
             "execution_match": True, "num_questions_asked": 1},
            {"task_id": "t3", "semantic_accuracy": 0.0, "accuracy_basis": "execution",
             "execution_match": False, "num_questions_asked": 1},
        ]
        write("noh", no_hitl)
        write("hl", hitl)
        monkeypatch.chdir(tmp_path)
        return exp.threshold_sweep("noh", "hl", steps=11, signal="total_uncertainty")

    def test_oracle_and_frontier(self, tmp_path, monkeypatch):
        rep = self._rows_sweep(tmp_path, monkeypatch)
        # never-ask = mean(a0) = (0+1+1)/3; always-ask = mean(a1) = (1+1+0)/3.
        assert rep["accuracy_never_ask"] == round(2 / 3, 4)
        assert rep["accuracy_always_ask"] == round(2 / 3, 4)
        # Oracle asks only on t1 (the one task asking helps) => accuracy 3/3.
        oracle = rep["oracle_routing"]
        assert oracle["accuracy"] == 1.0
        assert oracle["tasks_helped_by_asking"] == 1
        assert oracle["tasks_hurt_by_asking"] == 1
        assert oracle["questions_asked"] == 2
        assert "questions_at_fixed_accuracy" in rep
