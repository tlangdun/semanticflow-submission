"""Tests for class-aware scoring (semanticflow/evaluation/class_scoring.py)."""
from __future__ import annotations

import json
import os
from pathlib import Path

from semanticflow.evaluation.class_scoring import score_by_class


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _arm(tmp: Path, name: str, rows: list[dict]) -> None:
    _write_jsonl(tmp / "experiments" / name / "results.jsonl", rows)


def _setup(tmp: Path) -> str:
    """A 4-task world: 2 capability (clear+determinable), 2 underspecified — one of which
    is the real_030 case (reaches intent, but the arbitrary gold execution disagrees)."""
    tasks = [
        {"task_id": "c1", "ambiguity_class": "clear", "nl_request": "count orders"},
        {"task_id": "d1", "ambiguity_class": "determinable", "nl_request": "by status"},
        {"task_id": "u1", "ambiguity_class": "underspecified", "nl_request": "how's business?"},
        {"task_id": "u2", "ambiguity_class": "underspecified", "nl_request": "the metrics?"},
    ]
    tasks_path = tmp / "data" / "tasks.jsonl"
    _write_jsonl(tasks_path, tasks)

    # baseline: single-agent, no detection, weaker capability
    _arm(tmp, "base", [
        {"task_id": "c1", "execution_match": True, "ambiguity_detected": False},
        {"task_id": "d1", "execution_match": False, "ambiguity_detected": False},
        {"task_id": "u1", "execution_match": False, "ambiguity_detected": False},
        {"task_id": "u2", "execution_match": True, "ambiguity_detected": False},
    ])
    # multi no-HITL: better capability, detects both underspecified
    _arm(tmp, "noh", [
        {"task_id": "c1", "execution_match": True, "ambiguity_detected": False},
        {"task_id": "d1", "execution_match": True, "ambiguity_detected": False},
        {"task_id": "u1", "execution_match": False, "ambiguity_detected": True},
        {"task_id": "u2", "execution_match": True, "ambiguity_detected": True},
    ])
    # HITL: u1 recovers to intent AND gold (spec 1.0, exec True); u2 is the real_030 case
    # (spec 1.0 reaches intent, but exec False against the arbitrary gold).
    _arm(tmp, "hitl", [
        {"task_id": "c1", "execution_match": True, "ambiguity_detected": False},
        {"task_id": "d1", "execution_match": True, "ambiguity_detected": False},
        {"task_id": "u1", "execution_match": True, "spec_field_score": 1.0,
         "ambiguity_detected": True, "num_questions_asked": 3},
        {"task_id": "u2", "execution_match": False, "spec_field_score": 1.0,
         "ambiguity_detected": True, "num_questions_asked": 3},
    ])
    return str(tasks_path)


def test_capability_split_and_accuracy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tasks_path = _setup(tmp_path)
    rep = score_by_class("base", "noh", "hitl", tasks_path=tasks_path)

    assert rep["n_capability"] == 2  # c1 + d1
    assert rep["n_underspecified"] == 2
    assert rep["capability_execution_accuracy"]["baseline"]["n"] == 1   # only c1
    assert rep["capability_execution_accuracy"]["no_hitl"]["n"] == 2    # c1 + d1


def test_detection_is_zero_for_baseline_full_for_multi(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tasks_path = _setup(tmp_path)
    rep = score_by_class("base", "noh", "hitl", tasks_path=tasks_path)

    assert rep["underspecified_detection_rate"]["baseline"]["n"] == 0
    assert rep["underspecified_detection_rate"]["no_hitl"]["n"] == 2


def test_recovery_to_intent_credits_real030_case(tmp_path, monkeypatch):
    """u2 reaches the intended spec (spec_field_score==1.0) but exec_match is False:
    intent-scoring counts it as recovered; exact-gold execution does not."""
    monkeypatch.chdir(tmp_path)
    tasks_path = _setup(tmp_path)
    rep = score_by_class("base", "noh", "hitl", tasks_path=tasks_path)

    rec = rep["underspecified_recovery_to_intent"]
    assert rec["exact_intent"]["n"] == 2          # u1 + u2 both reach intent
    assert rec["exact_gold_execution"]["n"] == 1  # only u1 matches the arbitrary gold
    disagree = [t["task_id"] for t in rep["underspecified_table"] if t["intent_gold_disagree"]]
    assert disagree == ["u2"]
