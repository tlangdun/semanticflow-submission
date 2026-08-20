"""Class-aware scoring — score each task by the question it actually tests.

Exact-gold execution accuracy is the right metric for tasks with one defensible
answer, but it is *misleading* on the underspecified tasks ("How's business?", "Show me
the numbers"), whose gold is one arbitrary interpretation among several reasonable ones
(see docs §13). On those tasks exact-gold scoring measures interpretation-luck, not
capability, and it mis-credits the HITL loop: a clarified-but-different reading is
scored as a *regression* even when it perfectly reaches the intended interpretation
(real_030 — spec matches intent, but the arbitrary gold's COUNT(*) differs).

This module scores by task class instead (labels in ``data/real_tasks.jsonl``):

- **capability set** (``clear`` + ``determinable``) — one dominant reading, so the model
  should get them with no HITL. Scored by **execution accuracy** (``execution_match``).
- **underspecified set** — metric/grain is a free choice. Scored by the two questions the
  thesis is actually about, *not* exact gold:
    * **detection** (no-HITL arm): did the system recognize it must ask? (``ambiguity_detected``)
    * **recovery-to-intent** (HITL arm): given the human's answer, did the refined spec
      reach the *intended* interpretation? Measured by ``spec_field_score`` against
      ``expected_semantic`` (the labelled intent), at an exact (==1.0) and a partial
      (>=``partial_threshold``) bar — never against the single arbitrary gold's execution.

The point is to separate "can it compute" from "can it detect + recover ambiguity" so
neither number contaminates the other.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from semanticflow.experiment import _load_experiment_results

CAPABILITY_CLASSES = ("clear", "determinable")


def _by_task(experiment_id: str) -> dict[str, dict]:
    return {r["task_id"]: r for r in _load_experiment_results(experiment_id)}


def _load_task_labels(tasks_path: str) -> dict[str, dict]:
    """task_id -> raw task record (carries ambiguity_class, nl_request, expected_semantic).

    Read straight from the JSONL — the ``Task`` dataclass / ``_load_tasks_metadata`` do
    not surface the additive labelling fields (``ambiguity_class`` etc.) added in §13.
    """
    out: dict[str, dict] = {}
    for line in Path(tasks_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            out[rec["task_id"]] = rec
    return out


def _rate(num: int, den: int) -> dict[str, Any]:
    return {"n": num, "of": den, "rate": round(num / den, 4) if den else None}


def score_by_class(
    baseline_id: str,
    no_hitl_id: str,
    hitl_id: str,
    tasks_path: str = "data/real_tasks.jsonl",
    partial_threshold: float = 0.8,
) -> dict[str, Any]:
    """Class-aware scorecard across three arms, reusing saved runs (no re-run).

    ``baseline_id`` / ``no_hitl_id`` are no-HITL arms (capability accuracy + detection);
    ``hitl_id`` is the triggered-HITL arm (recovery-to-intent). All three are scored on
    the capability set by execution accuracy. Returns the per-set metrics plus the
    underspecified per-task table for auditing.
    """
    meta = _load_task_labels(tasks_path)
    cls = {tid: (t.get("ambiguity_class") or "clear") for tid, t in meta.items()}
    runs = {"baseline": _by_task(baseline_id), "no_hitl": _by_task(no_hitl_id),
            "hitl": _by_task(hitl_id)}
    # Only score tasks present in every arm, to keep the comparison paired.
    common = set.intersection(*(set(r) for r in runs.values())) & set(cls)
    cap = sorted(t for t in common if cls[t] in CAPABILITY_CLASSES)
    und = sorted(t for t in common if cls[t] == "underspecified")

    def exec_match(run: dict, tid: str) -> bool:
        return bool(run[tid].get("execution_match"))

    def spec_intent(run: dict, tid: str) -> float:
        return float(run[tid].get("spec_field_score") or 0.0)

    capability = {
        arm: _rate(sum(exec_match(run, t) for t in cap), len(cap))
        for arm, run in runs.items()
    }

    detection = {
        arm: _rate(sum(bool(runs[arm][t].get("ambiguity_detected")) for t in und), len(und))
        for arm in ("baseline", "no_hitl")
    }

    hitl = runs["hitl"]
    recovery = {
        "exact_intent": _rate(sum(spec_intent(hitl, t) >= 0.999 for t in und), len(und)),
        "partial_intent": _rate(
            sum(spec_intent(hitl, t) >= partial_threshold for t in und), len(und)),
        # The old, unfair metric, kept for contrast: exact-gold execution on these tasks.
        "exact_gold_execution": _rate(sum(exec_match(hitl, t) for t in und), len(und)),
    }

    # Per-task underspecified audit (where intent-scoring and gold-scoring disagree is
    # the interesting part — e.g. spec_field_score==1 but execution_match False).
    table = [
        {
            "task_id": t,
            "nl_request": meta[t].get("nl_request", ""),
            "no_hitl_detected": bool(runs["no_hitl"][t].get("ambiguity_detected")),
            "hitl_questions": int(hitl[t].get("num_questions_asked") or 0),
            "hitl_spec_intent": round(spec_intent(hitl, t), 3),
            "hitl_exec_match": exec_match(hitl, t),
            "intent_gold_disagree": (spec_intent(hitl, t) >= 0.999) != exec_match(hitl, t),
        }
        for t in und
    ]

    return {
        "tasks_path": tasks_path,
        "arms": {"baseline": baseline_id, "no_hitl": no_hitl_id, "hitl": hitl_id},
        "n_capability": len(cap),
        "n_underspecified": len(und),
        "capability_execution_accuracy": capability,
        "underspecified_detection_rate": detection,
        "underspecified_recovery_to_intent": recovery,
        "underspecified_table": table,
    }
