from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from statistics import stdev
from typing import Any
from uuid import uuid4

import numpy as np
from scipy import stats

from semanticflow.agents.orchestrator import OrchestrationResult, Orchestrator
from semanticflow.config import Settings
from semanticflow.evaluation import make_sim_human
from semanticflow.llm import build_cheap_client
from semanticflow.log import get_logger
from semanticflow.tasks.loader import load_tasks

logger = get_logger("semanticflow.experiment")


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _collect_generated_files(state: dict) -> list[str]:
    files = state.get("generated_files") or []
    return [str(item) for item in files]


def _collect_hashes(files: list[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for file_path in files:
        path = Path(file_path)
        if path.exists():
            hashes[file_path] = _hash_file(path)
    return hashes


def _save_artifacts(result: OrchestrationResult, run_dir: Path) -> list[str]:
    state = result.state
    artifacts: list[str] = []
    generated_files = _collect_generated_files(state)
    for file_path in generated_files:
        source = Path(file_path)
        if not source.exists():
            continue
        relative = source.as_posix().lstrip("/")
        dest = run_dir / "generated" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(source.read_bytes())
        artifacts.append(str(dest))

    dbt_results = state.get("dbt_results", {})
    _write_text(run_dir / "logs/dbt.log", dbt_results.get("dbt_logs", ""))
    artifacts.append(str(run_dir / "logs/dbt.log"))

    _write_text(run_dir / "logs/mf_validate.log", state.get("mf_validate_stdout", ""))
    _write_text(run_dir / "logs/mf_validate.err", state.get("mf_validate_stderr", ""))
    _write_text(run_dir / "logs/mf_query.log", state.get("mf_query_stdout", ""))
    _write_text(run_dir / "logs/mf_query.err", state.get("mf_query_stderr", ""))
    artifacts.extend(
        [
            str(run_dir / "logs/mf_validate.log"),
            str(run_dir / "logs/mf_validate.err"),
            str(run_dir / "logs/mf_query.log"),
            str(run_dir / "logs/mf_query.err"),
        ]
    )

    _write_json(run_dir / "semantic_spec.json", state.get("semantic_spec", {}))
    _write_json(run_dir / "model_outputs.json", state.get("model_outputs", {}))
    _write_json(run_dir / "llm_debug.json", state.get("llm_debug", {}))
    raw_outputs = state.get("llm_raw_outputs")
    if raw_outputs is not None:
        _write_json(run_dir / "llm_raw.json", raw_outputs)
        artifacts.append(str(run_dir / "llm_raw.json"))
    _write_json(run_dir / "evaluation.json", state.get("evaluation", {}))
    artifacts.extend(
        [
            str(run_dir / "semantic_spec.json"),
            str(run_dir / "model_outputs.json"),
            str(run_dir / "llm_debug.json"),
            str(run_dir / "evaluation.json"),
        ]
    )
    return artifacts


def _usage_summary(state: dict) -> dict[str, Any]:
    """Real per-provider token usage + estimated cost from the orchestrator's tracker.

    Falls back to zeros (keyed by the providers that produced specs) if usage wasn't
    captured — e.g. the mock client, which makes no real API calls."""
    usage = state.get("llm_usage") or {}
    if not usage:
        providers = state.get("model_outputs", {}).keys()
        zero = {p: 0.0 for p in providers}
        return {"token_counts": zero, "estimated_cost": dict(zero), "usage_detail": {}}
    token_counts = {p: v.get("total_tokens", 0.0) for p, v in usage.items()}
    estimated_cost = {p: round(v.get("cost_usd", 0.0), 6) for p, v in usage.items()}
    return {"token_counts": token_counts, "estimated_cost": estimated_cost, "usage_detail": usage}


def log_run(
    result: OrchestrationResult,
    settings: Settings,
    experiment_id: str,
    results_path: Path,
) -> dict:
    run_id = uuid4().hex
    timestamp = _timestamp()
    state = result.state
    evaluation = state.get("evaluation", {})
    dbt_results = state.get("dbt_results", {})
    generated_files = _collect_generated_files(state)
    file_hashes = _collect_hashes(generated_files)
    run_dir = Path("experiments") / experiment_id / run_id
    artifacts = _save_artifacts(result, run_dir)
    usage = _usage_summary(state)

    row = {
        "experiment_id": experiment_id,
        "run_id": run_id,
        "mode": result.mode,
        "timestamp": timestamp,
        "task_id": result.task_id,
        "config_path": settings.config_path,
        "dbt_project_dir": settings.dbt_project_dir,
        "llm_provider": settings.llm_provider,
        "semantic_accuracy": evaluation.get("overall_score"),
        "accuracy_basis": evaluation.get("accuracy_basis"),
        "n_repeats": evaluation.get("n_repeats"),
        "accuracy_std": evaluation.get("accuracy_std"),
        "accuracy_runs": evaluation.get("accuracy_runs"),
        # Uncertainty signal + HITL effort — needed for the offline threshold sweep.
        "epistemic_uncertainty": state.get("epistemic_uncertainty"),
        "epistemic_uncertainty_norm": state.get("epistemic_uncertainty_norm"),
        "structural_uncertainty": state.get("structural_uncertainty"),
        "aleatoric_uncertainty": state.get("aleatoric_uncertainty"),
        "aleatoric_degraded": state.get("aleatoric_degraded"),
        "total_uncertainty": state.get("total_uncertainty"),
        # Execution-based consistency (present only when settings.execution_consistency).
        "execution_uncertainty": state.get("execution_uncertainty"),
        "execution_agreement": state.get("execution_agreement"),
        "total_uncertainty_exec": state.get("total_uncertainty_exec"),
        # Post-HITL accuracy levers (observability): which fired and how.
        "execution_selection_valid": state.get("execution_selection_valid"),
        "execution_selection_reason": state.get("execution_selection_reason"),
        "execution_selection_provider": state.get("execution_selection_provider"),
        "execution_selection_overridden_by_hitl": state.get(
            "execution_selection_overridden_by_hitl"
        ),
        "verification_repairs": state.get("verification_repairs"),
        "auto_fixes_applied": result.auto_fixes_applied,
        "refinement_iterations": result.refinement_iterations,
        "semantic_agreement_score": state.get("semantic_agreement_score"),
        "ambiguity": state.get("ambiguity"),
        "ambiguity_detected": result.ambiguity_detected,
        # Verification-grounded trigger signal (deterministic schema checks on the spec).
        "verification_score": state.get("verification_score"),
        "verification_violations": state.get("verification_violations"),
        "num_questions_asked": len(result.clarifying_questions_asked or []),
        "user_answers": result.user_answers,
        # HITL/repair observability: separates "asked and got nothing" from "did not
        # ask", and "repair exhausted" from "never failed".
        "clarification_failed": state.get("clarification_failed"),
        "post_execution_hitl": state.get("post_execution_hitl"),
        "slot_guard_reverted": state.get("slot_guard_reverted"),
        "error_recovery_exhausted": state.get("error_recovery_exhausted"),
        "execution_repair_exhausted": state.get("execution_repair_exhausted"),
        "preflight_violations": state.get("preflight_violations"),
        "spec_field_score": evaluation.get("spec_field_score"),
        "metric_name_similarity": evaluation.get("metric_name_similarity"),
        "measures_f1": evaluation.get("measures_f1"),
        "group_by_f1": evaluation.get("group_by_f1"),
        "filters_f1": evaluation.get("filters_f1"),
        "aggregation_match": evaluation.get("aggregation_match"),
        "dbt_parse_success": bool(dbt_results.get("dbt_parse_success")),
        "dbt_build_success": bool(dbt_results.get("dbt_build_success")),
        "mf_validate_success": bool(state.get("mf_validate_success")),
        "mf_validate_stdout": state.get("mf_validate_stdout", ""),
        "mf_validate_stderr": state.get("mf_validate_stderr", ""),
        "mf_validate_issues": state.get("mf_validate_issues", []),
        "mf_query_success": bool(state.get("mf_query_success")),
        "mf_query_stdout": state.get("mf_query_stdout", ""),
        "mf_query_stderr": state.get("mf_query_stderr", ""),
        "mf_query_result": state.get("mf_query_result", []),
        "gold_sql_present": bool(state.get("gold_sql_present")),
        "gold_sql_exec_success": bool(state.get("gold_sql_exec_success")),
        "gold_result": state.get("gold_result", []),
        "execution_accuracy": state.get("execution_accuracy"),
        "execution_match": state.get("execution_match"),
        "result_diff_summary": state.get("result_diff_summary", {}),
        "llm_debug": state.get("llm_debug", {}),
        "latency_sec": result.runtime_sec,
        "token_counts": usage["token_counts"],
        "estimated_cost": usage["estimated_cost"],
        "usage_detail": usage["usage_detail"],
        "generated_files": generated_files,
        "file_hashes": file_hashes,
        "artifacts": artifacts,
        "artifacts_dir": str(run_dir),
    }

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=str) + "\n")
    return row


def _aggregate_results(rows: list[dict]) -> dict[str, float]:
    def _avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    semantic_scores = [row.get("semantic_accuracy") for row in rows if row.get("semantic_accuracy") is not None]
    parse_rates = [1.0 if row.get("dbt_parse_success") else 0.0 for row in rows]
    build_rates = [1.0 if row.get("dbt_build_success") else 0.0 for row in rows]
    mf_validate_rates = [1.0 if row.get("mf_validate_success") else 0.0 for row in rows]
    mf_query_rates = [1.0 if row.get("mf_query_success") else 0.0 for row in rows]
    # Only average execution accuracy over tasks actually scored on the execution basis
    # (gold ran and produced rows). Rows that fell back to spec-fields log a voided
    # execution_accuracy of 0.0 that would otherwise drag this average down.
    exec_scores = [
        row.get("execution_accuracy")
        for row in rows
        if row.get("execution_accuracy") is not None
        and row.get("accuracy_basis") == "execution"
    ]
    return {
        "semantic_accuracy_avg": _avg([float(item) for item in semantic_scores]),
        "dbt_parse_success_rate": _avg(parse_rates),
        "dbt_build_success_rate": _avg(build_rates),
        "mf_validate_success_rate": _avg(mf_validate_rates),
        "mf_query_success_rate": _avg(mf_query_rates),
        "execution_accuracy_avg": _avg([float(item) for item in exec_scores]),
    }


def _exec_correct(row: dict) -> bool | None:
    """Binary correctness from execution-equivalence only (None if not execution-scored)."""
    if row.get("accuracy_basis") != "execution":
        return None
    if row.get("execution_match") is not None:
        return bool(row["execution_match"])
    acc = row.get("semantic_accuracy")
    return None if acc is None else float(acc) >= 1.0


def _mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact (binomial) McNemar p-value for discordant counts b and c.

    The discordant pairs are Binomial(b+c, 0.5) under H0; this is the exact two-sided
    binomial test (scipy), which is the defensible standard for a small-n thesis."""
    n = b + c
    if n == 0:
        return 1.0
    return float(stats.binomtest(min(b, c), n=n, p=0.5, alternative="two-sided").pvalue)


def paired_stats(
    baseline_by_task: dict[str, dict],
    treatment_by_task: dict[str, dict],
    task_ids: list[str],
    n_boot: int = 2000,
    seed: int = 12345,
) -> dict:
    """Paired execution-accuracy comparison on tasks scored by execution in BOTH runs.

    Reports the effect size (accuracy delta) with a bootstrap 95% CI, plus McNemar's
    exact test on the discordant pairs. With small n the p-value is exploratory — the
    CI on the effect size is the honest primary read."""
    pairs: list[tuple[bool, bool]] = []
    for tid in task_ids:
        b = _exec_correct(baseline_by_task.get(tid, {}))
        t = _exec_correct(treatment_by_task.get(tid, {}))
        if b is None or t is None:
            continue
        pairs.append((b, t))
    n = len(pairs)
    if n == 0:
        return {"n_pairs": 0, "note": "no tasks execution-scored in both runs"}

    base_acc = sum(1 for b, _ in pairs if b) / n
    treat_acc = sum(1 for _, t in pairs if t) / n
    b_only = sum(1 for b, t in pairs if b and not t)   # baseline right, treatment wrong (regressions)
    c_only = sum(1 for b, t in pairs if t and not b)   # treatment right, baseline wrong (gains)

    # Paired bootstrap of the accuracy delta: resample task indices with replacement,
    # recompute treat-minus-base each time, take the percentile CI (np.percentile uses
    # standard linear interpolation rather than a truncated-index lookup).
    rng = np.random.default_rng(seed)
    base = np.array([1.0 if b else 0.0 for b, _ in pairs])
    treat = np.array([1.0 if t else 0.0 for _, t in pairs])
    idx = rng.integers(0, n, size=(n_boot, n))
    deltas = treat[idx].mean(axis=1) - base[idx].mean(axis=1)
    lo, hi = np.percentile(deltas, [2.5, 97.5])

    mcnemar_p = _mcnemar_exact_p(b_only, c_only)
    return {
        "n_pairs": n,
        "baseline_accuracy": round(base_acc, 4),
        "treatment_accuracy": round(treat_acc, 4),
        "delta": round(treat_acc - base_acc, 4),
        "delta_ci95": [round(float(lo), 4), round(float(hi), 4)],
        "gains": c_only,
        "regressions": b_only,
        "mcnemar_exact_p": round(mcnemar_p, 4),
        "significant_at_05": mcnemar_p < 0.05,
    }


def _load_experiment_results(experiment_id: str) -> list[dict]:
    """Load all results from an experiment."""
    exp_dir = Path("experiments") / experiment_id
    results_file = exp_dir / "results.jsonl"

    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    rows = []
    with results_file.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _aggregate_cost(rows: list[dict]) -> dict[str, Any]:
    """Total token usage + estimated cost across result rows (no experiment_id).

    Prefers each row's per-provider ``usage_detail`` (input/output/calls/cost); falls
    back to the flat ``token_counts``/``estimated_cost`` for rows logged before usage
    detail existed."""
    by_provider: dict[str, dict[str, float]] = {}

    def _bucket(provider: str) -> dict[str, float]:
        return by_provider.setdefault(
            provider,
            {"input_tokens": 0.0, "output_tokens": 0.0, "total_tokens": 0.0,
             "cost_usd": 0.0, "calls": 0.0},
        )

    for row in rows:
        detail = row.get("usage_detail") or {}
        if detail:
            for provider, u in detail.items():
                agg = _bucket(provider)
                agg["input_tokens"] += u.get("input_tokens", 0.0)
                agg["output_tokens"] += u.get("output_tokens", 0.0)
                agg["total_tokens"] += u.get("total_tokens", 0.0)
                agg["cost_usd"] += u.get("cost_usd", 0.0)
                agg["calls"] += u.get("calls", 0.0)
        else:
            # Legacy rows: only flat totals available.
            for provider, tok in (row.get("token_counts") or {}).items():
                agg = _bucket(provider)
                agg["total_tokens"] += tok or 0.0
            for provider, cost in (row.get("estimated_cost") or {}).items():
                _bucket(provider)["cost_usd"] += cost or 0.0

    total_cost = round(sum(v["cost_usd"] for v in by_provider.values()), 6)
    total_tokens = sum(v["total_tokens"] for v in by_provider.values())
    n_tasks = len(rows)
    latencies = [r.get("latency_sec") for r in rows if r.get("latency_sec") is not None]
    return {
        "n_tasks": n_tasks,
        "by_provider": {p: {k: round(v, 6) for k, v in agg.items()} for p, agg in by_provider.items()},
        "total_tokens": total_tokens,
        "total_cost_usd": total_cost,
        "avg_cost_per_task_usd": round(total_cost / n_tasks, 6) if n_tasks else 0.0,
        "total_latency_sec": round(sum(latencies), 2) if latencies else 0.0,
    }


def summarize_cost(experiment_id: str) -> dict[str, Any]:
    """Total token usage + estimated cost across a saved experiment's rows."""
    report = _aggregate_cost(_load_experiment_results(experiment_id))
    return {"experiment_id": experiment_id, **report}


def _load_tasks_metadata(tasks_path: str = "data/real_tasks.jsonl") -> dict[str, dict]:
    """Load task metadata (ambiguity, difficulty) for grouping."""
    tasks = load_tasks(tasks_path)
    return {t.task_id: {"ambiguity": t.ambiguity, "difficulty": t.difficulty} for t in tasks}


def compare_experiments(
    baseline_id: str,
    treatment_id: str,
    tasks_path: str = "data/real_tasks.jsonl",
) -> dict:
    """Compare two experiments and generate a detailed report.
    
    Args:
        baseline_id: Experiment ID for baseline (e.g., single agent)
        treatment_id: Experiment ID for treatment (e.g., multi-agent HITL)
        tasks_path: Path to tasks file for metadata
    
    Returns:
        Comparison report with metrics, deltas, and winner
    """
    baseline_rows = _load_experiment_results(baseline_id)
    treatment_rows = _load_experiment_results(treatment_id)
    
    # Index by task_id for comparison
    baseline_by_task = {row["task_id"]: row for row in baseline_rows}
    treatment_by_task = {row["task_id"]: row for row in treatment_rows}
    
    # Get task metadata
    task_meta = _load_tasks_metadata(tasks_path)
    
    # Find common tasks
    common_tasks = set(baseline_by_task.keys()) & set(treatment_by_task.keys())
    
    # Aggregate metrics
    baseline_agg = _aggregate_results([baseline_by_task[t] for t in common_tasks])
    treatment_agg = _aggregate_results([treatment_by_task[t] for t in common_tasks])
    
    # Calculate deltas
    metrics = {}
    for key in baseline_agg:
        base_val = baseline_agg[key]
        treat_val = treatment_agg[key]
        metrics[key] = {
            "baseline": base_val,
            "treatment": treat_val,
            "delta": treat_val - base_val,
        }
    
    # Group by ambiguity level
    by_ambiguity: dict[str, dict] = {}
    for level in ["low", "medium", "high"]:
        level_tasks = [t for t in common_tasks if task_meta.get(t, {}).get("ambiguity") == level]
        if level_tasks:
            base_scores = [
                baseline_by_task[t].get("semantic_accuracy", 0) or 0
                for t in level_tasks
            ]
            treat_scores = [
                treatment_by_task[t].get("semantic_accuracy", 0) or 0
                for t in level_tasks
            ]
            base_avg = sum(base_scores) / len(base_scores)
            treat_avg = sum(treat_scores) / len(treat_scores)
            by_ambiguity[level] = {
                "task_count": len(level_tasks),
                "baseline": base_avg,
                "treatment": treat_avg,
                "delta": treat_avg - base_avg,
            }
    
    # Group by difficulty
    by_difficulty: dict[str, dict] = {}
    for level in ["simple", "medium", "complex"]:
        level_tasks = [t for t in common_tasks if task_meta.get(t, {}).get("difficulty") == level]
        if level_tasks:
            base_scores = [
                baseline_by_task[t].get("semantic_accuracy", 0) or 0
                for t in level_tasks
            ]
            treat_scores = [
                treatment_by_task[t].get("semantic_accuracy", 0) or 0
                for t in level_tasks
            ]
            base_avg = sum(base_scores) / len(base_scores)
            treat_avg = sum(treat_scores) / len(treat_scores)
            by_difficulty[level] = {
                "task_count": len(level_tasks),
                "baseline": base_avg,
                "treatment": treat_avg,
                "delta": treat_avg - base_avg,
            }
    
    # Per-task comparison
    per_task = []
    for task_id in sorted(common_tasks):
        base_row = baseline_by_task[task_id]
        treat_row = treatment_by_task[task_id]
        base_score = base_row.get("semantic_accuracy", 0) or 0
        treat_score = treat_row.get("semantic_accuracy", 0) or 0
        per_task.append({
            "task_id": task_id,
            "ambiguity": task_meta.get(task_id, {}).get("ambiguity", "unknown"),
            "difficulty": task_meta.get(task_id, {}).get("difficulty", "unknown"),
            "baseline_score": base_score,
            "treatment_score": treat_score,
            "delta": treat_score - base_score,
            "baseline_mf_success": base_row.get("mf_query_success", False),
            "treatment_mf_success": treat_row.get("mf_query_success", False),
        })
    
    # Paired statistics on execution-scored tasks: overall + the key cut (high-ambiguity,
    # where HITL is meant to help, vs the rest).
    common_list = sorted(common_tasks)
    high_tasks = [t for t in common_list if task_meta.get(t, {}).get("ambiguity") == "high"]
    rest_tasks = [t for t in common_list if task_meta.get(t, {}).get("ambiguity") != "high"]
    stats = {
        "overall": paired_stats(baseline_by_task, treatment_by_task, common_list),
        "high_ambiguity": paired_stats(baseline_by_task, treatment_by_task, high_tasks),
        "low_medium_ambiguity": paired_stats(baseline_by_task, treatment_by_task, rest_tasks),
    }

    # Determine winner
    semantic_delta = metrics.get("semantic_accuracy_avg", {}).get("delta", 0)
    mf_delta = metrics.get("mf_query_success_rate", {}).get("delta", 0)
    
    if semantic_delta > 0.05 or mf_delta > 0.1:
        winner = "treatment"
    elif semantic_delta < -0.05 or mf_delta < -0.1:
        winner = "baseline"
    else:
        winner = "tie"
    
    return {
        "baseline_id": baseline_id,
        "treatment_id": treatment_id,
        "task_count": len(common_tasks),
        "metrics": metrics,
        "by_ambiguity": by_ambiguity,
        "by_difficulty": by_difficulty,
        "per_task": per_task,
        "stats": stats,
        "winner": winner,
    }


def threshold_sweep(
    no_hitl_id: str,
    hitl_id: str,
    steps: int = 21,
    basis_filter: str | None = "execution",
    signal: str = "epistemic_uncertainty",
) -> dict:
    """Offline effort-accuracy sweep of the HITL trigger threshold.

    Combines a no-HITL run (the uncertainty signal + no-ask accuracy ``a0``) with a
    HITL run (the if-asked accuracy ``a1``), keyed by task. For each threshold tau the
    task is "triggered" when its uncertainty >= tau, taking ``a1``; otherwise ``a0``.
    This yields the coverage/effort vs accuracy curve without re-running the LLM.

    IMPORTANT: for a clean curve the HITL run should be a *force-ask* run (every task
    asks), so ``a1`` is defined for all tasks — run it with
    ``SEMANTICFLOW_AGREEMENT_THRESHOLD=1.01``. Otherwise ``a1==a0`` for tasks the live
    run did not trigger, understating the achievable gain.

    ``basis_filter='execution'`` restricts to tasks scored by execution-equivalence
    (the trustworthy metric); pass ``None`` to include spec-field-scored tasks too.
    """
    no_hitl = {r["task_id"]: r for r in _load_experiment_results(no_hitl_id)}
    hitl = {r["task_id"]: r for r in _load_experiment_results(hitl_id)}

    rows = []
    for tid in sorted(set(no_hitl) & set(hitl)):
        n, h = no_hitl[tid], hitl[tid]
        u, a0, a1 = n.get(signal), n.get("semantic_accuracy"), h.get("semantic_accuracy")
        if u is None or a0 is None or a1 is None:
            continue
        if basis_filter and (n.get("accuracy_basis") != basis_filter or h.get("accuracy_basis") != basis_filter):
            continue
        rows.append({"task_id": tid, "u": float(u), "a0": float(a0), "a1": float(a1),
                     "questions": int(h.get("num_questions_asked") or 0)})

    if not rows:
        return {"error": "no comparable tasks with uncertainty + accuracy in both runs",
                "no_hitl_id": no_hitl_id, "hitl_id": hitl_id}

    n_tasks = len(rows)
    curve = []
    for i in range(steps):
        tau = i / (steps - 1) if steps > 1 else 0.0
        triggered = [r for r in rows if r["u"] >= tau]
        acc = sum((r["a1"] if r["u"] >= tau else r["a0"]) for r in rows) / n_tasks
        curve.append({
            "threshold": round(tau, 4),
            "num_triggered": len(triggered),
            "trigger_fraction": round(len(triggered) / n_tasks, 4),
            "questions_asked": sum(r["questions"] for r in triggered),
            "mean_accuracy": round(acc, 4),
        })

    never = round(sum(r["a0"] for r in rows) / n_tasks, 4)   # never ask (tau > 1)
    always = round(sum(r["a1"] for r in rows) / n_tasks, 4)  # always ask (tau = 0)
    # Best accuracy with the fewest triggers as the tie-breaker (least human effort).
    best = max(curve, key=lambda c: (c["mean_accuracy"], -c["num_triggered"]))

    # ORACLE routing: a perfect signal asks ONLY on tasks where asking actually flips a
    # wrong answer right (a1 > a0) and never on tasks where it would hurt (a1 < a0).
    # This is the unreachable upper bound on what *any* uncertainty signal could buy —
    # the gap between `best` and `oracle` is the headroom a better signal can recover,
    # and the gap between `always` and `oracle` is the waste in asking on everything.
    helped = [r for r in rows if r["a1"] > r["a0"]]
    hurt = [r for r in rows if r["a1"] < r["a0"]]
    oracle_acc = sum(max(r["a0"], r["a1"]) for r in rows) / n_tasks
    oracle = {
        "accuracy": round(oracle_acc, 4),
        "num_triggered": len(helped),
        "questions_asked": sum(r["questions"] for r in helped),
        "tasks_helped_by_asking": len(helped),
        "tasks_hurt_by_asking": len(hurt),
    }

    # Questions-at-fixed-accuracy: the honest selective-HITL metric is not "beat
    # always-ask on accuracy" (always-ask is a strong baseline) but "match a target
    # accuracy for the fewest questions". For each target we report the cheapest curve
    # point (fewest questions) that reaches it.
    frontier = {}
    for label, target in (("match_90pct_of_always", 0.90 * always),
                          ("match_95pct_of_always", 0.95 * always),
                          ("match_always", always)):
        feasible = [c for c in curve if c["mean_accuracy"] >= target - 1e-9]
        if feasible:
            pt = min(feasible, key=lambda c: (c["questions_asked"], c["num_triggered"]))
            frontier[label] = {
                "target_accuracy": round(target, 4),
                "threshold": pt["threshold"],
                "questions_asked": pt["questions_asked"],
                "trigger_fraction": pt["trigger_fraction"],
                "mean_accuracy": pt["mean_accuracy"],
            }
        else:
            frontier[label] = {"target_accuracy": round(target, 4), "unreachable": True}

    return {
        "no_hitl_id": no_hitl_id,
        "hitl_id": hitl_id,
        "signal": signal,
        "n_tasks": n_tasks,
        "basis_filter": basis_filter,
        "accuracy_never_ask": never,
        "accuracy_always_ask": always,
        "best_point": best,
        "oracle_routing": oracle,
        "questions_at_fixed_accuracy": frontier,
        "curve": curve,
    }


def _aggregate_repeats(results: list[OrchestrationResult]) -> OrchestrationResult:
    """Collapse repeated runs of one task into a representative result.

    Keeps the last run's state/artifacts but overwrites the headline accuracy with the
    mean over repeats and ``execution_match`` with the majority, and records per-run
    accuracies + std so downstream consumers stay one-row-per-task while LLM
    stochasticity is captured."""
    rep = results[-1]
    if len(results) == 1:
        rep.state.setdefault("evaluation", {}).update({"n_repeats": 1, "accuracy_std": 0.0})
        return rep

    scores = [r.state.get("evaluation", {}).get("overall_score") for r in results]
    valid = [s for s in scores if s is not None]
    exec_matches = [
        bool(r.state.get("evaluation", {}).get("execution_match"))
        for r in results
        if r.state.get("evaluation", {}).get("accuracy_basis") == "execution"
    ]
    mean = sum(valid) / len(valid) if valid else None
    # Sample std (n-1 denominator): we are estimating run-to-run variability from a small
    # sample of repeats, not describing a full population.
    std = stdev(valid) if len(valid) > 1 else 0.0

    evaluation = dict(rep.state.get("evaluation", {}))
    if mean is not None:
        evaluation["overall_score"] = mean
    if exec_matches:
        evaluation["execution_match"] = sum(exec_matches) / len(exec_matches) >= 0.5
    evaluation["accuracy_runs"] = valid
    evaluation["accuracy_std"] = std
    evaluation["n_repeats"] = len(results)
    rep.state["evaluation"] = evaluation
    rep.runtime_sec = sum(r.runtime_sec for r in results) / len(results)

    # Sum LLM usage across ALL repeats: the representative state otherwise carries only
    # the last repeat's usage, undercounting this task's true token spend by ~k x.
    merged_usage: dict[str, dict[str, float]] = {}
    for r in results:
        for provider, u in (r.state.get("llm_usage") or {}).items():
            agg = merged_usage.setdefault(
                provider,
                {"input_tokens": 0.0, "output_tokens": 0.0, "total_tokens": 0.0,
                 "cost_usd": 0.0, "calls": 0.0},
            )
            for key in agg:
                agg[key] += u.get(key, 0.0)
    rep.state["llm_usage"] = merged_usage
    return rep


def run_experiment(
    mode: str,
    tasks_path: str,
    out_path: str,
    settings: Settings,
    experiment_id: str | None = None,
) -> None:
    tasks = load_tasks(tasks_path)
    orchestrator = Orchestrator(settings)
    experiment_id = experiment_id or f"exp-{int(time.time())}"
    results_path = Path(out_path)

    repeats = max(1, getattr(settings, "experiment_repeats", 1))
    hitl_mode = mode == "multi_agent_hitl"
    # LLM sim human (opt-in via SEMANTICFLOW_SIM_HUMAN=llm): one cheap-tier client shared
    # across tasks, only built when the HITL arm will actually consult it. The keyword
    # human (default) costs nothing and preserves comparability with earlier runs.
    sim_mode = getattr(settings, "sim_human_mode", "keyword")
    sim_client = (
        build_cheap_client(settings, orchestrator.usage_tracker)
        if (hitl_mode and sim_mode == "llm")
        else None
    )
    # Hard per-task wall-clock cap. A stalled LLM call with no inner timeout can otherwise
    # hang the entire run on one task (observed: a task stuck for hours). SIGALRM fires in
    # the main thread (where this loop runs) and interrupts the synchronous run_task; the
    # task is then recorded as a timeout stub so the run continues and the task count holds.
    import signal as _signal
    task_timeout = max(0, int(getattr(settings, "task_timeout_sec", 0) or 0))
    _have_alarm = hasattr(_signal, "SIGALRM")

    class _TaskTimeout(Exception):
        pass

    def _on_alarm(signum, frame):  # pragma: no cover - timing dependent
        raise _TaskTimeout()

    rows = []
    for task in tasks:
        # Run the full pipeline `repeats` times to quantify LLM stochasticity, then
        # aggregate into one representative result (mean accuracy, majority execution_match).
        run_results = []
        if task_timeout and _have_alarm:
            _signal.signal(_signal.SIGALRM, _on_alarm)
            _signal.alarm(task_timeout)
        try:
            for _ in range(repeats):
                # Scoped simulated human: answers only the slot a question targets, capped by
                # a budget — so question quality matters and human effort is measured
                # honestly. Built per repeat: the budget persists across consultations within
                # one run (pre-execution + post-execution), so a fresh human per repeat keeps
                # repeats independent.
                simulated_human = make_sim_human(
                    task.expected_semantic,
                    budget=settings.hitl_answer_budget,
                    mode=sim_mode,
                    llm_client=sim_client,
                    nl_request=task.nl_request,
                )
                start = time.time()
                r = orchestrator.run_task(
                    task,
                    mode=mode,
                    hitl_mode=hitl_mode,
                    user_input_provider=simulated_human if hitl_mode else None,
                )
                r.runtime_sec = time.time() - start
                run_results.append(r)
        except _TaskTimeout:
            logger.warning(
                f"Task {task.task_id} exceeded task_timeout_sec={task_timeout}s; "
                f"recording timeout stub and continuing"
            )
            stub = {
                "experiment_id": experiment_id, "task_id": task.task_id, "mode": mode,
                "timed_out": True, "accuracy_basis": "timeout",
                "semantic_accuracy": 0.0, "execution_accuracy": 0.0,
                "ambiguity": getattr(task, "ambiguity", "unknown"),
            }
            results_path.parent.mkdir(parents=True, exist_ok=True)
            with results_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(stub, default=str) + "\n")
            rows.append(stub)
            continue
        finally:
            if task_timeout and _have_alarm:
                _signal.alarm(0)

        result = _aggregate_repeats(run_results)
        row = log_run(result, settings, experiment_id=experiment_id, results_path=results_path)
        rows.append(row)

    summary = _aggregate_results(rows)
    summary["cost"] = _aggregate_cost(rows)
    _write_json(Path("experiments") / experiment_id / "summary.json", summary)
