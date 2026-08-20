from __future__ import annotations

import json
import time
from pathlib import Path

import typer

from semanticflow.agents.orchestrator import Orchestrator
from semanticflow.config import load_settings
from semanticflow.experiment import log_run, run_experiment
from semanticflow.tasks.loader import load_tasks

app = typer.Typer(help="SemanticFlow CLI")


@app.command("list-tasks")
def list_tasks(
    tasks_path: str = typer.Option("data/real_tasks.jsonl", help="Path to tasks JSONL."),
) -> None:
    tasks = load_tasks(tasks_path)
    for task in tasks:
        typer.echo(f"{task.task_id}\t{task.difficulty}\t{task.ambiguity}\t{task.nl_request}")


@app.command("run-task")
def run_task(
    task_id: str = typer.Option(..., help="Task ID to execute."),
    tasks_path: str = typer.Option("data/real_tasks.jsonl", help="Path to tasks JSONL."),
    mode: str | None = typer.Option(None, help="Override orchestration mode."),
    skip_dbt: bool = typer.Option(False, help="Skip dbt parse/build."),
    hitl: bool = typer.Option(False, help="Enable human-in-the-loop clarification."),
    experiment_id: str | None = typer.Option(None, help="Experiment identifier."),
    results_path: str | None = typer.Option(None, help="Results JSONL output path."),
) -> None:
    tasks = load_tasks(tasks_path)
    task = next((item for item in tasks if item.task_id == task_id), None)
    if not task:
        raise typer.BadParameter(f"Task {task_id} not found in {tasks_path}.")

    settings = load_settings()
    if skip_dbt:
        settings.run_dbt = False

    # Fix for Windows Unicode errors
    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')

    orchestrator = Orchestrator(settings)
    def ask_user(questions: list[str]) -> dict[str, str]:
        answers: dict[str, str] = {}
        for question in questions:
            answers[question] = typer.prompt(question)
        return answers

    should_run_hitl = hitl or (mode == "multi_agent_hitl")
    result = orchestrator.run_task(
        task,
        mode=mode,
        hitl_mode=should_run_hitl,
        user_input_provider=ask_user if should_run_hitl else None,
    )

    experiment_id = experiment_id or f"run-{int(time.time())}"
    results_path = results_path or f"experiments/{experiment_id}/results.jsonl"
    log_run(result, settings, experiment_id=experiment_id, results_path=Path(results_path))

    output_state = dict(result.state)
    output_state.pop("llm_raw_outputs", None)
    output = {
        "task_id": result.task_id,
        "mode": result.mode,
        "runtime_sec": result.runtime_sec,
        "clarifying_questions": result.clarifying_questions,
        "ambiguity_detected": result.ambiguity_detected,
        "clarifying_questions_asked": result.clarifying_questions_asked,
        "user_answers": result.user_answers,
        "state": output_state,
    }
    typer.echo(json.dumps(output, indent=2, default=str))


@app.command("run-experiment")
def run_experiment_command(
    mode: str = typer.Option("multi_agent_no_hitl", help="Experiment mode."),
    tasks_path: str = typer.Option("data/real_tasks.jsonl", help="Path to tasks JSONL."),
    out_path: str | None = typer.Option(None, help="Results JSONL output path."),
    skip_dbt: bool = typer.Option(False, help="Skip dbt parse/build."),
    experiment_id: str | None = typer.Option(None, help="Experiment identifier."),
) -> None:
    settings = load_settings()
    if skip_dbt:
        settings.run_dbt = False
    experiment_id = experiment_id or f"exp-{int(time.time())}"
    out_path = out_path or f"experiments/{experiment_id}/results.jsonl"
    run_experiment(
        mode=mode,
        tasks_path=tasks_path,
        out_path=out_path,
        settings=settings,
        experiment_id=experiment_id,
    )
    typer.echo(f"Saved results to {Path(out_path)}")


@app.command("compare-experiments")
def compare_experiments_command(
    baseline: str = typer.Option(..., help="Baseline experiment ID (e.g., exp-baseline)."),
    treatment: str = typer.Option(..., help="Treatment experiment ID (e.g., exp-multiagent-hitl)."),
    output: str | None = typer.Option(None, help="Output file for comparison report."),
    tasks_path: str = typer.Option("data/all_tasks.jsonl", help="Tasks JSONL for ambiguity strata (use the full benchmark, not a subset)."),
) -> None:
    """Compare two experiments and generate a report."""
    from semanticflow.experiment import compare_experiments

    report = compare_experiments(baseline, treatment, tasks_path=tasks_path)
    
    if output:
        Path(output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        typer.echo(f"Comparison report saved to {output}")
    else:
        # Print formatted report to console
        typer.echo("\n" + "=" * 60)
        typer.echo(f"EXPERIMENT COMPARISON: {baseline} vs {treatment}")
        typer.echo("=" * 60)
        
        typer.echo(f"\n{'Metric':<35} {'Baseline':>10} {'Treatment':>10} {'Δ':>10}")
        typer.echo("-" * 65)
        
        for metric, values in report.get("metrics", {}).items():
            base_val = values.get("baseline", 0)
            treat_val = values.get("treatment", 0)
            delta = values.get("delta", 0)
            delta_str = f"+{delta:.2%}" if delta >= 0 else f"{delta:.2%}"
            typer.echo(f"{metric:<35} {base_val:>10.2%} {treat_val:>10.2%} {delta_str:>10}")
        
        typer.echo("\n" + "-" * 65)
        typer.echo(f"Tasks compared: {report.get('task_count', 0)}")
        
        # Show per-ambiguity breakdown if available
        if "by_ambiguity" in report:
            typer.echo("\nBy Ambiguity Level:")
            for level, data in report["by_ambiguity"].items():
                typer.echo(f"  {level}: baseline={data['baseline']:.2%}, treatment={data['treatment']:.2%}, Δ={data['delta']:+.2%}")
        
        stats = report.get("stats", {})
        if stats:
            typer.echo("\nPaired execution-accuracy stats (effect size primary; McNemar exploratory):")
            for label, s in stats.items():
                if not s or s.get("n_pairs", 0) == 0:
                    typer.echo(f"  {label}: n=0 (no execution-scored pairs)")
                    continue
                ci = s["delta_ci95"]
                typer.echo(
                    f"  {label}: n={s['n_pairs']}, Δ={s['delta']:+.2%} "
                    f"CI95=[{ci[0]:+.2%},{ci[1]:+.2%}], gains={s['gains']} regressions={s['regressions']}, "
                    f"McNemar p={s['mcnemar_exact_p']}{' *' if s['significant_at_05'] else ''}"
                )

        winner = report.get("winner", "tie")
        typer.echo(f"\nWinner: {winner.upper()}")
        typer.echo("=" * 60)


@app.command("sweep-threshold")
def sweep_threshold_command(
    no_hitl: str = typer.Option(..., help="No-HITL experiment ID (provides uncertainty signal + no-ask accuracy)."),
    hitl: str = typer.Option(..., help="HITL experiment ID (if-asked accuracy; ideally a force-ask run)."),
    steps: int = typer.Option(21, help="Number of threshold grid points in [0,1]."),
    signal: str = typer.Option("epistemic_uncertainty", help="Uncertainty column to trigger on (e.g. total_uncertainty)."),
    include_spec: bool = typer.Option(False, help="Include spec-field-scored tasks (default: execution-scored only)."),
    output: str | None = typer.Option(None, help="Write full sweep JSON here."),
) -> None:
    """Offline effort-accuracy sweep of the HITL trigger threshold (RTS-style curve)."""
    from semanticflow.experiment import threshold_sweep

    report = threshold_sweep(no_hitl, hitl, steps=steps, signal=signal,
                             basis_filter=None if include_spec else "execution")
    if output:
        Path(output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        typer.echo(f"Sweep written to {output}")
    if "error" in report:
        typer.echo(f"ERROR: {report['error']}")
        raise typer.Exit(1)

    typer.echo("\n" + "=" * 64)
    typer.echo(f"THRESHOLD SWEEP  ({report['n_tasks']} tasks, basis={report['basis_filter']})")
    typer.echo("=" * 64)
    typer.echo(f"never-ask accuracy : {report['accuracy_never_ask']:.2%}")
    typer.echo(f"always-ask accuracy: {report['accuracy_always_ask']:.2%}")
    b = report["best_point"]
    typer.echo(f"best point         : acc={b['mean_accuracy']:.2%} at threshold={b['threshold']} "
               f"({b['num_triggered']}/{report['n_tasks']} triggered, {b['questions_asked']} questions)")
    typer.echo(f"\n{'threshold':>10} {'triggered':>10} {'questions':>10} {'accuracy':>10}")
    typer.echo("-" * 44)
    for c in report["curve"]:
        typer.echo(f"{c['threshold']:>10.3f} {c['num_triggered']:>10} {c['questions_asked']:>10} {c['mean_accuracy']:>10.2%}")
    typer.echo("=" * 64)


@app.command("validate-uncertainty")
def validate_uncertainty_command(
    experiment: str = typer.Option(..., help="No-HITL experiment ID to analyze."),
    output: str | None = typer.Option(None, help="Write full report JSON here."),
) -> None:
    """AUROC of each uncertainty signal vs execution-correctness (is the signal real?)."""
    from semanticflow.evaluation.uncertainty_validation import validate_uncertainty

    report = validate_uncertainty(experiment)
    if output:
        Path(output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        typer.echo(f"Report written to {output}")
    if "error" in report:
        typer.echo(f"ERROR: {report['error']}")
        raise typer.Exit(1)

    typer.echo("\n" + "=" * 60)
    typer.echo(f"UNCERTAINTY VALIDATION: {experiment}")
    typer.echo("=" * 60)
    typer.echo(f"execution-labelled tasks: {report['n_execution_labelled']} "
               f"(correct={report['n_correct']}, incorrect={report['n_incorrect']})")
    if "warning" in report:
        typer.echo(f"\n⚠️  {report['warning']}")
        return
    for sig, s in report["signals"].items():
        typer.echo(f"\n{sig}:")
        if "error" in s:
            typer.echo(f"  {s['error']}")
            continue
        auroc = s["auroc_predicts_error"]
        if not auroc:
            typer.echo("  AUROC: n/a (only one class present)")
        else:
            line = f"  AUROC (predicts error): {auroc['auroc']:.3f}"
            if "ci95" in auroc:
                line += f"  95% CI [{auroc['ci95'][0]:.3f}, {auroc['ci95'][1]:.3f}]"
            if auroc.get("unstable"):
                line += f"  ⚠️ {auroc.get('note', 'unstable')}"
            typer.echo(line)
        typer.echo(f"  mean on correct={s['mean_on_correct']:.3f}  on incorrect={s['mean_on_incorrect']:.3f}")
        b = s["accuracy_by_uncertainty_tercile"]
        typer.echo(f"  accuracy by tercile: low={b['low_uncertainty']}, "
                   f"mid={b['mid_uncertainty']}, high={b['high_uncertainty']}")
    typer.echo("=" * 60)


@app.command("score-by-class")
def score_by_class_command(
    baseline: str = typer.Option(..., help="Single-agent no-HITL arm (capability + detection)."),
    no_hitl: str = typer.Option(..., help="Multi-agent no-HITL arm (capability + detection)."),
    hitl: str = typer.Option(..., help="Multi-agent triggered-HITL arm (recovery-to-intent)."),
    tasks_path: str = typer.Option("data/real_tasks.jsonl", help="Tasks file with ambiguity_class labels."),
    output: str | None = typer.Option(None, help="Write full report JSON here."),
) -> None:
    """Class-aware scoring: capability by execution; underspecified by detection + recovery-to-intent."""
    from semanticflow.evaluation.class_scoring import score_by_class

    report = score_by_class(baseline, no_hitl, hitl, tasks_path=tasks_path)
    if output:
        Path(output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        typer.echo(f"Report written to {output}")

    def _fmt(r: dict) -> str:
        return f"{r['n']}/{r['of']} = {r['rate']:.0%}" if r["rate"] is not None else "n/a"

    typer.echo("\n" + "=" * 64)
    typer.echo("CLASS-AWARE SCORECARD")
    typer.echo("=" * 64)
    typer.echo(f"CAPABILITY set (clear+determinable, n={report['n_capability']}) — execution accuracy")
    for arm, r in report["capability_execution_accuracy"].items():
        typer.echo(f"  {arm:<12}: {_fmt(r)}")
    typer.echo(f"\nUNDERSPECIFIED set (n={report['n_underspecified']})")
    typer.echo("  detection (recognised it must ask):")
    for arm, r in report["underspecified_detection_rate"].items():
        typer.echo(f"    {arm:<10}: {_fmt(r)}")
    rec = report["underspecified_recovery_to_intent"]
    typer.echo("  HITL recovery to INTENDED interpretation:")
    typer.echo(f"    exact intent (spec==1.0)  : {_fmt(rec['exact_intent'])}")
    typer.echo(f"    partial intent (spec>=0.8): {_fmt(rec['partial_intent'])}")
    typer.echo(f"    [old] exact-gold execution: {_fmt(rec['exact_gold_execution'])}  (unfair baseline)")
    disagree = [t["task_id"] for t in report["underspecified_table"] if t["intent_gold_disagree"]]
    if disagree:
        typer.echo(f"\n  intent/gold disagree on: {', '.join(disagree)} "
                   f"(reached intent but arbitrary gold differs, or vice-versa)")
    typer.echo("=" * 64)


@app.command("conformal-threshold")
def conformal_threshold_command(
    experiment: str = typer.Option(..., help="No-HITL experiment ID (provides signal + execution labels)."),
    signal: str = typer.Option("total_uncertainty", help="Uncertainty column to threshold on."),
    alpha: float = typer.Option(0.2, help="Target accepted-set error rate (auto-answered tasks)."),
    delta: float = typer.Option(0.1, help="Confidence slack for the conservative calibration."),
    splits: int = typer.Option(200, help="Random calibration/test splits to average over."),
    output: str | None = typer.Option(None, help="Write full report JSON here."),
) -> None:
    """Calibrate the HITL trigger with a finite-sample risk guarantee (split-conformal)."""
    from semanticflow.evaluation.conformal import conformal_selective_hitl

    report = conformal_selective_hitl(experiment, signal=signal, alpha=alpha,
                                      delta=delta, n_splits=splits)
    if output:
        Path(output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        typer.echo(f"Report written to {output}")
    if "error" in report:
        typer.echo(f"ERROR: {report['error']}")
        raise typer.Exit(1)

    typer.echo("\n" + "=" * 60)
    typer.echo(f"CONFORMAL SELECTIVE HITL: {experiment}  (signal={signal})")
    typer.echo("=" * 60)
    typer.echo(f"labelled tasks: {report['n_labelled']}  base accuracy: {report['base_accuracy']:.2%}")
    typer.echo(f"target accepted-error alpha = {report['target_alpha']:.2%}")
    typer.echo(f"mean test accepted-risk : {report['mean_test_accepted_risk']}")
    typer.echo(f"mean test ask-rate      : {report['mean_test_ask_rate']}")
    typer.echo(f"mean test accept-accuracy: {report['mean_test_accepted_accuracy']}")
    typer.echo(f"guarantee holds out-of-sample: {report['guarantee_holds']}")
    if report.get("splits_with_no_region"):
        typer.echo(f"  ({report['splits_with_no_region']}/{report['n_splits']} splits had no "
                   "region meeting the target → ask everyone)")
    typer.echo("=" * 60)


@app.command("experiment-cost")
def experiment_cost_command(
    experiment: str = typer.Option(..., help="Experiment ID to total cost/tokens for."),
    output: str | None = typer.Option(None, help="Write full cost summary JSON here."),
) -> None:
    """Total token usage + estimated cost across an experiment's runs."""
    from semanticflow.experiment import summarize_cost

    try:
        report = summarize_cost(experiment)
    except FileNotFoundError as exc:
        typer.echo(f"ERROR: {exc}")
        raise typer.Exit(1) from exc
    if output:
        Path(output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        typer.echo(f"Report written to {output}")

    typer.echo("\n" + "=" * 60)
    typer.echo(f"EXPERIMENT COST: {experiment}")
    typer.echo("=" * 60)
    typer.echo(f"tasks: {report['n_tasks']}  |  total tokens: {report['total_tokens']:,.0f}  "
               f"|  total latency: {report['total_latency_sec']:.1f}s")
    typer.echo(f"TOTAL COST: ${report['total_cost_usd']:.4f}  "
               f"(${report['avg_cost_per_task_usd']:.4f}/task)")
    if not report["by_provider"]:
        typer.echo("\n⚠️  no usage recorded (mock run, or rows predate usage capture).")
    for provider, u in sorted(report["by_provider"].items()):
        typer.echo(f"\n  {provider}: ${u['cost_usd']:.4f}  "
                   f"| {u['total_tokens']:,.0f} tok (in {u['input_tokens']:,.0f} / "
                   f"out {u['output_tokens']:,.0f})  | {u['calls']:,.0f} calls")
    if report["total_cost_usd"] == 0 and report["total_tokens"] > 0:
        typer.echo("\n⚠️  cost is $0 but tokens were used — the model isn't in the pricing "
                   "table (semanticflow/llm/usage.py). Token counts are still accurate.")
    elif report["total_tokens"] == 0 and report["by_provider"]:
        typer.echo("\n⚠️  no tokens recorded (mock run, or rows predate usage capture).")
    typer.echo("=" * 60)


if __name__ == "__main__":
    app()
