from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypedDict

from semanticflow.agents.ambiguity_analyzer import (
    AmbiguityAnalyzerAgent,
    clarified_slots,
    ensure_slot_coverage,
    prioritize_questions,
)
from semanticflow.agents.codegen import CodegenAgent
from semanticflow.agents.designer import DesignerAgent
from semanticflow.agents.error_recovery import analyze_errors, apply_recovery
from semanticflow.agents.evaluator import EvaluatorAgent
from semanticflow.agents.executor import ExecutorAgent
from semanticflow.agents.result_validator import validate_query_result
from semanticflow.agents.semantic_mapper import SemanticMapperAgent, guard_agreed_slots
from semanticflow.agents.types import AgentMessage
from semanticflow.config import Settings
from semanticflow.dbt_integration.project_inspect import duckdb_data_types
from semanticflow.evaluation.spec_verification import verify_spec, violations_to_questions
from semanticflow.evaluation.verification_repair import auto_repair_from_verification
from semanticflow.evaluation.execution_selection import Candidate, select_by_execution
from semanticflow.agents.execution_repair import (
    clarifying_questions_from_error,
    repair_spec_from_execution,
)
from semanticflow.llm import (
    BaseLLMClient,
    LLMError,
    MockClient,
    UsageTracker,
    build_cheap_client,
    build_provider_client,
)
from semanticflow.log import get_logger
from semanticflow.tasks.loader import Task

logger = get_logger("semanticflow.orchestrator")


class GraphState(TypedDict):
    task: Task
    state: dict[str, Any]


@dataclass
class OrchestrationResult:
    task_id: str
    mode: str
    state: dict[str, Any]
    clarifying_questions: list[str]
    runtime_sec: float
    ambiguity_detected: bool = False
    clarifying_questions_asked: list[str] = field(default_factory=list)
    user_answers: dict[str, str] | None = None
    auto_fixes_applied: list[str] = field(default_factory=list)
    validation_issues: list[dict[str, Any]] = field(default_factory=list)
    refinement_iterations: int = 0


class Orchestrator:
    def __init__(self, settings: Settings, llm_client: BaseLLMClient | None = None) -> None:
        self.settings = settings
        self.llm_client = llm_client
        # One shared tracker across all providers/agents; reset per task in run_task.
        self.usage_tracker = UsageTracker()

        llm_clients: dict[str, BaseLLMClient] = {}
        for provider in settings.ensemble_providers:
            try:
                llm_clients[provider] = build_provider_client(provider, settings, self.usage_tracker)
            except LLMError:
                llm_clients[provider] = MockClient()
        if all(isinstance(c, MockClient) for c in llm_clients.values()):
            logger.warning(
                "All LLM providers fell back to MockClient (no usable API keys); "
                "outputs are placeholders and experiment results will not be meaningful."
            )

        self.semantic_mapper = SemanticMapperAgent(settings=settings, llm_clients=llm_clients)
        single_provider = settings.llm_provider or "mock"
        try:
            single_client = build_provider_client(single_provider, settings, self.usage_tracker)
        except LLMError:
            single_client = MockClient()
        self.semantic_mapper_single = SemanticMapperAgent(
            settings=settings,
            llm_clients={single_provider: single_client},
        )
        # Cheap-tier client for low-stakes calls: clarifying-question generation and the
        # last-resort execution repair. None when no provider key is usable — every
        # caller degrades to its deterministic path.
        self.cheap_llm_client = build_cheap_client(settings, self.usage_tracker)
        analyzer_client = (
            self.cheap_llm_client or llm_clients.get("openai") or llm_clients.get("anthropic")
        )
        self.ambiguity_analyzer = AmbiguityAnalyzerAgent(settings=settings, llm_client=analyzer_client)
        # Pass LLM client to designer for intelligent measure-to-column mapping
        designer_client = llm_clients.get("openai") or llm_clients.get("anthropic") or llm_clients.get("mock")
        self.designer = DesignerAgent(llm_client=designer_client)
        self.codegen = CodegenAgent()
        self.executor = ExecutorAgent(settings=settings)
        self.evaluator = EvaluatorAgent(settings=settings)

    def _reset_generated_semantic_layer(self, task: Task) -> None:
        from pathlib import Path

        project_dir = task.dbt_project or self.settings.dbt_project_dir
        semantic_dir = Path(project_dir) / "models" / "semantic"
        for fname in ("semantic_models.yml", "metrics.yml"):
            path = semantic_dir / fname
            if path.exists():
                try:
                    path.unlink()
                except OSError as exc:  # pragma: no cover - fs dependent
                    logger.warning(f"Could not reset {path}: {exc}")

    async def _run_agent(self, agent, state: GraphState) -> GraphState:
        msg = AgentMessage(task=state["task"], state=state.get("state", {}))
        msg = await agent.run(msg)
        return {"task": msg.task, "state": msg.state}

    # The pipeline is a fixed linear chain; all branching (HITL, recovery retries,
    # baseline mapper swap) lives in run_task, so a graph framework added no value over a
    # plain async loop. Stages can be entered partway (e.g. "designer") to re-run after
    # a HITL refinement or an applied auto-fix without redoing the mapper.
    _PIPELINE = ("semantic_mapper", "designer", "codegen", "executor", "evaluator")

    async def _run_pipeline(self, task: Task, state: dict[str, Any], start_at: str) -> dict[str, Any]:
        agents = {
            "semantic_mapper": self.semantic_mapper,
            "designer": self.designer,
            "codegen": self.codegen,
            "executor": self.executor,
            "evaluator": self.evaluator,
        }
        gs: GraphState = {"task": task, "state": state}
        started = False
        for name in self._PIPELINE:
            if name == start_at:
                started = True
            if started:
                gs = await self._run_agent(agents[name], gs)
        return gs.get("state", {})

    def _compute_execution_consistency(
        self, task: Task, model_outputs: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """Execute each provider's representative spec and cluster the result tables by
        execution-equivalence (the strongest black-box UQ signal for text-to-SQL).

        Each provider's spec is run designer->codegen->executor on a FRESH semantic layer
        (reset between providers so codegen never merges incompatible models — the bug
        that produced "Frankenstein" specs). The clustered ``execution_uncertainty`` is
        returned; the caller is responsible for restoring the consensus artifacts.

        PENDING LIVE VALIDATION: exercises the full dbt/mf path k times and could not be
        smoke-tested under the API-credit freeze. Default-off via
        ``settings.execution_consistency``. The clustering math is unit-tested
        (tests/test_execution_consistency.py); only the per-provider re-execution wiring
        here is unverified end-to-end.
        """
        from semanticflow.evaluation.execution_consistency import execution_consistency

        providers = [p for p, spec in model_outputs.items() if spec]
        tables: list[list[dict[str, Any]] | None] = []
        successes: list[bool] = []
        for provider in providers:
            self._reset_generated_semantic_layer(task)
            sub_state = asyncio.run(
                self._run_pipeline(
                    task, {"semantic_spec": dict(model_outputs[provider])}, start_at="designer"
                )
            )
            ok = bool(sub_state.get("mf_query_success"))
            successes.append(ok)
            tables.append(sub_state.get("mf_query_result") if ok else None)

        ec = execution_consistency(tables, successes)
        # Execution-grounded answer selection: the per-provider executions above are the same
        # data the consistency signal uses — reuse them to PICK the final answer from the
        # largest execution-equivalence cluster (and never an errored/empty result when a
        # valid one exists), instead of relying solely on the text consensus vote.
        candidates = [
            Candidate(
                provider=provider,
                spec=dict(model_outputs[provider]),
                result_rows=tables[i],
                mf_query_success=successes[i],
            )
            for i, provider in enumerate(providers)
        ]
        sel = select_by_execution(candidates, compare_rules=getattr(task, "compare_rules", None))
        return {
            "execution_uncertainty": ec.execution_uncertainty,
            "execution_agreement": ec.agreement,
            "execution_n_clusters": ec.n_clusters,
            "execution_n_failed": ec.n_failed,
            "execution_cluster_sizes": ec.cluster_sizes,
            "execution_consistency_providers": providers,
            "execution_selection_valid": sel.valid,
            "execution_selection_reason": sel.reason,
            "execution_selection_provider": sel.provider,
            "execution_selection_agreement": sel.agreement,
            "execution_selected_spec": sel.spec,
        }

    def run_task(
        self,
        task: Task,
        mode: str | None = None,
        hitl_mode: bool | None = None,
        user_input_provider: Callable[[list[str]], dict[str, str]] | None = None,
        ambiguity_analyzer: AmbiguityAnalyzerAgent | None = None,
    ) -> OrchestrationResult:
        mode = mode or self.settings.mode
        if hitl_mode is None:
            hitl_mode = mode == "multi_agent_hitl"
        analyzer = ambiguity_analyzer or self.ambiguity_analyzer

        start = time.time()
        # The benchmark tasks' static schema_context has column NAMES only; without types,
        # date columns whose names carry no time token (customers.first_order) are declared
        # categorical and query-time granularity can never be applied to them. Enrich from
        # the warehouse (cached, read-only) so every arm sees the same typed schema.
        if not task.schema_context.get("data_types"):
            types = duckdb_data_types(self.settings.dbt_project_dir)
            if types:
                task.schema_context["data_types"] = types
        # Each benchmark task is independent: reset the generated semantic layer so codegen
        # writes fresh instead of merging in stale models/dimensions from previous tasks
        # (which left a synthetic `ds=current_date` agg_time_dimension and leftover models).
        self._reset_generated_semantic_layer(task)
        # Token/cost accounting is per task: zero the shared tracker before this task's
        # LLM calls so the snapshot logged below covers only this task.
        self.usage_tracker.reset()
        state: dict[str, Any] = {}
        ambiguity_detected = False
        clarifying_questions_asked: list[str] = []
        user_answers: dict[str, str] | None = None
        hitl_refined = False

        if mode == "baseline_single_agent":
            # Run the FULL pipeline (mapper→designer→codegen→executor→evaluator) with the
            # single-provider mapper swapped in — so the baseline is execution-scorable and the
            # only variable vs the multi-agent treatment is single- vs multi-provider mapping.
            # Restore in finally so a raised run can't leave the orchestrator permanently
            # swapped (which would corrupt every subsequent task in the same process).
            self.semantic_mapper, self.semantic_mapper_single = self.semantic_mapper_single, self.semantic_mapper
            try:
                state = asyncio.run(self._run_pipeline(task, {}, start_at="semantic_mapper"))
            finally:
                self.semantic_mapper, self.semantic_mapper_single = self.semantic_mapper_single, self.semantic_mapper
        else:
            mapper_state = asyncio.run(
                self._run_agent(self.semantic_mapper, {"task": task, "state": {}})
            )
            state = mapper_state.get("state", {})
            self_reported_ambiguity = state.get("ambiguity") == "high"

            # Verification-grounded trigger: deterministic schema checks on the generated
            # spec. This is the only signal with AUROC > 0.5 (the model's self-reported
            # ambiguity is uniform/uncalibrated), and it fires high-precision on STRUCTURAL
            # ambiguity — relative-time on static data, ratio not expressible, unknown
            # column, missing grain. Its violations are schema-grounded clarifying prompts.
            verification = verify_spec(
                task.nl_request, state.get("semantic_spec", {}), task.schema_context
            )
            state["verification_violations"] = verification.violations
            state["verification_score"] = verification.score

            # Hybrid uncertainty trigger. total_uncertainty (aleatoric + cross-model
            # structural/epistemic) is the strongest signal (AUROC 0.79 with the 3-provider
            # ensemble; ~0.50 single-provider, so this fires meaningfully only with >1
            # provider). It catches SEMANTIC vagueness ("How's business?") — models disagree
            # on a structurally-valid spec — which verification (structural only) cannot see.
            total_uncertainty = float(state.get("total_uncertainty") or 0.0)
            uncertainty_threshold = float(getattr(self.settings, "uncertainty_threshold", 0.5))
            uncertainty_high = total_uncertainty >= uncertainty_threshold
            state["uncertainty_triggered"] = uncertainty_high

            ambiguity_detected = (
                self_reported_ambiguity or verification.score > 0 or uncertainty_high
            )

            if ambiguity_detected and analyzer:
                # Spend an LLM call on question generation only when a human will
                # actually answer (HITL arm with a provider); otherwise the rule-based
                # questions are recorded for observability at zero API cost.
                analysis = analyzer.analyze(
                    task.nl_request,
                    task.schema_context,
                    state.get("model_outputs", {}),
                    state.get("epistemic_uncertainty", 0.0),
                    use_llm=bool(hitl_mode and user_input_provider),
                )
                state["ambiguity_points"] = analysis.ambiguity_points
                state["clarifying_questions"] = analysis.clarifying_questions
                state["ambiguity_reason"] = analysis.explanation

            # Surface verification violations as clarifying questions even when the semantic
            # analyzer found nothing — but phrased as answerable QUESTIONS via deterministic
            # templates ("What absolute date range...?"), not raw diagnostics, so whether
            # the human can answer no longer depends on the diagnostic's wording.
            if verification.violations:
                # Verification questions go FIRST: the simulated human's answer budget
                # is consumed in question order, and these schema-grounded questions
                # (absolute date range, missing grain, unknown column) target the
                # failures the analyzer's generic questions cannot resolve. Appending
                # them last meant the budget was often exhausted before they were asked.
                qs = violations_to_questions(verification.violations)
                qs.extend(
                    q for q in (state.get("clarifying_questions") or []) if q not in qs
                )
                state["clarifying_questions"] = qs

            if hitl_mode and ambiguity_detected and user_input_provider:
                # Coverage backstop LAST (after verification-first + analyzer questions):
                # a slot no question targets can never be revealed by the human.
                state["clarifying_questions"] = ensure_slot_coverage(
                    state.get("clarifying_questions") or []
                )
                # Ask the shape-determining slots (measure, grouping, grain) first so a
                # budget-limited human answers the questions that decide correctness, rather
                # than spending the budget on filter/sorting questions asked earlier.
                state["clarifying_questions"] = prioritize_questions(
                    state["clarifying_questions"]
                )
                clarifying_questions_asked = state.get("clarifying_questions", [])
                user_answers = user_input_provider(clarifying_questions_asked)
                # All answers empty = the clarification round FAILED (unanswerable
                # questions or exhausted budget). Record it instead of proceeding as if
                # nothing happened — this separates "asked and got nothing" from "did
                # not ask" in the analysis.
                if user_answers and not any((a or "").strip() for a in user_answers.values()):
                    state["clarification_failed"] = True
                    logger.info(
                        f"Task {task.task_id}: clarification round yielded no usable answers"
                    )
                else:
                    # Only refine when something was actually answered: refining on
                    # all-empty answers burns the refiner's LLM-fallback call for nothing
                    # and risks perturbing a spec no answer asked to change.
                    _pre_refine = state.get("semantic_spec", {})
                    refined = self.semantic_mapper.refine_with_feedback(
                        _pre_refine,
                        task.nl_request,
                        user_answers,
                        task.schema_context,
                    )
                    # Slot-scoped guard: revert refinements that overwrite a slot all
                    # ensemble providers already agreed on (these harm correct tasks).
                    # consensus_spec = the PRE-refinement (consensus) spec captured above.
                    if getattr(self.settings, "slot_scoped_hitl", True):
                        refined, _reverted = guard_agreed_slots(
                            _pre_refine, refined, state.get("model_outputs"),
                            clarified_slots=clarified_slots(user_answers),
                        )
                        if _reverted:
                            state["slot_guard_reverted"] = _reverted
                    state["semantic_spec"] = refined
                    hitl_refined = refined != _pre_refine

            # Verifier-guided repair: re-run the schema verifier on the finalised spec (after
            # any HITL incorporation) and deterministically auto-apply the structural fixes
            # that do not need a human (missing grain, relative-time->absolute, close-match
            # column), before the final execution. No-op when nothing is auto-fixable.
            _final_verif = verify_spec(
                task.nl_request, state.get("semantic_spec", {}), task.schema_context
            )
            _repaired_spec, _vfixes = auto_repair_from_verification(
                state.get("semantic_spec", {}), _final_verif.violations, task.schema_context
            )
            if _vfixes:
                state["semantic_spec"] = _repaired_spec
                state["verification_repairs"] = _vfixes

            state = asyncio.run(self._run_pipeline(task, state, start_at="designer"))

        # Post-execution: Error recovery with retry loop
        auto_fixes_applied: list[str] = []
        validation_issues: list[dict[str, Any]] = []
        refinement_iterations = 0
        # Hard cap at 2 to prevent runaway API costs
        max_recovery_attempts = min(getattr(self.settings, "max_recovery_attempts", 2), 2)

        # Retry loop for error recovery
        for attempt in range(max_recovery_attempts):
            mf_validate_success = state.get("mf_validate_success", False)
            mf_query_success = state.get("mf_query_success", False)
            
            if mf_validate_success and mf_query_success:
                break  # Success - no need to retry
            
            stderr = state.get("mf_validate_stderr", "") + state.get("mf_query_stderr", "")
            stdout = state.get("mf_validate_stdout", "") + state.get("mf_query_stdout", "")
            
            if not stderr and not stdout:
                break  # No error info to analyze
            
            error_analysis = analyze_errors(
                stderr, stdout,
                semantic_spec=state.get("semantic_spec"),
                schema_context=task.schema_context,
            )
            state["error_analysis"] = {
                "errors": error_analysis.errors,
                "can_auto_recover": error_analysis.can_auto_recover,
                "summary": error_analysis.summary,
            }
            
            if not error_analysis.can_auto_recover:
                logger.info(f"Errors not auto-recoverable after attempt {attempt + 1}")
                state["error_recovery_exhausted"] = True
                break
            
            logger.info(f"Attempting auto-recovery (attempt {attempt + 1}/{max_recovery_attempts})")
            updated_spec, updated_design, fixes = apply_recovery(
                error_analysis,
                state.get("semantic_spec", {}),
                state.get("design_proposals", {}),
            )
            
            if not fixes:
                logger.info("No fixes could be applied")
                state["error_recovery_exhausted"] = True
                break
            
            auto_fixes_applied.extend(fixes)
            state["semantic_spec"] = updated_spec
            state["design_proposals"] = updated_design
            refinement_iterations += 1
            
            # Re-run from the designer (not codegen) with the fixed spec: codegen reads
            # design_proposals, which the designer rebuilds from semantic_spec — so spec
            # fixes are lost if we re-enter at codegen.
            logger.info(f"Re-running pipeline with {len(fixes)} fixes applied")
            state = asyncio.run(self._run_pipeline(task, state, start_at="designer"))

        # Execution-based consistency signal (default-off; costly extra dbt/mf runs).
        # Run AFTER the consensus pipeline, then restore the consensus semantic layer so
        # the logged artifacts still reflect the consensus run, not the last sub-run.
        if (
            getattr(self.settings, "execution_consistency", False)
            and mode != "baseline_single_agent"
            and state.get("model_outputs")
        ):
            try:
                ec = self._compute_execution_consistency(task, state["model_outputs"])
                state.update(ec)
                # If execution-grounded selection found a provider whose EXECUTED result is
                # the agreed/valid answer, adopt its spec as the consensus: the restore step
                # below re-runs designer->codegen->executor on state["semantic_spec"], so the
                # logged artifacts and query result reflect the selected (working) answer.
                #
                # EXCEPTION: the per-provider candidates were sampled BEFORE clarification.
                # When HITL answers changed the consensus spec and that spec executed to a
                # non-empty result, adopting a pre-HITL candidate would silently discard the
                # human's answers — so the human-informed spec wins. When the refined spec
                # itself failed execution, selection proceeds, and the answers are re-applied
                # deterministically on top of the adopted spec.
                if ec.get("execution_selection_valid") and ec.get("execution_selected_spec"):
                    _consensus_executed = bool(
                        state.get("mf_query_success") and state.get("mf_query_result")
                    )
                    if hitl_refined and _consensus_executed:
                        state["execution_selection_overridden_by_hitl"] = True
                    else:
                        state["semantic_spec"] = dict(ec["execution_selected_spec"])
                        if hitl_refined and user_answers:
                            _reapplied, _changed = (
                                self.semantic_mapper._apply_answers_deterministic(
                                    state["semantic_spec"], user_answers, task.schema_context
                                )
                            )
                            if _changed:
                                state["semantic_spec"] = _reapplied
                                auto_fixes_applied.append("hitl_reapplied_after_selection")
                # Combine within/cross-model spec disagreement with execution disagreement
                # into a single hybrid signal for the sweep/conformal layers to trigger on.
                state["total_uncertainty_exec"] = (
                    float(state.get("total_uncertainty", 0.0)) + ec["execution_uncertainty"]
                )
            except Exception as exc:  # pragma: no cover - live dbt/mf dependent
                logger.warning(f"execution-consistency signal failed: {exc}")
            finally:
                # Restore consensus artifacts: reset + re-run the consensus spec so the
                # on-disk generated files and `state` query results stay consistent.
                self._reset_generated_semantic_layer(task)
                state = asyncio.run(self._run_pipeline(task, state, start_at="designer"))

        # Execution-feedback repair: if the final answer still errored or returned empty
        # (after recovery and execution-grounded selection), feed the MetricFlow error back,
        # deterministically repair the spec (unknown-metric suggestion, fan-out->co-located
        # source model, relative-time->absolute date range) and re-run. Bounded; no-op on
        # success. Deterministic first; when deterministic repair finds nothing, ONE
        # cheap-tier LLM repair call is allowed per task (the error text is the most
        # informative repair context available, and this path only spends on tasks that
        # already failed every free fix).
        _llm_repair_used = False
        for _ in range(min(getattr(self.settings, "max_recovery_attempts", 2), 2)):
            if state.get("mf_query_success") and state.get("mf_query_result"):
                break
            _mf_error = (
                (state.get("mf_query_stderr") or "")
                + (state.get("mf_query_stdout") or "")
                + (state.get("mf_validate_stderr") or "")
            )
            _before = state.get("semantic_spec", {})
            _repaired = repair_spec_from_execution(
                _before, _mf_error, task.schema_context, nl_request=task.nl_request,
                llm_client=None,
            )
            _fix_label = "execution_feedback_repair"
            if _repaired == _before:
                if (
                    _llm_repair_used
                    or not getattr(self.settings, "llm_execution_repair", True)
                    or self.cheap_llm_client is None
                ):
                    break
                _llm_repair_used = True
                _repaired = repair_spec_from_execution(
                    _before, _mf_error, task.schema_context, nl_request=task.nl_request,
                    llm_client=self.cheap_llm_client,
                )
                if _repaired == _before:
                    break
                _fix_label = "execution_feedback_repair_llm"
            auto_fixes_applied.append(_fix_label)
            state["semantic_spec"] = _repaired
            state = asyncio.run(self._run_pipeline(task, state, start_at="designer"))
        if not (state.get("mf_query_success") and state.get("mf_query_result")):
            # All repair avenues spent (or never applicable) and the answer is still an
            # error/empty — flag it so "repair exhausted" is distinguishable from
            # "never failed" in the failure-mode analysis.
            state["execution_repair_exhausted"] = True

        # Post-execution HITL: pre-execution clarification asks about intent the mapper
        # was unsure of; this asks about the one thing only execution could reveal — the
        # concrete failure. Deterministic question (zero LLM), the same budgeted human,
        # one extra designer->executor re-run only when the answer changed the spec.
        if (
            hitl_mode
            and user_input_provider
            and getattr(self.settings, "post_execution_hitl", True)
            and not (state.get("mf_query_success") and state.get("mf_query_result"))
        ):
            _mf_error = (
                (state.get("mf_query_stderr") or "")
                + (state.get("mf_query_stdout") or "")
                + (state.get("mf_validate_stderr") or "")
            )
            post_questions = clarifying_questions_from_error(_mf_error)
            post_answers = user_input_provider(post_questions)
            clarifying_questions_asked = list(clarifying_questions_asked) + post_questions
            user_answers = {**(user_answers or {}), **post_answers}
            state["post_execution_hitl"] = {
                "questions": post_questions,
                "answered": any((a or "").strip() for a in post_answers.values()),
                "spec_changed": False,
            }
            if state["post_execution_hitl"]["answered"]:
                _before = state.get("semantic_spec", {})
                _refined = self.semantic_mapper.refine_with_feedback(
                    _before, task.nl_request, post_answers, task.schema_context
                )
                # Slot-scoped guard: protect slots all providers agreed on from being
                # overwritten by this post-execution refinement (consensus_spec = _before).
                if getattr(self.settings, "slot_scoped_hitl", True):
                    _refined, _reverted = guard_agreed_slots(
                        _before, _refined, state.get("model_outputs"),
                        clarified_slots=clarified_slots(post_answers),
                    )
                    if _reverted:
                        auto_fixes_applied.append(
                            "slot_guard_reverted:" + ",".join(_reverted)
                        )
                if _refined != _before:
                    state["semantic_spec"] = _refined
                    state["post_execution_hitl"]["spec_changed"] = True
                    auto_fixes_applied.append("post_execution_hitl_refinement")
                    state = asyncio.run(self._run_pipeline(task, state, start_at="designer"))

        # Validate query results
        query_results = state.get("mf_query_result", [])
        if query_results:
            semantic_spec = state.get("semantic_spec", {})
            time_dim = semantic_spec.get("group_by", [None])[0] if semantic_spec.get("group_by") else None
            
            validation = validate_query_result(
                query_results,
                semantic_spec=semantic_spec,
                time_dimension=time_dim,
            )
            state["result_validation"] = {
                "is_valid": validation.is_valid,
                "row_count": validation.row_count,
                "summary": validation.summary,
                "suggestions": validation.suggestions,
            }
            validation_issues = [
                {"severity": i.severity.value, "check": i.check_name, "message": i.message}
                for i in validation.issues
            ]

        # Surface designer warnings (e.g. synthetic current_date time dimension, or a
        # metric emitted with no valid measure) so degraded specs are visible rather than
        # silently producing untrustworthy results.
        design_warning = state.get("design_proposals", {}).get("design_warning")
        if design_warning:
            validation_issues.append(
                {"severity": "warning", "check": "design", "message": design_warning}
            )

        # Pre-flight violations from codegen (deterministic checks on the written design:
        # metric-with-no-measure, unknown column references) — surfaced so a degraded
        # design is diagnosable without reading the dbt/mf logs.
        for violation in state.get("preflight_violations") or []:
            validation_issues.append(
                {"severity": "warning", "check": "preflight", "message": violation}
            )

        # Per-task LLM token usage + estimated cost, captured by the shared tracker.
        state["llm_usage"] = self.usage_tracker.snapshot()

        runtime_sec = time.time() - start
        clarifying = state.get("clarifying_questions") or []
        
        logger.info(
            f"Task {task.task_id} completed in {runtime_sec:.2f}s",
            auto_fixes=len(auto_fixes_applied),
            validation_issues=len(validation_issues),
        )
        
        return OrchestrationResult(
            task_id=task.task_id,
            mode=mode,
            state=state,
            clarifying_questions=clarifying,
            runtime_sec=runtime_sec,
            ambiguity_detected=ambiguity_detected,
            clarifying_questions_asked=clarifying_questions_asked,
            user_answers=user_answers,
            auto_fixes_applied=auto_fixes_applied,
            validation_issues=validation_issues,
            refinement_iterations=refinement_iterations,
        )
