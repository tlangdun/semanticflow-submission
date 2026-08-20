from __future__ import annotations

from typing import Any, TypedDict

from pydantic import BaseModel, ConfigDict

from semanticflow.tasks.loader import Task


class AgentState(TypedDict, total=False):
    task: Task
    semantic_spec: dict[str, Any]
    design_proposals: dict[str, Any]
    generated_files: list[str]
    merge_operations: list[dict[str, str]]
    dbt_results: dict[str, Any]
    errors: list[str]
    warnings: list[str]
    uncertainty_score: float
    ambiguity_points: list[str]
    clarifying_questions: list[str]
    model_outputs: dict[str, dict[str, Any]]
    model_confidences: dict[str, float]
    epistemic_uncertainty: float
    semantic_agreement_score: float
    ambiguity: str
    ambiguity_reason: str
    consensus_method: str
    time_spine: dict[str, Any]
    mf_validate_success: bool
    mf_validate_stdout: str
    mf_validate_stderr: str
    mf_validate_issues: list[dict[str, Any]]
    mf_query_success: bool
    mf_query_stdout: str
    mf_query_stderr: str
    mf_query_result: list[dict[str, Any]]
    gold_sql_present: bool
    gold_sql_exec_success: bool
    gold_result: list[dict[str, Any]]
    execution_accuracy: float
    execution_match: bool
    result_diff_summary: dict[str, Any]
    evaluation: dict[str, Any]


class AgentMessage(BaseModel):
    task: Task
    state: dict[str, Any]


class Agent(BaseModel):
    name: str
    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def run(self, msg: AgentMessage) -> AgentMessage:  # pragma: no cover - interface
        raise NotImplementedError
