from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class Task(BaseModel):
    task_id: str
    nl_request: str
    schema_context: dict
    expected_semantic: dict | None = None
    # Multi-reference (multi-gold) support for genuinely ambiguous / underspecified tasks
    # (AmbigQA EMNLP'20; AmbiQT EMNLP'23; Papicchio et al. DBML'24). A single
    # ``expected_semantic`` is still scored exactly as before; when ``acceptable_semantics``
    # is set, a spec is scored as the MAX over {expected_semantic} ∪ acceptable_semantics so
    # any *valid* disambiguation wins. ``acceptable_results`` does the same for execution: a
    # result is correct if it execution-matches ``gold_query_result`` OR any acceptable table.
    # Both are OPTIONAL and backward-compatible (None ⇒ original single-gold behaviour).
    acceptable_semantics: list[dict] | None = None
    acceptable_results: list[list[dict[str, Any]]] | None = None
    gold_sql: str | None = None
    gold_query_result: list[dict[str, Any]] | dict[str, Any] | None = None
    result_schema: dict | None = None
    compare_rules: dict | None = None
    difficulty: str = "unknown"
    ambiguity: str = "unknown"
    dbt_project: str | None = None


def _iter_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield line


def load_tasks(
    path: str,
    difficulty: str | None = None,
    ambiguity: str | None = None,
) -> list[Task]:
    tasks: list[Task] = []
    for line in _iter_lines(Path(path)):
        payload = json.loads(line)
        task = Task.model_validate(payload)
        if difficulty and task.difficulty != difficulty:
            continue
        if ambiguity and task.ambiguity != ambiguity:
            continue
        tasks.append(task)
    return tasks
