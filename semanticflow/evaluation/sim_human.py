"""Scoped simulated human for HITL experiments.

The previous experiment harness answered every clarifying question with the entire
``expected_semantic`` dict — an oracle that leaks the full gold answer regardless of
what was asked, making question quality unmeasurable and inflating the with-HITL
condition to a perfect-information ceiling.

This module instead answers only the *slot* a question targets (measure, grouping,
time granularity, time window/filter, or limit/order), so a vague question extracts
nothing and a targeted one resolves exactly one ambiguity. An answer budget caps how
many questions the simulated human is willing to answer, so human effort is honest.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

# slot -> substrings that signal the question is about that slot
_SLOT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "measure": ("measure", "aggregat", "count", "sum ", "average", "avg",
                "calculat", "how much", "what metric", "which metric"),
    "aggregation": ("aggregat", "sum or", "or sum", "average or", "or average",
                    "count or", "or count", "sum, ", "how should", "which calculation"),
    "group_by": ("group", "dimension", "break down", "breakdown", "broken down",
                 "split", "per ", "by which"),
    "time_granularity": ("granularity", "grain", "daily", "weekly", "monthly",
                         "per day", "per week", "per month", "how often",
                         "over time", "trend"),
    "filters": ("period", "date range", "time range", "last ", "since", "window",
                "filter", "only", "exclude", "status", "which orders", "include"),
    "limit": ("top ", "limit", "how many rows", "rank", "first "),
}


def _get(expected: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in expected and expected[key] not in (None, [], ""):
            return expected[key]
    return None


def _render(slot: str, expected: dict[str, Any]) -> str | None:
    """Natural-language reveal of just one gold slot, or None if absent."""
    if slot == "measure":
        val = _get(expected, "base_measures", "base_measure")
        if val:
            items = val if isinstance(val, list) else [val]
            return f"Use the measure {', '.join(map(str, items))}."
    elif slot == "aggregation":
        val = _get(expected, "aggregation")
        if val:
            return f"Use a {val} aggregation."
    elif slot == "group_by":
        val = _get(expected, "group_by", "dimensions")
        if val:
            items = val if isinstance(val, list) else [val]
            reveal = f"Group by {', '.join(map(str, items))}."
            # Grouping and time grain are one decision in these specs (a grain without a
            # time grouping is meaningless), so a human asked "how should it be broken
            # down?" naturally states both ("by month over order date").
            gran = _get(expected, "time_granularity")
            if gran:
                reveal += f" Use {gran} granularity."
            return reveal
        return "Do not group by any dimension; return a single total."
    elif slot == "time_granularity":
        val = _get(expected, "time_granularity")
        if val:
            return f"Use {val} granularity."
        return "No specific time granularity is needed."
    elif slot == "filters":
        val = _get(expected, "filters")
        if val:
            items = val if isinstance(val, list) else [val]
            return f"Apply the filter(s): {', '.join(map(str, items))}."
        return "No filters; include all rows."
    elif slot == "limit":
        limit = _get(expected, "limit")
        order = _get(expected, "order_by")
        parts = []
        if order:
            parts.append(f"order by {order}")
        if limit:
            parts.append(f"limit to {limit} rows")
        if parts:
            return "; ".join(parts).capitalize() + "."
    return None


def scoped_human_answer(question: str, expected_semantic: dict[str, Any] | None) -> str:
    """Answer a clarifying question revealing ONLY the gold slot it asks about.

    Returns an empty string when the question doesn't map to a recognizable slot
    (modelling a human who cannot tell what is being asked) or when the targeted
    slot has no gold value.
    """
    if not expected_semantic:
        return ""
    q = question.lower()
    matched = [slot for slot, kws in _SLOT_KEYWORDS.items() if any(kw in q for kw in kws)]
    if not matched:
        return ""
    revealed = [r for r in (_render(slot, expected_semantic) for slot in matched) if r]
    return " ".join(revealed)


def make_simulated_human(
    expected_semantic: dict[str, Any] | None,
    budget: int = 3,
) -> Callable[[list[str]], dict[str, str]]:
    """Build a ``user_input_provider`` that answers at most ``budget`` informative
    questions (scoped to the queried slot); further questions go unanswered.

    The budget persists ACROSS calls: the same human may be consulted again later in the
    task (e.g. the post-execution-failure touchpoint), and total effort stays capped.

    Repeating an answer already given costs no budget: several questions routed to the
    same slot reveal the same fact, and charging for each repeat exhausted the budget
    before any new information was extracted."""
    answered = 0
    given: set[str] = set()

    def provider(questions: list[str]) -> dict[str, str]:
        nonlocal answered
        answers: dict[str, str] = {}
        for question in questions:
            if answered >= budget:
                answers[question] = ""
                continue
            answer = scoped_human_answer(question, expected_semantic)
            answers[question] = answer
            if answer and answer not in given:
                given.add(answer)
                answered += 1
        return answers

    return provider


_IDK = "I don't know."


def llm_human_answer(
    question: str,
    expected_semantic: dict[str, Any] | None,
    llm_client: Any,
    nl_request: str = "",
) -> str:
    """Free-form NL answer from a cheap LLM conditioned on the hidden gold intent.

    Unlike the keyword human, this breaks the canned-phrasing handshake with the
    deterministic refiner: answers are realistic prose the pipeline must actually parse.
    The prompt enforces the same slot-scoping contract (reveal ONLY what was asked) and
    permits an explicit "I don't know". Returns "" on any client failure so the caller
    degrades to the unanswered-question path."""
    if not expected_semantic:
        return ""
    import json

    system = (
        "You are simulating a business user who asked a data question and is now "
        "answering an analyst's clarifying question. You are not a data engineer: "
        "answer in plain English, one or two short sentences."
    )
    user = (
        f"You originally asked: \"{nl_request or '(your data question)'}\"\n\n"
        "Your true intent is captured by this specification (the analyst CANNOT see it):\n"
        f"{json.dumps(expected_semantic, default=str)}\n\n"
        f"The analyst asks: \"{question}\"\n\n"
        "Rules:\n"
        "- Reveal ONLY the part of your intent the question explicitly asks about; do not "
        "volunteer other parts of the specification.\n"
        "- If the question asks about something with no value in your intent, say you "
        "don't need it (e.g. 'No filters needed').\n"
        f"- If you cannot tell what is being asked, reply exactly: {_IDK}\n"
        "Answer:"
    )
    try:
        raw = llm_client.complete(system, user, temperature=0.0)
    except Exception:
        return ""
    return (raw or "").strip()


def llm_human_answer_batch(
    questions: list[str],
    expected_semantic: dict[str, Any] | None,
    llm_client: Any,
    nl_request: str = "",
) -> dict[str, str]:
    """Answer ALL clarifying questions in a SINGLE LLM call — one realistic clarification
    round (the analyst sends the questions, the user replies once addressing all of them).
    This removes the per-question budget/ordering bottleneck of one-at-a-time answering and
    is also cheaper (one call, not N). Still slot-scoped: reveal only what each question
    asks, ``IDK`` for anything not in the intent. Returns {question: answer}."""
    if not expected_semantic or not questions:
        return {q: "" for q in questions}
    import json
    import re

    numbered = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
    system = (
        "You are simulating a business user who asked a data question and is now answering "
        "an analyst's clarifying questions. You are not a data engineer: answer in plain "
        "English."
    )
    user = (
        f'You originally asked: "{nl_request or "(your data question)"}"\n\n'
        "Your true intent is captured by this specification (the analyst CANNOT see it):\n"
        f"{json.dumps(expected_semantic, default=str)}\n\n"
        "The analyst asks these clarifying questions:\n"
        f"{numbered}\n\n"
        "Rules:\n"
        "- Answer EVERY question.\n"
        "- For each, reveal ONLY the part of your intent that question asks about; do not "
        "volunteer other parts of the specification.\n"
        "- If a question asks about something with no value in your intent, say you do not "
        "need it (e.g. 'No filters needed').\n"
        f"- If you cannot tell what a question asks, answer it exactly: {_IDK}\n"
        'Return ONLY a JSON object mapping each question number (as a string) to your '
        'one-sentence plain-English answer, e.g. {"1": "...", "2": "..."}.'
    )
    try:
        raw = llm_client.complete(system, user, temperature=0.0)
    except Exception:
        return {q: "" for q in questions}
    answers: dict[str, str] = {}
    m = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            for i, q in enumerate(questions):
                v = obj.get(str(i + 1), obj.get(i + 1, ""))
                answers[q] = str(v).strip() if v is not None else ""
        except Exception:
            answers = {}
    return {q: answers.get(q, "") for q in questions}


def make_llm_simulated_human(
    expected_semantic: dict[str, Any] | None,
    llm_client: Any,
    budget: int = 3,
    nl_request: str = "",
    batch: bool = True,
) -> Callable[[list[str]], dict[str, str]]:
    """LLM-backed ``user_input_provider``. By default ``batch=True``: all questions are
    answered in ONE call (a single realistic clarification round). Set ``batch=False`` for
    the legacy one-at-a-time mode with a per-question informative-answer ``budget``."""
    answered = 0
    given: set[str] = set()

    def provider(questions: list[str]) -> dict[str, str]:
        nonlocal answered
        if batch:
            return llm_human_answer_batch(questions, expected_semantic, llm_client, nl_request)
        answers: dict[str, str] = {}
        for question in questions:
            if answered >= budget:
                answers[question] = ""
                continue
            answer = llm_human_answer(question, expected_semantic, llm_client, nl_request)
            answers[question] = answer
            key = answer.strip().lower()
            if answer and key != _IDK.lower() and key not in given:
                given.add(key)
                answered += 1
        return answers

    return provider


def make_sim_human(
    expected_semantic: dict[str, Any] | None,
    budget: int = 3,
    mode: str = "keyword",
    llm_client: Any = None,
    nl_request: str = "",
) -> Callable[[list[str]], dict[str, str]]:
    """Build the simulated human for an experiment arm.

    mode="llm" requires a client; without one it falls back to the keyword human so a
    missing API key degrades gracefully instead of silently changing the experiment."""
    if mode == "llm" and llm_client is not None:
        return make_llm_simulated_human(
            expected_semantic, llm_client, budget=budget, nl_request=nl_request
        )
    return make_simulated_human(expected_semantic, budget=budget)
