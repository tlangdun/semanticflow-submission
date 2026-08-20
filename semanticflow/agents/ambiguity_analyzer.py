from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from semanticflow.agents.types import Agent
from semanticflow.config import Settings
from semanticflow.llm import BaseLLMClient


class AmbiguityAnalysisResult(BaseModel):
    ambiguity_points: list[str] = Field(default_factory=list)
    clarifying_questions: list[str] = Field(default_factory=list)
    explanation: str = ""


def _generate_specific_fallback_questions(
    nl_request: str,
    schema_context: dict[str, Any],
    conflicting_specs: dict[str, dict[str, Any]],
) -> list[str]:
    """Generate specific clarifying questions based on detected conflicts."""
    questions: list[str] = []
    request_lower = nl_request.lower()
    
    # Extract what's available in schema
    models = schema_context.get("models", [])
    all_columns: set[str] = set()
    for cols in schema_context.get("columns", {}).values():
        all_columns.update(cols)
    
    # Check for metric ambiguity
    metrics_mentioned = set()
    for provider, spec in conflicting_specs.items():
        if isinstance(spec, dict):
            metrics_mentioned.add(spec.get("metric_name", ""))
    metrics_mentioned.discard("")
    
    if len(metrics_mentioned) > 1:
        options = ", ".join(f"'{m}'" for m in metrics_mentioned)
        questions.append(
            f"Which metric do you want? Options detected: {options}. "
            "Please specify the exact calculation (e.g., 'count of orders' or 'sum of revenue')."
        )
    
    # Check for measure ambiguity
    measures_mentioned = set()
    for provider, spec in conflicting_specs.items():
        if isinstance(spec, dict):
            for m in spec.get("base_measures", []) or []:
                measures_mentioned.add(m)
    
    if len(measures_mentioned) > 1:
        options = ", ".join(f"'{m}'" for m in measures_mentioned)
        questions.append(
            f"Which measure should be used? Options: {options}. "
            "Is this a COUNT, SUM, or another aggregation?"
        )
    
    # Check for dimension/grouping ambiguity
    dimensions_mentioned = set()
    for provider, spec in conflicting_specs.items():
        if isinstance(spec, dict):
            for d in spec.get("group_by", []) or []:
                dimensions_mentioned.add(d)
    
    if len(dimensions_mentioned) > 1:
        options = ", ".join(f"'{d}'" for d in dimensions_mentioned)
        questions.append(
            f"Which dimensions should the data be grouped by? Options: {options}."
        )
    
    # Check for time-related ambiguity
    time_keywords = ["day", "week", "month", "year", "daily", "weekly", "monthly"]
    if any(kw in request_lower for kw in time_keywords):
        granularities = set()
        for provider, spec in conflicting_specs.items():
            if isinstance(spec, dict) and spec.get("time_granularity"):
                granularities.add(spec.get("time_granularity"))
        
        if len(granularities) > 1:
            questions.append(
                f"What time granularity do you need? Options: {', '.join(granularities)}. "
                "(e.g., 'daily' means one row per day)"
            )
    
    # Check for filter ambiguity
    if "last" in request_lower or "recent" in request_lower:
        time_windows = set()
        for provider, spec in conflicting_specs.items():
            if isinstance(spec, dict):
                for f in spec.get("filters", []) or []:
                    if "day" in str(f).lower() or "month" in str(f).lower():
                        time_windows.add(str(f))
        
        if len(time_windows) > 1:
            questions.append(
                "What time period should be included? "
                "Please specify exactly (e.g., 'last 30 days', 'last 7 days', 'this month')."
            )
    
    # Fallback if no specific questions generated - use schema-aware questions
    # No hardcoded model names like "orders" or "customers"
    if not questions:
        # Find which models are mentioned in the request
        request_words = set(request_lower.replace("_", " ").split())
        
        for model in models:
            model_lower = model.lower()
            model_words = set(model_lower.replace("_", " ").split())
            # Check for word overlap (including singular/plural)
            model_variations = model_words | {w.rstrip("s") for w in model_words} | {w + "s" for w in model_words}
            
            if request_words & model_variations:
                # Find numeric columns in this model for measure suggestions
                model_cols = schema_context.get("columns", {}).get(model, [])
                numeric_cols = [c for c in model_cols if any(p in c.lower() for p in ["amount", "count", "total", "num", "value", "price", "qty"])]
                id_cols = [c for c in model_cols if c.lower().endswith("_id")]
                
                col_examples = numeric_cols[:3] if numeric_cols else id_cols[:2]
                if col_examples:
                    questions.append(
                        f"For {model} metrics: Do you want to COUNT rows, SUM a column ({', '.join(col_examples)}), or something else?"
                    )
                    break
        
        # Still no questions? Use generic schema-aware ones
        if not questions:
            questions = [
                f"What specific calculation do you need? (Available columns: {', '.join(list(all_columns)[:5])}...)",
                "Should results be grouped by time (daily/weekly/monthly) or by another dimension?",
            ]
    
    return questions[:3]  # Limit to 3 questions


# Question-set coverage backstop. The LLM analyzer's questions are stochastic: on vague
# requests ("How's business?") it sometimes asks only about measure/filters, and a slot
# nobody asks about can never be revealed by the human — the task is lost before
# refinement starts. Keywords mirror the sim-human's slot routing so the backstop
# questions are guaranteed answerable.
_GROUPING_COVERAGE = ("group", "break down", "breakdown", "broken down", "dimension",
                      "split", "single total")
_GRAIN_COVERAGE = ("granularity", "grain", "daily", "weekly", "monthly",
                   "over time", "trend")

GROUPING_BACKSTOP_QUESTION = (
    "Should the result be broken down by a dimension (for example by order date or "
    "by customer), or returned as a single total?"
)
GRAIN_BACKSTOP_QUESTION = (
    "What time granularity should the results use: daily, weekly, or monthly (if any)?"
)


def ensure_slot_coverage(questions: list[str]) -> list[str]:
    """Append deterministic backstop questions for structural slots no generated
    question covers (grouping, time grain). Zero-LLM."""
    out = list(questions)
    text = " ".join(q.lower() for q in out)
    if not any(k in text for k in _GROUPING_COVERAGE):
        out.append(GROUPING_BACKSTOP_QUESTION)
    if not any(k in text for k in _GRAIN_COVERAGE):
        out.append(GRAIN_BACKSTOP_QUESTION)
    return out


# Slot priorities for ordering clarifying questions. The result-table SHAPE is set by the
# measure, the grouping and the time grain; filters narrow it and sorting only reorders it.
# A human with a finite answer budget must therefore be asked the shape-determining
# questions FIRST, or the decisive slots go unanswered (the grouping/grain backstops were
# previously appended last and never reached under a realistic, budget-limited user).
_MEASURE_KW = ("metric", "measure", "calculat", "aggregat", "count of", "sum of",
               "total revenue", "which number", "what number")
_FILTER_KW = ("status", "completed", "filter", "time period", "date range", "which period",
              "all-time", "all historical", "last 30", "regardless of status", "absolute date")
_SORT_KW = ("sort", "sorted", "order by", "ascending", "descending", "newest", "oldest",
            "ranked", "rank ", "display", "show only", "include all", "zero")


def _question_priority(q: str) -> float:
    t = q.lower()
    if any(k in t for k in _GROUPING_COVERAGE):
        return 1.0  # grouping dimension
    if any(k in t for k in _GRAIN_COVERAGE):
        return 2.0  # time grain
    if any(k in t for k in _MEASURE_KW):
        return 0.0  # what to compute — most decisive
    if any(k in t for k in _SORT_KW):
        return 4.0  # ordering/display — least decisive
    if any(k in t for k in _FILTER_KW):
        return 3.0  # filters narrow but do not reshape
    return 2.5


def prioritize_questions(questions: list[str]) -> list[str]:
    """Stable-sort clarifying questions so the shape-determining slots (measure, grouping,
    grain) are asked before filter/sorting questions. Lets a budget-limited human answer the
    slots that actually decide correctness. Zero-LLM, order-stable within a priority band."""
    return sorted(questions, key=_question_priority)


def question_slot(question: str) -> str | None:
    """Map a clarifying question to the spec slot it targets (or None). Used to mark which
    slots the human explicitly clarified, so the slot guard does not revert a slot the human
    directly answered back to the providers' (possibly wrong) agreed default."""
    t = question.lower()
    if any(k in t for k in _GROUPING_COVERAGE):
        return "group_by"
    if any(k in t for k in _GRAIN_COVERAGE):
        return "time_granularity"
    if any(k in t for k in _MEASURE_KW):
        return "base_measures"
    if any(k in t for k in _FILTER_KW):
        return "filters"
    return None


def clarified_slots(user_answers: dict[str, str]) -> set[str]:
    """Slots the human substantively clarified (question routed to a slot AND answered
    non-empty). A measure clarification also covers the paired ``aggregation`` slot."""
    out: set[str] = set()
    for q, a in (user_answers or {}).items():
        if not (a or "").strip():
            continue
        slot = question_slot(q)
        if slot:
            out.add(slot)
            if slot == "base_measures":
                out.add("aggregation")
    return out


class AmbiguityAnalyzerAgent(Agent):
    name: str = "ambiguity_analyzer"
    settings: Settings
    llm_client: BaseLLMClient | None = None

    def analyze(
        self,
        nl_request: str,
        schema_context: dict[str, Any],
        conflicting_specs: dict[str, dict[str, Any]],
        epistemic_uncertainty: float,
        use_llm: bool = True,
    ) -> AmbiguityAnalysisResult:
        # Generate specific fallback questions first (used if LLM fails)
        fallback_questions = _generate_specific_fallback_questions(
            nl_request, schema_context, conflicting_specs
        )

        # ``use_llm=False`` lets non-HITL arms record ambiguity without paying for an LLM
        # question-generation call nobody will answer (the questions are only consumed by
        # a user_input_provider).
        if not self.llm_client or not use_llm:
            return AmbiguityAnalysisResult(
                ambiguity_points=["Multiple interpretations detected from model disagreement."],
                clarifying_questions=fallback_questions,
                explanation="LLM unavailable - using rule-based ambiguity detection."
                if not self.llm_client
                else "LLM question generation skipped (no HITL consumer) - rule-based questions.",
            )

        prompt = (
            "You are analyzing ambiguity in a natural language analytics request.\n\n"
            "Three AI models interpreted this request differently. Your job is to:\n"
            "1. Identify the SPECIFIC ambiguous terms or phrases that caused disagreement\n"
            "2. Generate 2-3 ACTIONABLE clarifying questions that will resolve the ambiguity\n\n"
            "IMPORTANT: Questions must be:\n"
            "- Specific and answerable (not vague like 'what do you mean?')\n"
            "- Include concrete options when possible (e.g., 'Do you want X or Y?')\n"
            "- Reference the actual schema columns/tables available\n\n"
            f"Original request: {nl_request}\n\n"
            f"Available schema:\n"
            f"  Models: {schema_context.get('models', [])}\n"
            f"  Columns: {schema_context.get('columns', {})}\n\n"
            f"Conflicting interpretations:\n"
        )
        
        for provider, spec in conflicting_specs.items():
            prompt += f"  {provider}: {spec}\n"
        
        prompt += (
            f"\nUncertainty score: {epistemic_uncertainty:.2f} (0=agreement, 1=total disagreement)\n\n"
            "Return JSON with:\n"
            "- ambiguity_points: list of specific ambiguous terms/phrases\n"
            "- clarifying_questions: list of 2-3 specific, actionable questions\n"
            "- explanation: brief explanation of the core disagreement"
        )
        
        try:
            result = self.llm_client.complete_json(
                "You are an expert at identifying ambiguity in analytics requests and generating helpful clarifying questions.",
                prompt,
                AmbiguityAnalysisResult,
                temperature=0.3,  # Lower temperature for more consistent questions
            )
            
            # Validate questions aren't too vague
            if result.clarifying_questions:
                vague_phrases = ["clarify", "what do you mean", "can you explain", "more details"]
                good_questions = [
                    q for q in result.clarifying_questions
                    if not any(vp in q.lower() for vp in vague_phrases)
                ]
                if len(good_questions) < len(result.clarifying_questions):
                    # Some questions were too vague, supplement with fallback
                    result.clarifying_questions = good_questions + fallback_questions
                    result.clarifying_questions = result.clarifying_questions[:3]
            
            return result
        except Exception:
            return AmbiguityAnalysisResult(
                ambiguity_points=["Model disagreement detected in interpretation."],
                clarifying_questions=fallback_questions,
                explanation="Ambiguity analysis encountered an error - using rule-based detection.",
            )
