"""Verification-grounded confidence for NL->semantic-layer specs.

The single-provider diagnostic showed every
*introspective* uncertainty signal — within-model self-consistency (aleatoric),
self-reported ``uncertainty_score``/``confidence``, and the count of the model's own
``ambiguity_points``/``clarifying_questions`` — sits at AUROC ~= 0.50 on this benchmark.
The model is confidently and uniformly wrong: it cannot tell when it has mis-mapped a
request. Resampling or asking the same model again cannot surface a *systematic* error.

This module instead derives confidence by *checking the generated spec against the
schema and the request* — external grounding the model's introspection lacks. Each check
targets a failure mode observed in the run:

* ``agg_verb``    -- the request asks for a non-sum aggregation (average / percentage /
                     ratio / median / distinct) but the spec carries no operator able to
                     express it, so codegen silently defaults to SUM (e.g. real_010:
                     "average order value" -> SUM(amount) = 496 vs gold 17).
* ``unknown_col`` -- a base_measure / group_by column is absent from the schema.
* ``missing_grain`` -- the request names a time grain (per month / daily / ...) but the
                     spec leaves ``time_granularity`` empty.
* ``stale_relative_time`` -- the request uses relative time ("this month", "last month",
                     "recent") that, on a static/historical warehouse, filters to an empty
                     result (e.g. real_016). This is the classic trap for a user who does
                     not know the data's date range.

The score is the count of fired checks; ``verify_spec`` also returns the human-readable
violations, which double as schema-grounded clarification prompts for HITL.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Request verbs that imply an aggregation the current spec schema cannot encode (it has
# no explicit operator field, so codegen defaults to SUM). Ordered longest-first so
# "number of" is matched before "of".
_NON_SUM_AGG = [
    ("average", "avg"), ("avg", "avg"), ("mean", "avg"),
    ("percentage", "ratio"), ("percent", "ratio"), ("proportion", "ratio"),
    ("ratio", "ratio"), ("rate of", "ratio"), ("share of", "ratio"), ("%", "ratio"),
    ("median", "median"), ("distinct", "distinct"), ("unique", "distinct"),
]
_TIME_GRAIN_WORDS = (
    "per day", "daily", "per week", "weekly", "per month", "monthly", "per quarter",
    "quarterly", "per year", "yearly", "annually", "by day", "by week", "by month",
    "by year", "each month", "each day", "over time", "by date",
)
_RELATIVE_TIME = (
    "this month", "last month", "this week", "last week", "this year", "last year",
    "today", "yesterday", "recent", "recently", "currently", "right now", "so far",
    "year to date", "ytd", "this quarter", "last quarter",
)


@dataclass
class VerificationResult:
    score: int                       # number of checks that fired (0 = clean)
    violations: list[str] = field(default_factory=list)
    fired: dict[str, bool] = field(default_factory=dict)


def _schema_columns(schema_context: dict[str, Any] | None) -> set[str]:
    cols: set[str] = set()
    if not schema_context:
        return cols
    by_model = schema_context.get("columns") or {}
    if isinstance(by_model, dict):
        for cs in by_model.values():
            if isinstance(cs, (list, tuple)):
                cols.update(str(c) for c in cs)
    return cols


_OP_PATTERNS = {
    "avg": ("avg(", "average(", "mean("),
    "ratio": ("/", "ratio(", "numerator", "denominator"),
    "median": ("median(", "percentile"),
    "distinct": ("distinct", "count_distinct", "unique("),
}


def _measure_has_operator(spec: dict[str, Any], agg: str) -> bool:
    """True if a base_measure literally carries the requested operator, e.g. 'AVG(amount)'.

    The current mapper emits bare columns ('amount'), so this is almost always False — but
    it future-proofs the check against a schema that does encode operators inline.
    """
    pats = _OP_PATTERNS.get(agg, ())
    blob = " ".join(str(m).lower() for m in (spec.get("base_measures") or []))
    return any(p in blob for p in pats)


def _bare_column(expr: str) -> str:
    """Strip an aggregation wrapper / alias so 'SUM(amount)' -> 'amount'."""
    m = re.search(r"\(([^)]*)\)", expr)
    inner = m.group(1) if m else expr
    return inner.strip().split(".")[-1].strip().strip("'\"")


def verify_spec(
    nl_request: str,
    spec: dict[str, Any] | None,
    schema_context: dict[str, Any] | None,
) -> VerificationResult:
    """Run deterministic schema-grounded checks on a generated spec."""
    spec = spec or {}
    nl = (nl_request or "").lower()
    cols = _schema_columns(schema_context)

    violations: list[str] = []
    fired: dict[str, bool] = {}

    # 1. Aggregation-verb expressibility ------------------------------------------------
    requested_agg = None
    for token, agg in _NON_SUM_AGG:
        if token in nl:
            requested_agg = agg
            break
    agg_fired = False
    if requested_agg:
        # Post-Path-A the designer derives avg/count/distinct/min/max/median from the NL
        # verb, so those are no longer a defect (real_010 'average' now executes correctly).
        # Only 'ratio'/'percentage' remains genuinely unexpressible (a ratio metric needs two
        # measures, which neither the spec schema nor the designer produces) — so restrict the
        # check to that class to avoid false positives on now-handled aggregations.
        encoded = bool(spec.get("aggregation")) or _measure_has_operator(spec, requested_agg)
        if requested_agg == "ratio" and not encoded:
            agg_fired = True
            violations.append(
                "request implies a ratio/percentage (numerator/denominator) but the spec "
                "encodes a single measure — a ratio metric is needed"
            )
    fired["agg_verb"] = agg_fired

    # 2. Unknown columns ----------------------------------------------------------------
    unknown_fired = False
    if cols:
        for bm in spec.get("base_measures") or []:
            if _bare_column(str(bm)) not in cols:
                unknown_fired = True
                violations.append(f"base_measure '{bm}' is not a known schema column")
        for gb in spec.get("group_by") or []:
            if _bare_column(str(gb)) not in cols:
                unknown_fired = True
                violations.append(f"group_by '{gb}' is not a known schema column")
    fired["unknown_col"] = unknown_fired

    # 3. Missing time grain -------------------------------------------------------------
    grain_fired = (
        any(w in nl for w in _TIME_GRAIN_WORDS) and not spec.get("time_granularity")
    )
    if grain_fired:
        violations.append("request names a time grain but time_granularity is empty")
    fired["missing_grain"] = grain_fired

    # 4. Stale relative-time filter -----------------------------------------------------
    stale_fired = any(w in nl for w in _RELATIVE_TIME)
    if stale_fired:
        violations.append(
            "request uses relative time on a static warehouse; the filter may return no "
            "rows (confirm the data's date range with the user)"
        )
    fired["stale_relative_time"] = stale_fired

    return VerificationResult(score=sum(fired.values()), violations=violations, fired=fired)


def violations_to_questions(violations: list[str]) -> list[str]:
    """Turn verifier violation strings into answerable clarifying QUESTIONS.

    The violations are diagnostics ("base_measure 'x' is not a known schema column");
    surfacing them verbatim as HITL prompts makes the answer depend on whether the
    diagnostic text happens to contain a slot keyword. These templates phrase each known
    violation as the question a schema-blind user can actually answer, worded to target
    one spec slot (measure / group-by / granularity / date-range)."""
    questions: list[str] = []
    for v in violations:
        if "ratio metric is needed" in v or "ratio/percentage" in v:
            questions.append(
                "This looks like a ratio or percentage. Which measure is the numerator "
                "and which is the denominator?"
            )
        elif "base_measure" in v and "not a known schema column" in v:
            m = re.search(r"base_measure '([^']+)'", v)
            col = m.group(1) if m else "that column"
            questions.append(
                f"The column '{col}' isn't in the schema. Which column should the "
                "measure be calculated from?"
            )
        elif "group_by" in v and "not a known schema column" in v:
            m = re.search(r"group_by '([^']+)'", v)
            col = m.group(1) if m else "that column"
            questions.append(
                f"'{col}' isn't a known column. Which dimension should the results be "
                "grouped by?"
            )
        elif "time grain" in v and "time_granularity is empty" in v:
            questions.append(
                "What time granularity should the results use (daily, weekly, or monthly)?"
            )
        elif "relative time on a static warehouse" in v:
            questions.append(
                "The data is historical and static. What absolute date range should the "
                "results cover?"
            )
        else:
            questions.append(f"Please confirm: {v}")
    # Dedup while preserving order: two unknown-column violations on the same slot
    # otherwise produce identical questions.
    seen: set[str] = set()
    return [q for q in questions if not (q in seen or seen.add(q))]
