"""Deterministic verifier-guided repair of semantic specs.

``spec_verification.verify_spec`` flags structural defects in a generated spec and
emits human-readable ``violations`` that double as clarifying questions. Today those
violations are only *surfaced* (to HITL or as questions); the spec itself is never
mended. This module closes that loop: given a spec, the verifier's violations, and the
schema, it DETERMINISTICALLY auto-applies the fixes that do not need a human, so the
final spec is repaired before the last execution (including after the HITL step).

Design constraints
------------------
* No LLM, no live calls — every fix is a pure function of (spec, violations, schema).
* Dispatch is keyed off the verifier's own violation *strings* (its stable contract;
  see ``spec_verification.verify_spec``), so this module never re-implements detection —
  it only acts on a defect the verifier already confirmed. The NL request is not passed
  to ``verify_spec``'s caller-visible result, so signals that need the request (the
  implied grain word, the relative phrase) are recovered from the spec text itself
  (``metric_name`` / ``filters`` / ``definition_notes``), which carry them in practice.
* Conservative: if no *safe* deterministic fix exists for a violation, it is LEFT (not
  guessed) and not reported in ``applied_fixes``.

Auto-fixed violation classes
----------------------------
* ``missing_grain``       — a named grain is recoverable from the spec text but
                            ``time_granularity`` is empty -> set it (day/week/month/...).
* ``stale_relative_time`` — a relative phrase ("this month", "last 7 days") that returns
                            empty on the static 2018 warehouse -> replace with an absolute
                            ``order_date`` range inside the data span.
* ``agg_verb``            — the verifier names the implied aggregation and the spec lacks
                            it -> set ``spec['aggregation']``. (In the current verifier the
                            only ``agg_verb`` class is ``ratio``, which is NOT safely
                            auto-fixable — a ratio needs a second measure — so it is left.)
* ``unknown_col``         — a base_measure / group_by absent from the schema that has an
                            unambiguous close match among the schema columns -> correct it.

Left for a human (no safe deterministic fix)
--------------------------------------------
* ``agg_verb`` == ratio   — needs a numerator/denominator the spec cannot express.
* ``unknown_col`` with no (or ambiguous) close match — would be a guess.
* any violation string this module does not recognise.

Warehouse date span
-------------------
The synthetic jaffle_shop warehouse holds order_date in [2018-01-01, 2018-04-09]
(verified via ``SELECT MIN/MAX(order_date) FROM main.orders``; the gold-authoring scripts
use the same 2018 windows, e.g. ``order_date BETWEEN '2018-03-01' AND '2018-04-30'``).
Relative-time phrases are rewritten to an absolute slice of this span using the same
``order_date >= '...'`` / ``order_date <= '...'`` filter-string convention the codebase
already uses (see scripts/author_gold.py, semanticflow/evaluation/sim_human.py).
"""
from __future__ import annotations

import copy
import datetime as _dt
import difflib
import re
from typing import Any

# --- warehouse data span (see module docstring) ----------------------------------------
WAREHOUSE_START = "2018-01-01"
WAREHOUSE_END = "2018-04-09"
_DATE_COL = "order_date"

# Named grain words -> the canonical ``time_granularity`` value the designer accepts
# (day / week / month / quarter / year). Ordered so multi-word phrases match first.
_GRAIN_WORDS: tuple[tuple[str, str], ...] = (
    ("per quarter", "quarter"), ("quarterly", "quarter"), ("by quarter", "quarter"),
    ("per month", "month"), ("monthly", "month"), ("by month", "month"),
    ("each month", "month"), ("per week", "week"), ("weekly", "week"),
    ("by week", "week"), ("per year", "year"), ("yearly", "year"),
    ("annually", "year"), ("by year", "year"), ("per day", "day"),
    ("daily", "day"), ("by day", "day"), ("each day", "day"), ("by date", "day"),
)

# Relative-time phrase -> absolute [start, end] window inside the warehouse span.
# The static warehouse's last month of data is April 2018, so "this/current" anchors
# to April and "last" to the preceding month (March). Ranges that would otherwise be
# open-ended ("recent", "so far") fall back to the full span so no rows are dropped.
_REL_TIME_WINDOWS: tuple[tuple[str, str, str], ...] = (
    ("year to date", "2018-01-01", WAREHOUSE_END),
    ("ytd", "2018-01-01", WAREHOUSE_END),
    ("this quarter", "2018-04-01", WAREHOUSE_END),
    ("last quarter", "2018-01-01", "2018-03-31"),
    ("this month", "2018-04-01", WAREHOUSE_END),
    ("last month", "2018-03-01", "2018-03-31"),
    ("this week", "2018-04-01", WAREHOUSE_END),
    ("last week", "2018-04-01", WAREHOUSE_END),
    ("this year", WAREHOUSE_START, WAREHOUSE_END),
    ("last year", WAREHOUSE_START, WAREHOUSE_END),
    ("today", WAREHOUSE_END, WAREHOUSE_END),
    ("yesterday", "2018-04-08", "2018-04-08"),
    ("year", WAREHOUSE_START, WAREHOUSE_END),  # bare fallback (after compound phrases)
)

# "last N days/weeks/months" -> window ending at WAREHOUSE_END.
_LAST_N_RE = re.compile(r"last\s+(\d+)\s+(day|days|week|weeks|month|months)")

# Verifier violation-string fingerprints (stable substrings of verify_spec's output).
_V_MISSING_GRAIN = "time_granularity is empty"
_V_STALE_TIME = "relative time on a static warehouse"
_V_RATIO = "ratio/percentage"
_V_UNKNOWN_PREFIX_MEASURE = "base_measure '"
_V_UNKNOWN_PREFIX_GROUP = "group_by '"


def _schema_columns(schema_context: dict[str, Any] | None) -> set[str]:
    """Mirror spec_verification._schema_columns so we judge the same column universe."""
    cols: set[str] = set()
    if not schema_context:
        return cols
    by_model = schema_context.get("columns") or {}
    if isinstance(by_model, dict):
        for cs in by_model.values():
            if isinstance(cs, (list, tuple)):
                cols.update(str(c) for c in cs)
    return cols


def _bare_column(expr: str) -> str:
    """Strip an aggregation wrapper / alias so 'SUM(amount)' -> 'amount'.

    Same normalisation spec_verification uses when it decides a column is unknown.
    """
    m = re.search(r"\(([^)]*)\)", expr)
    inner = m.group(1) if m else expr
    return inner.strip().split(".")[-1].strip().strip("'\"")


def _spec_text(spec: dict[str, Any]) -> str:
    """Concatenate the spec's free-text fields where an NL grain/relative phrase survives.

    The verifier ran against the NL request, which the repairer does not receive; the
    metric_name ('average_order_value_per_month'), definition_notes, and any echoed
    filters are where the originating phrase is recoverable deterministically.
    """
    parts: list[str] = [str(spec.get("metric_name") or ""),
                        str(spec.get("definition_notes") or "")]
    for f in spec.get("filters") or []:
        parts.append(str(f))
    for q in spec.get("clarifying_questions") or []:
        parts.append(str(q))
    return " ".join(parts).lower().replace("_", " ")


def _recover_grain(spec: dict[str, Any]) -> str | None:
    text = _spec_text(spec)
    for word, grain in _GRAIN_WORDS:
        if word in text:
            return grain
    return None


def _recover_relative_window(spec: dict[str, Any]) -> tuple[str, str] | None:
    text = _spec_text(spec)
    m = _LAST_N_RE.search(text)
    if m:
        n = int(m.group(1))
        unit = m.group(2).rstrip("s")
        end = _dt.date.fromisoformat(WAREHOUSE_END)
        days = {"day": n, "week": n * 7, "month": n * 30}[unit]
        start = end - _dt.timedelta(days=days)
        floor = _dt.date.fromisoformat(WAREHOUSE_START)
        if start < floor:
            start = floor
        return start.isoformat(), end.isoformat()
    for phrase, start, end in _REL_TIME_WINDOWS:
        if phrase in text:
            return start, end
    return None


def _existing_date_filter(spec: dict[str, Any]) -> bool:
    """True if the spec already carries an absolute order_date bound (don't double-add)."""
    for f in spec.get("filters") or []:
        fl = str(f).lower()
        if _DATE_COL in fl and re.search(r"\d{4}-\d{2}-\d{2}", fl):
            return True
    return False


def _close_match(unknown: str, cols: set[str]) -> str | None:
    """Return the single unambiguous close column match for *unknown*, else None.

    Safe only when exactly one candidate clears the cutoff (an ambiguous tie is a guess
    and is left for a human).
    """
    bare = _bare_column(unknown)
    if bare in cols:
        return None  # not actually unknown
    matches = difflib.get_close_matches(bare, sorted(cols), n=2, cutoff=0.8)
    if len(matches) == 1:
        return matches[0]
    if len(matches) >= 2 and matches[0] != matches[1]:
        # Accept only if the top match is strictly better than the runner-up.
        r0 = difflib.SequenceMatcher(None, bare, matches[0]).ratio()
        r1 = difflib.SequenceMatcher(None, bare, matches[1]).ratio()
        if r0 - r1 >= 0.05:
            return matches[0]
    return None


def _unknown_name(violation: str) -> str | None:
    """Extract the offending column from an unknown_col violation string."""
    m = re.search(r"'([^']+)'", violation)
    return m.group(1) if m else None


def auto_repair_from_verification(
    spec: dict[str, Any],
    violations: list[str],
    schema_context: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    """Deterministically apply the human-free fixes implied by verifier violations.

    Parameters
    ----------
    spec:
        The generated semantic spec (metric_name, base_measures, group_by, aggregation,
        filters, time_granularity, ...) — the shape produced by the semantic mapper.
    violations:
        The ``VerificationResult.violations`` list returned by
        ``spec_verification.verify_spec`` for this spec.
    schema_context:
        The task's schema context (``{"columns": {model: [col, ...]}}``).

    Returns
    -------
    (repaired_spec, applied_fixes):
        ``repaired_spec`` is a deep copy with auto-fixes applied (the input is never
        mutated). ``applied_fixes`` is a list of human-readable strings, one per change.
        No-op (unchanged copy, empty list) when *violations* is empty or none are
        auto-fixable.
    """
    if not violations:
        return copy.deepcopy(spec or {}), []

    repaired = copy.deepcopy(spec or {})
    cols = _schema_columns(schema_context)
    applied: list[str] = []

    for violation in violations:
        v = str(violation)

        # --- missing_grain: set time_granularity from a grain word in the spec text ----
        if _V_MISSING_GRAIN in v and not repaired.get("time_granularity"):
            grain = _recover_grain(repaired)
            if grain:
                repaired["time_granularity"] = grain
                applied.append(
                    f"set time_granularity to '{grain}' (named grain present but unset)"
                )
            # else: cannot recover the grain deterministically -> leave for a human.
            continue

        # --- stale_relative_time: rewrite to an absolute in-span order_date window ------
        if _V_STALE_TIME in v:
            if _existing_date_filter(repaired):
                continue  # an absolute bound already constrains the query.
            window = _recover_relative_window(repaired)
            if window:
                start, end = window
                filters = list(repaired.get("filters") or [])
                filters.append(f"{_DATE_COL} >= '{start}'")
                filters.append(f"{_DATE_COL} <= '{end}'")
                repaired["filters"] = filters
                applied.append(
                    f"converted relative-time filter to absolute range "
                    f"{_DATE_COL} in ['{start}', '{end}'] (static 2018 warehouse)"
                )
            # else: relative phrase not recoverable from spec text -> leave for a human.
            continue

        # --- agg_verb (ratio): NOT safely auto-fixable -> leave for a human ------------
        if _V_RATIO in v:
            continue

        # --- unknown_col: correct to an unambiguous close schema column ----------------
        if _V_UNKNOWN_PREFIX_MEASURE in v or _V_UNKNOWN_PREFIX_GROUP in v:
            name = _unknown_name(v)
            if not name or not cols:
                continue
            field = "base_measures" if _V_UNKNOWN_PREFIX_MEASURE in v else "group_by"
            match = _close_match(name, cols)
            if not match:
                continue  # no safe single close match -> leave for a human.
            items = list(repaired.get(field) or [])
            changed = False
            for i, item in enumerate(items):
                if str(item) == name:
                    items[i] = match
                    changed = True
            if changed:
                repaired[field] = items
                applied.append(
                    f"corrected {field} '{name}' -> '{match}' "
                    f"(closest schema column)"
                )
            continue

        # Unrecognised violation -> not auto-fixable, leave untouched.

    return repaired, applied
