from __future__ import annotations

import datetime as _dt
import numbers
import re
from dataclasses import dataclass
from typing import Any

_DATE_STR = re.compile(r"^(\d{4}-\d{2}-\d{2})(?:[T ]00:00:00(?:\.0+)?)?$")


@dataclass
class ComparisonResult:
    match: bool
    accuracy: float
    diff: dict[str, Any]


def _normalize_cell(value: Any) -> Any:
    """Bridge representation gaps between gold (pandas/duckdb) and actual (MetricFlow text).

    Dates/timestamps are reduced to ISO date strings when the time component is
    midnight, so a pandas ``Timestamp('2018-03-01 00:00:00')`` compares equal to the
    string ``'2018-03-01'`` that MetricFlow emits.
    """
    if value is None:
        return None
    if isinstance(value, _dt.datetime):
        if (value.hour, value.minute, value.second, value.microsecond) == (0, 0, 0, 0):
            return value.date().isoformat()
        return value.isoformat()
    if isinstance(value, _dt.date):
        return value.isoformat()
    if isinstance(value, str):
        match = _DATE_STR.match(value.strip())
        return match.group(1) if match else value.strip()
    return value


def _coerce_numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    if isinstance(value, numbers.Number):  # numpy scalars etc.
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return None


def _value_equal(a: Any, b: Any, rules: dict[str, Any]) -> bool:
    nulls_equal = rules.get("nulls_equal", True)
    a = _normalize_cell(a)
    b = _normalize_cell(b)
    if a is None or b is None:
        return a is None and b is None if nulls_equal else False

    a_num = _coerce_numeric(a)
    b_num = _coerce_numeric(b)
    if a_num is not None and b_num is not None:
        abs_tol = float(rules.get("abs_tol", 0.0))
        rel_tol = float(rules.get("rel_tol", 0.0))
        diff = abs(a_num - b_num)
        if diff <= abs_tol:
            return True
        if rel_tol > 0:
            denom = max(abs(a_num), abs(b_num), 1e-12)
            return diff / denom <= rel_tol
        return False

    return a == b


def _row_equal(row_a: dict[str, Any], row_b: dict[str, Any], columns: list[str], rules: dict[str, Any]) -> bool:
    for col in columns:
        if not _value_equal(row_a.get(col), row_b.get(col), rules):
            return False
    return True


def _normalize_rows(rows: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not rows:
        return []
    return [row for row in rows if isinstance(row, dict)]


def _extract_schema(result_schema: dict[str, Any] | None) -> tuple[list[str] | None, dict[str, str]]:
    if not result_schema:
        return None, {}
    columns = result_schema.get("columns")
    if isinstance(columns, list):
        if columns and isinstance(columns[0], dict):
            names = [item.get("name") for item in columns if isinstance(item, dict) and item.get("name")]
            types = {
                item.get("name"): item.get("type")
                for item in columns
                if isinstance(item, dict) and item.get("name") and item.get("type")
            }
            return names or None, types
        if columns and all(isinstance(item, str) for item in columns):
            return list(columns), {}
    fields = result_schema.get("fields")
    if isinstance(fields, list):
        names = [item.get("name") for item in fields if isinstance(item, dict) and item.get("name")]
        types = {
            item.get("name"): item.get("type")
            for item in fields
            if isinstance(item, dict) and item.get("name") and item.get("type")
        }
        return names or None, types
    if isinstance(result_schema, dict) and all(isinstance(key, str) for key in result_schema.keys()):
        # Allow mapping of column -> type
        types = {
            key: value for key, value in result_schema.items() if isinstance(value, str)
        }
        return list(result_schema.keys()) or None, types
    return None, {}


def _sort_key(value: Any) -> tuple[int, str]:
    norm = _normalize_cell(value)
    if norm is None:
        return (0, "")
    num = _coerce_numeric(norm)
    if num is not None:
        return (1, f"{num:.12g}")
    return (2, str(norm))


# Default numeric tolerance applied when a task supplies no compare_rules (or omits
# both tolerances). Exact float equality between DuckDB gold (float64) and MetricFlow's
# text-parsed output spuriously fails on any AVG/ratio measure. Integer counts have
# diff == 0 and still match exactly. A task may still force exactness with rel_tol=0.0.
_DEFAULT_RULES: dict[str, Any] = {"rel_tol": 1e-6}

# Above this column count, brute-forcing column permutations (n!) is too expensive, so
# the agnostic path assumes positional column order instead.
_MAX_PERMUTE_COLS = 7


def _columns_of(rows: list[dict[str, Any]]) -> list[str]:
    """Stable, order-preserving union of column names across rows."""
    cols: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in cols:
                cols.append(key)
    return cols


def _infer_sort_columns(
    rows: list[dict[str, Any]], columns: list[str], rules: dict[str, Any]
) -> list[str]:
    """Columns whose values are monotonic (non-decreasing or non-increasing) across rows.

    For a "top-N by metric" result these are the de-facto sort key(s). Used only to find
    *tie runs*; a column that is monotonic by coincidence is harmless — it just means we
    canonicalize within rows that also tie on it.
    """
    if len(rows) < 2:
        return []
    sort_cols: list[str] = []
    for col in columns:
        keys = [_sort_key(row.get(col)) for row in rows]
        non_decr = all(keys[i] <= keys[i + 1] for i in range(len(keys) - 1))
        non_incr = all(keys[i] >= keys[i + 1] for i in range(len(keys) - 1))
        has_tie = any(keys[i] == keys[i + 1] for i in range(len(keys) - 1))
        # Require a tie: a strictly-monotonic column (e.g. an id sorted 1,2,3) creates no
        # reordering ambiguity and must not be mistaken for the ranking key — that would
        # block grouping the real metric ties.
        if (non_decr or non_incr) and has_tie:
            sort_cols.append(col)
    return sort_cols


def _canonicalize_tie_runs(
    rows: list[dict[str, Any]], sort_cols: list[str], columns: list[str], rules: dict[str, Any]
) -> list[dict[str, Any]]:
    """Within each maximal run of rows tied on ``sort_cols``, sort by a canonical full-row
    key so ties become order-independent.

    This removes the only ambiguity in an ordered comparison — the arbitrary order of rows
    sharing a sort-key value (e.g. two customers tied on lifetime value) — without touching
    the primary order sequence, so a genuine wrong-direction (asc vs desc) result is still
    caught. Fixes the flaky "top-N with ties" tasks (real_002, real_017).
    """
    if not sort_cols or len(rows) < 2:
        return list(rows)

    def row_key(row: dict[str, Any]) -> tuple:
        return tuple(_sort_key(row.get(c)) for c in columns)

    out: list[dict[str, Any]] = []
    run = [rows[0]]
    for row in rows[1:]:
        if _row_equal(run[0], row, sort_cols, rules):
            run.append(row)
        else:
            out.extend(sorted(run, key=row_key) if len(run) > 1 else run)
            run = [row]
    out.extend(sorted(run, key=row_key) if len(run) > 1 else run)
    return out


def _gold_is_prefix(
    expected: list[dict[str, Any]], actual: list[dict[str, Any]], columns: list[str], rules: dict[str, Any]
) -> bool:
    """True when ``expected`` (gold, already tie-canonicalized, strictly shorter) equals the
    first ``len(expected)`` rows of ``actual`` positionally.

    A tie run in ``actual`` that straddles the cutoff is compared fairly: gold's tail rows
    that sit inside the straddling run may match ANY member of that run (gold's LIMIT picked
    arbitrarily among the tie), mirroring :func:`_canonicalize_tie_runs`. Everything before
    the run must still match positionally — no set-based acceptance elsewhere.
    """
    n = len(expected)
    # Sort key(s) inferred from the LONGER table: a tie straddling the cutoff lives there
    # and may be invisible in gold (gold can hold a single member of the tie).
    sort_cols = _infer_sort_columns(actual, columns, rules)
    run_start = run_end = n - 1
    if sort_cols:
        while run_start > 0 and _row_equal(actual[run_start - 1], actual[n - 1], sort_cols, rules):
            run_start -= 1
        while run_end + 1 < len(actual) and _row_equal(actual[run_end + 1], actual[n - 1], sort_cols, rules):
            run_end += 1
    if run_end == n - 1:
        # No tie straddles the cutoff: plain positional prefix.
        return all(_row_equal(expected[i], actual[i], columns, rules) for i in range(n))
    if not all(_row_equal(expected[i], actual[i], columns, rules) for i in range(run_start)):
        return False
    pool = list(actual[run_start: run_end + 1])
    for row in expected[run_start:]:
        for i, cand in enumerate(pool):
            if _row_equal(row, cand, columns, rules):
                del pool[i]
                break
        else:
            return False
    return True


def _match_rows(
    expected: list[dict[str, Any]],
    actual: list[dict[str, Any]],
    columns: list[str],
    ordered: bool,
    rules: dict[str, Any],
) -> tuple[int, int, int, int, bool]:
    """Return (matched, missing, extra, mismatched, prefix) comparing rows over ``columns``.

    ``prefix`` is True only for the ordered gold-as-prefix acceptance (see below); the
    extra rows it reports are the benign continuation beyond gold's cutoff.
    """
    matched = missing = extra = mismatched = 0
    if ordered:
        # Make tied rows order-independent (stable tiebreak) so identical result sets do not
        # flip pass/fail across runs purely on the arbitrary order of equal-sort-key rows.
        sort_cols = _infer_sort_columns(expected, columns, rules)
        expected = _canonicalize_tie_runs(expected, sort_cols, columns, rules)
        actual = _canonicalize_tie_runs(actual, sort_cols, columns, rules)
        for idx in range(max(len(expected), len(actual))):
            if idx >= len(expected):
                extra += 1
            elif idx >= len(actual):
                missing += 1
            elif _row_equal(expected[idx], actual[idx], columns, rules):
                matched += 1
            else:
                mismatched += 1
        # Gold-as-ordered-prefix acceptance: when the task fixes no N ("top customers"),
        # gold's LIMIT is arbitrary. If gold is strictly shorter and equals the produced
        # head EXACTLY (positionally, modulo a tie straddling the cutoff), the produced
        # result subsumes gold and counts as a match. Never applied in the other direction
        # (produced shorter than gold stays a failure) nor to unordered comparisons.
        if expected and len(expected) < len(actual) and _gold_is_prefix(expected, actual, columns, rules):
            return len(expected), 0, len(actual) - len(expected), 0, True
    else:
        remaining = actual[:]
        for row in expected:
            for i, cand in enumerate(remaining):
                if _row_equal(row, cand, columns, rules):
                    matched += 1
                    del remaining[i]
                    break
            else:
                missing += 1
        extra = len(remaining)
    return matched, missing, extra, mismatched, False


def _result(
    matched: int, missing: int, extra: int, mismatched: int,
    total: int, column_match: bool, type_mismatch: list, mode: str,
    prefix: bool = False,
    **extra_diff: Any,
) -> ComparisonResult:
    # Under prefix acceptance the extra rows are the benign continuation beyond gold's
    # arbitrary cutoff: exclude them from the denominator and from the match veto.
    denom = (total - extra) if prefix else total
    accuracy = matched / denom if denom else 1.0
    match = (
        accuracy == 1.0 and missing == 0 and (extra == 0 or prefix) and mismatched == 0
        and column_match and not type_mismatch
    )
    diff = {
        "missing_rows": missing, "extra_rows": extra, "mismatched_rows": mismatched,
        "column_match": column_match, "type_mismatch": type_mismatch, "mode": mode,
        **extra_diff,
    }
    if prefix:
        diff["prefix_match"] = True  # keep scoring auditable: matched as gold-prefix
    return ComparisonResult(match=match, accuracy=accuracy, diff=diff)


def _compare_agnostic(
    expected: list[dict[str, Any]], actual: list[dict[str, Any]], ordered: bool, rules: dict[str, Any]
) -> ComparisonResult:
    """Compare ignoring column NAMES but keeping column STRUCTURE consistent.

    Required because MetricFlow names output columns by its own convention
    (e.g. ``metric_time__day``), which never matches hand-authored gold aliases. We find
    a single column bijection (a permutation of actual's columns onto gold's) under which
    the whole table matches — *not* a per-row value bag, which would let a dimension value
    and a measure value swap places and still "match" (a false positive).
    """
    if not expected and not actual:
        # [] vs [] is reported but flagged so callers can refuse to credit it.
        return _result(0, 0, 0, 0, 0, True, [], "column_agnostic", both_empty=True)

    exp_cols = _columns_of(expected)
    act_cols = _columns_of(actual)
    if len(act_cols) < len(exp_cols):
        # Generated has FEWER columns than gold — gold's structure can't be covered, so
        # this is a genuine structural shortfall (not extra descriptive context).
        m, miss, ex, mm, _ = _match_rows(expected, actual, exp_cols, ordered, rules)
        return _result(m, miss, ex, mm, max(len(expected), len(actual)),
                       column_match=False, type_mismatch=[], mode="column_count_mismatch")

    # When the generated table has EXTRA columns (e.g. MetricFlow adds first_name/last_name
    # alongside the requested customer_id + measure), project gold's columns onto a SUBSET
    # of actual's: try every length-|gold| ordered selection of actual columns. A match
    # still requires every gold column covered AND exact row cardinality (missing==extra==
    # mismatched==0), so extra descriptive columns are tolerated but wrong values, wrong
    # row sets, or wrong cardinality are not. Equal column counts reduce to the original
    # full-permutation bijection.
    projecting = len(act_cols) > len(exp_cols)
    base_mode = "column_projected" if projecting else "column_agnostic"
    if len(act_cols) > _MAX_PERMUTE_COLS:
        perms: Any = [tuple(act_cols[: len(exp_cols)])]  # too wide; assume positional order
    else:
        from itertools import permutations
        perms = permutations(act_cols, len(exp_cols))

    total = max(len(expected), len(actual))
    best: ComparisonResult | None = None
    for perm in perms:
        mapping = {act_col: exp_col for act_col, exp_col in zip(perm, exp_cols)}
        remapped = [{mapping[c]: row.get(c) for c in perm} for row in actual]
        m, miss, ex, mm, prefix = _match_rows(expected, remapped, exp_cols, ordered, rules)
        result = _result(m, miss, ex, mm, total, True, [], base_mode, prefix=prefix)
        if result.match:
            return result
        if best is None or result.accuracy > best.accuracy:
            best = result
    assert best is not None
    return best


def compare_results(
    expected_rows: list[dict[str, Any]] | None,
    actual_rows: list[dict[str, Any]] | None,
    result_schema: dict[str, Any] | None = None,
    compare_rules: dict[str, Any] | None = None,
) -> ComparisonResult:
    rules = {**_DEFAULT_RULES, **(compare_rules or {})}
    ordering = rules.get("ordering", "set")
    ordered = ordering in {"ordered", "order", "sequence"}

    expected = _normalize_rows(expected_rows)
    actual = _normalize_rows(actual_rows)

    if rules.get("column_agnostic"):
        return _compare_agnostic(expected, actual, ordered, rules)

    schema_columns, schema_types = _extract_schema(result_schema)
    if schema_columns:
        columns = schema_columns
    else:
        keys = set()
        for row in expected + actual:
            keys.update(row.keys())
        columns = sorted(keys)

    expected_columns = set(columns)
    actual_columns = set()
    for row in actual:
        actual_columns.update(row.keys())
    column_match = expected_columns.issubset(actual_columns) if expected_columns else True

    type_mismatch = []
    if schema_types:
        for col, expected_type in schema_types.items():
            if not actual:
                continue
            sample = actual[0].get(col)
            if sample is None:
                continue
            actual_type = type(sample).__name__
            if expected_type and actual_type.lower() not in expected_type.lower():
                type_mismatch.append({"column": col, "expected": expected_type, "actual": actual_type})

    matched, missing_rows, extra_rows, mismatched_rows, prefix = _match_rows(
        expected, actual, columns, ordered, rules
    )
    total_rows = max(len(expected), len(actual))
    return _result(matched, missing_rows, extra_rows, mismatched_rows, total_rows,
                   column_match, type_mismatch, mode="named", prefix=prefix)
