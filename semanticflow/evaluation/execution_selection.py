"""Execution-grounded answer selection.

We run several providers, execute each provider's spec, and today take the final
answer from a *text* consensus vote over the specs. That vote can land on a spec that
errors or returns an empty table even when another provider's spec executed cleanly to
a correct result — text agreement is not execution truth.

This module picks the final answer from the providers' EXECUTED results instead:

  1. Partition candidates into VALID (query succeeded AND returned a non-empty table)
     and INVALID (errored or empty).
  2. Cluster the VALID candidates by execution-equivalence — the same relation used in
     :mod:`execution_consistency`, i.e. :func:`compare_results` (column-name/order-
     agnostic, date- and tolerance-aware), so MetricFlow's column-naming quirks never
     split one real answer into two clusters.
  3. Choose a representative from the LARGEST cluster (the result the most providers
     independently agree on). Ties break deterministically: larger cluster, then higher
     spec confidence, then provider input order.
  4. If NO valid candidate exists, return a selection flagged ``no_valid_execution`` and
     change nothing — the caller falls back to its current text-consensus behaviour.

Pure and deterministic: no LLM, no live dbt/mf calls. ``None``/empty inputs are handled
gracefully (an empty candidate list yields a ``no_valid_execution`` no-op).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from semanticflow.evaluation.result_compare import compare_results


@dataclass
class Candidate:
    """One provider's executed spec.

    Attributes:
        provider: provider name (e.g. ``"anthropic"``); also the deterministic
            tie-break key (input order is preserved).
        spec: the provider's semantic spec dict. May carry a ``confidence`` float used
            as a secondary tie-break.
        result_rows: the executed result table (list of row dicts), or ``None`` if the
            query did not run / produced nothing.
        mf_query_success: whether the MetricFlow query succeeded.
    """

    provider: str
    spec: dict[str, Any] = field(default_factory=dict)
    result_rows: list[dict[str, Any]] | None = None
    mf_query_success: bool = False

    def is_valid(self) -> bool:
        """Valid == query succeeded AND it returned a non-empty result table.

        An errored or empty result is the strongest "don't trust me" signal, so it is
        never selectable while any non-empty success exists.
        """
        return bool(self.mf_query_success) and bool(self.result_rows)


@dataclass
class ExecutionSelection:
    """Result of :func:`select_by_execution`.

    When ``valid`` is ``False`` the selection is a safe no-op: every field except
    ``reason`` is empty/zero and the caller should keep its existing consensus answer.
    """

    valid: bool                      # True iff a valid executed candidate was chosen
    reason: str                      # human-readable explanation of the decision
    provider: str | None = None      # chosen provider (None if no_valid_execution)
    spec: dict[str, Any] | None = None
    result_rows: list[dict[str, Any]] | None = None
    cluster_size: int = 0            # size of the winning execution-equivalence cluster
    n_valid: int = 0                 # number of valid (success + non-empty) candidates
    agreement: float = 0.0           # cluster_size / n_valid  (∈ [0, 1])


def _spec_confidence(spec: dict[str, Any] | None) -> float:
    """Spec confidence for tie-breaking; missing/garbage -> 0.0 (lowest priority)."""
    if not isinstance(spec, dict):
        return 0.0
    value = spec.get("confidence")
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def select_by_execution(
    candidates: list[Candidate] | None,
    compare_rules: dict[str, Any] | None = None,
) -> ExecutionSelection:
    """Pick the final answer from providers' EXECUTED results.

    Args:
        candidates: one :class:`Candidate` per provider. ``None``/empty is handled.
        compare_rules: forwarded to :func:`compare_results`. ``column_agnostic`` is
            forced on (providers' specs alias columns differently), matching
            :mod:`execution_consistency`.

    Returns:
        :class:`ExecutionSelection`. ``valid=False`` (``reason="no_valid_execution"``)
        when no candidate succeeded with rows — a no-op so the caller keeps its current
        text-consensus result.
    """
    if not candidates:
        return ExecutionSelection(valid=False, reason="no_valid_execution")

    rules = {"column_agnostic": True, **(compare_rules or {})}

    # Preserve input order so tie-breaks are deterministic and stable.
    valid = [c for c in candidates if c.is_valid()]
    n_valid = len(valid)
    if n_valid == 0:
        return ExecutionSelection(valid=False, reason="no_valid_execution", n_valid=0)

    # Greedy single-linkage clustering by pairwise execution-equivalence, mirroring
    # execution_consistency.execution_consistency. n is tiny (<= #providers), so an
    # O(n^2) sweep is cheap and avoids hashing a tolerance-/order-agnostic table.
    clusters: list[list[Candidate]] = []  # each cluster: members in input order
    for cand in valid:
        placed = False
        for cluster in clusters:
            if compare_results(
                cluster[0].result_rows, cand.result_rows, compare_rules=rules
            ).match:
                cluster.append(cand)
                placed = True
                break
        if not placed:
            clusters.append([cand])

    # Each cluster's representative is its highest-confidence member (earliest input
    # order breaks confidence ties). The winning cluster is the largest; equal-size
    # clusters break by their representative's confidence, then input order — so a
    # standalone high-confidence provider can outrank an equally-small rival but never a
    # genuinely larger agreeing cluster.
    def rep_of(cluster: list[Candidate]) -> Candidate:
        return max(cluster, key=lambda c: (_spec_confidence(c.spec), -valid.index(c)))

    largest = max(
        clusters,
        key=lambda c: (
            len(c),
            _spec_confidence(rep_of(c).spec),
            -valid.index(rep_of(c)),
        ),
    )
    cluster_size = len(largest)
    chosen = rep_of(largest)

    n_clusters = len(clusters)
    if n_clusters == 1:
        reason = f"unanimous_execution: all {cluster_size} valid providers agree"
    else:
        reason = (
            f"majority_execution: {cluster_size}/{n_valid} providers agree "
            f"(provider={chosen.provider})"
        )

    return ExecutionSelection(
        valid=True,
        reason=reason,
        provider=chosen.provider,
        spec=chosen.spec,
        result_rows=chosen.result_rows,
        cluster_size=cluster_size,
        n_valid=n_valid,
        agreement=cluster_size / n_valid,
    )
