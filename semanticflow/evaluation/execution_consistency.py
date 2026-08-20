"""Execution-based consistency — the strongest black-box uncertainty signal for
text-to-SQL.

Both the IBM/NeurIPS-2024 consistency-based UQ work and Somov & Tutubalina (2025,
arXiv:2508.14056) find the same thing: sample several queries, EXECUTE them, and
cluster by identical result sets. Confidence is the share of samples landing in the
largest cluster; uncertainty is its complement. This beats embedding- or
token-similarity clustering (and the bag-of-tokens JSD / field-set Jaccard signals
used elsewhere in this repo), because two specs that read differently can be
semantically identical — and only execution reveals that.

We reuse :func:`compare_results` (column-name/order-agnostic, date- and
tolerance-aware) as the equivalence relation, so MetricFlow's column-naming quirks
never split a cluster that is really one answer.

A sample whose query failed to execute carries no result to cluster; it is the
strongest possible "I'm unsure" signal, so each failed sample becomes its own
singleton cluster (maximising disagreement) rather than being silently dropped.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from semanticflow.evaluation.result_compare import compare_results


@dataclass
class ExecutionConsistency:
    n_samples: int
    n_executed: int          # samples that produced a result table (query succeeded)
    n_failed: int            # samples whose query failed (each a singleton cluster)
    n_clusters: int          # distinct execution-equivalence classes (failures included)
    largest_cluster: int     # size of the modal answer
    agreement: float         # largest_cluster / n_samples  (∈ [0, 1])
    execution_uncertainty: float   # 1 - agreement            (∈ [0, 1])
    cluster_sizes: list[int] = field(default_factory=list)
    # Index of one representative sample from the largest cluster — used to pick a
    # spec that actually executes and agrees, instead of field-voting a Frankenstein
    # spec across providers.
    modal_sample_index: int | None = None


def execution_consistency(
    result_tables: list[list[dict[str, Any]] | None],
    successes: list[bool] | None = None,
    compare_rules: dict[str, Any] | None = None,
) -> ExecutionConsistency:
    """Cluster sampled query results by execution-equivalence and score consistency.

    Args:
        result_tables: one result table (list of row dicts) per sample; ``None`` for a
            sample whose query did not run.
        successes: optional per-sample success flags. When omitted, a sample counts as
            executed iff its table is not ``None``. A sample marked unsuccessful is
            treated as a failure (singleton) even if it returned rows.
        compare_rules: forwarded to :func:`compare_results`. ``column_agnostic`` is
            forced on (gold-style aliases never match MetricFlow's column names).

    Returns:
        :class:`ExecutionConsistency`. ``agreement`` over the largest cluster is the
        confidence; ``execution_uncertainty = 1 - agreement`` is the signal to trigger
        HITL on (higher = more disagreement among the samples).
    """
    n = len(result_tables)
    if n == 0:
        return ExecutionConsistency(0, 0, 0, 0, 0, 0.0, 0.0, [], None)

    if successes is None:
        successes = [t is not None for t in result_tables]

    rules = {"column_agnostic": True, **(compare_rules or {})}

    # Greedy single-linkage clustering by pairwise execution-equivalence. n is tiny
    # (k * providers, typically <= 9), so an O(n^2) sweep is comfortably cheap and
    # avoids needing a canonical hash of a tolerance-/order-agnostic table.
    clusters: list[dict[str, Any]] = []  # {"rep": table, "members": [idx, ...]}
    n_failed = 0
    for idx, table in enumerate(result_tables):
        if not successes[idx] or table is None:
            n_failed += 1
            clusters.append({"rep": None, "members": [idx], "failed": True})
            continue
        placed = False
        for cluster in clusters:
            if cluster.get("failed"):
                continue
            if compare_results(cluster["rep"], table, compare_rules=rules).match:
                cluster["members"].append(idx)
                placed = True
                break
        if not placed:
            clusters.append({"rep": table, "members": [idx], "failed": False})

    cluster_sizes = sorted((len(c["members"]) for c in clusters), reverse=True)
    largest = max(clusters, key=lambda c: len(c["members"]))
    largest_size = len(largest["members"])
    # Prefer a representative from a *successful* modal cluster; if the modal cluster is
    # a failure singleton, there is no usable representative.
    modal_index = largest["members"][0] if not largest.get("failed") else None
    agreement = largest_size / n
    return ExecutionConsistency(
        n_samples=n,
        n_executed=n - n_failed,
        n_failed=n_failed,
        n_clusters=len(clusters),
        largest_cluster=largest_size,
        agreement=agreement,
        execution_uncertainty=1.0 - agreement,
        cluster_sizes=cluster_sizes,
        modal_sample_index=modal_index,
    )
