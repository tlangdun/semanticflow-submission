"""Tests for execution-grounded answer selection
(semanticflow/evaluation/execution_selection.py)."""
from __future__ import annotations

from semanticflow.evaluation.execution_selection import (
    Candidate,
    select_by_execution,
)


def _rows(value: int) -> list[dict]:
    """A distinct single-cell result table keyed by ``value``."""
    return [{"metric": value}]


# Two tables that are execution-equivalent despite different column NAMES — exercises the
# column-agnostic equivalence relation reused from compare_results.
_ALIASED_A = [{"order_total": 100}]
_ALIASED_B = [{"orders_amount": 100}]


def test_all_agree_that_cluster_wins() -> None:
    cands = [
        Candidate("anthropic", {"confidence": 0.5}, _rows(7), True),
        Candidate("openai", {"confidence": 0.5}, _rows(7), True),
        Candidate("google", {"confidence": 0.5}, _rows(7), True),
    ]
    sel = select_by_execution(cands)
    assert sel.valid is True
    assert sel.cluster_size == 3
    assert sel.n_valid == 3
    assert sel.agreement == 1.0
    assert sel.result_rows == _rows(7)
    assert "unanimous" in sel.reason


def test_majority_cluster_wins() -> None:
    # 2 providers agree on 7, 1 dissents with 9 -> majority (cluster of 2) wins.
    cands = [
        Candidate("anthropic", {"confidence": 0.5}, _rows(7), True),
        Candidate("openai", {"confidence": 0.5}, _rows(9), True),
        Candidate("google", {"confidence": 0.5}, _rows(7), True),
    ]
    sel = select_by_execution(cands)
    assert sel.valid is True
    assert sel.cluster_size == 2
    assert sel.n_valid == 3
    assert sel.result_rows == _rows(7)
    assert sel.provider in {"anthropic", "google"}
    assert "majority" in sel.reason


def test_column_aliased_results_cluster_together() -> None:
    # Same answer, different column names: must land in one cluster (column-agnostic).
    cands = [
        Candidate("anthropic", {}, _ALIASED_A, True),
        Candidate("openai", {}, _ALIASED_B, True),
    ]
    sel = select_by_execution(cands)
    assert sel.valid is True
    assert sel.cluster_size == 2
    assert sel.agreement == 1.0


def test_consensus_candidate_errors_valid_one_chosen() -> None:
    # The text-consensus-style candidate errored; another executed cleanly -> pick valid.
    cands = [
        Candidate("anthropic", {"confidence": 0.9}, None, False),       # errored
        Candidate("openai", {"confidence": 0.4}, _rows(5), True),       # valid
    ]
    sel = select_by_execution(cands)
    assert sel.valid is True
    assert sel.provider == "openai"
    assert sel.result_rows == _rows(5)
    assert sel.n_valid == 1


def test_empty_result_is_invalid_non_empty_chosen() -> None:
    # A succeeded-but-empty result must never beat a non-empty valid one.
    cands = [
        Candidate("anthropic", {"confidence": 0.9}, [], True),          # empty -> invalid
        Candidate("openai", {"confidence": 0.1}, _rows(3), True),       # valid
    ]
    sel = select_by_execution(cands)
    assert sel.valid is True
    assert sel.provider == "openai"
    assert sel.n_valid == 1


def test_all_invalid_returns_no_valid_execution_flag() -> None:
    cands = [
        Candidate("anthropic", {}, None, False),     # errored
        Candidate("openai", {}, [], True),           # empty
        Candidate("google", {}, None, True),         # success flag but no rows
    ]
    sel = select_by_execution(cands)
    assert sel.valid is False
    assert sel.reason == "no_valid_execution"
    assert sel.provider is None
    assert sel.result_rows is None
    assert sel.cluster_size == 0


def test_empty_and_none_inputs_do_not_crash() -> None:
    for arg in (None, []):
        sel = select_by_execution(arg)
        assert sel.valid is False
        assert sel.reason == "no_valid_execution"


def test_tie_broken_by_confidence() -> None:
    # Two singleton clusters of equal size (1): higher spec confidence wins.
    cands = [
        Candidate("anthropic", {"confidence": 0.2}, _rows(1), True),
        Candidate("openai", {"confidence": 0.8}, _rows(2), True),
    ]
    sel = select_by_execution(cands)
    assert sel.valid is True
    assert sel.provider == "openai"
    assert sel.result_rows == _rows(2)


def test_tie_broken_deterministically_by_input_order() -> None:
    # Equal cluster size AND equal confidence -> earliest input order wins, stably.
    cands = [
        Candidate("anthropic", {"confidence": 0.5}, _rows(1), True),
        Candidate("openai", {"confidence": 0.5}, _rows(2), True),
    ]
    sel = select_by_execution(cands)
    assert sel.provider == "anthropic"
    # Reordering the inputs flips the deterministic winner — confirms it is order-driven.
    sel2 = select_by_execution(list(reversed(cands)))
    assert sel2.provider == "openai"


def test_largest_cluster_beats_higher_confidence_singleton() -> None:
    # A 2-provider agreeing cluster (lower confidence) must beat a high-confidence loner.
    cands = [
        Candidate("anthropic", {"confidence": 0.99}, _rows(9), True),   # singleton
        Candidate("openai", {"confidence": 0.10}, _rows(4), True),      # cluster of 2
        Candidate("google", {"confidence": 0.10}, _rows(4), True),
    ]
    sel = select_by_execution(cands)
    assert sel.cluster_size == 2
    assert sel.result_rows == _rows(4)
    assert sel.provider in {"openai", "google"}
