"""Tests for the slot-scoped HITL refinement guard (guard_agreed_slots).

The guard reverts a HITL refinement on a slot back to the consensus value IFF the change
is a "protected overwrite": the consensus value is non-empty AND all ensemble providers
agreed on it. It keeps the refinement when the consensus slot was empty (gain via fill) or
when the providers disagreed, and is a no-op for < 2 providers (single-provider baseline).
"""

from semanticflow.agents.semantic_mapper import guard_agreed_slots


def _three(measures, group_by=None, filters=None, aggregation=None, gran=None):
    """Build three identical per-provider specs from the given slot values."""
    base = {
        "base_measures": measures,
        "group_by": group_by or [],
        "filters": filters or [],
        "aggregation": aggregation,
        "time_granularity": gran,
    }
    return {"openai": dict(base), "anthropic": dict(base), "gemini": dict(base)}


def test_reverts_non_empty_agreed_overwrite():
    """Harm case: all 3 providers agree base_measures=['amount'], refinement flips it."""
    consensus = {"base_measures": ["amount"]}
    refined = {"base_measures": ["order_count"]}
    model_outputs = _three(["amount"])

    guarded, reverted = guard_agreed_slots(consensus, refined, model_outputs)

    assert "base_measures" in reverted
    assert guarded["base_measures"] == ["amount"]


def test_keeps_empty_slot_fill():
    """Gain case: consensus group_by empty -> filling it is allowed, not reverted."""
    consensus = {"group_by": []}
    refined = {"group_by": ["order_date"]}
    model_outputs = _three(["amount"], group_by=[])

    guarded, reverted = guard_agreed_slots(consensus, refined, model_outputs)

    assert reverted == []
    assert guarded["group_by"] == ["order_date"]


def test_keeps_when_providers_disagree():
    """Gain case: providers disagree on group_by -> refinement is allowed to resolve it."""
    consensus = {"group_by": ["customer_id"]}
    refined = {"group_by": ["order_date"]}
    model_outputs = {
        "openai": {"group_by": ["customer_id"]},
        "anthropic": {"group_by": ["order_date"]},
        "gemini": {"group_by": ["status"]},
    }

    guarded, reverted = guard_agreed_slots(consensus, refined, model_outputs)

    assert reverted == []
    assert guarded["group_by"] == ["order_date"]


def test_noop_when_model_outputs_none():
    consensus = {"base_measures": ["amount"]}
    refined = {"base_measures": ["order_count"]}

    guarded, reverted = guard_agreed_slots(consensus, refined, None)

    assert reverted == []
    assert guarded == refined


def test_noop_when_single_provider():
    consensus = {"base_measures": ["amount"]}
    refined = {"base_measures": ["order_count"]}
    model_outputs = {"anthropic": {"base_measures": ["amount"]}}

    guarded, reverted = guard_agreed_slots(consensus, refined, model_outputs)

    assert reverted == []
    assert guarded == refined


def test_mixed_protect_one_keep_other():
    """One protected-overwrite slot reverted, one empty-fill slot kept, same call."""
    consensus = {
        "base_measures": ["amount"],  # non-empty + agreed -> protected
        "group_by": [],               # empty -> fill allowed
    }
    refined = {
        "base_measures": ["order_count"],  # overwrite -> should revert
        "group_by": ["order_date"],        # fill -> should keep
    }
    model_outputs = _three(["amount"], group_by=[])

    guarded, reverted = guard_agreed_slots(consensus, refined, model_outputs)

    assert reverted == ["base_measures"]
    assert guarded["base_measures"] == ["amount"]
    assert guarded["group_by"] == ["order_date"]
