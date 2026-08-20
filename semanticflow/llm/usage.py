"""Thread-safe accumulation of LLM token usage and estimated cost.

The semantic mapper issues k self-consistency samples x N providers CONCURRENTLY against
shared client instances, so usage must be aggregated under a lock in a shared tracker
rather than stored on a client. The orchestrator owns one tracker, resets it per task,
and snapshots it into the result state for logging.
"""
from __future__ import annotations

import threading

# Approximate USD per 1,000,000 tokens, as (input_rate, output_rate). These DRIFT —
# VERIFY against current provider pricing before quoting costs in the thesis. Keys are
# matched as substrings of the model id, longest first. Unknown models are still counted
# in tokens but priced at 0 (cost is derived; token counts are the hard data).
_PRICING_PER_M: dict[str, tuple[float, float]] = {
    "gpt-5.4-mini": (0.25, 2.00),  # estimate — verify against current OpenAI pricing
    "gpt-5.4": (1.25, 10.00),      # estimate — verify against current OpenAI pricing
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "claude-opus-4": (15.00, 75.00),
    "claude-sonnet-4": (3.00, 15.00),
    "claude-haiku-4": (1.00, 5.00),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-3.1-pro": (2.00, 12.00),  # estimate — verify against current Google pricing
}


def price_per_million(model: str) -> tuple[float, float]:
    for key in sorted(_PRICING_PER_M, key=len, reverse=True):
        if key in model:
            return _PRICING_PER_M[key]
    return (0.0, 0.0)


class UsageTracker:
    """Accumulates per-provider token usage + estimated cost across concurrent calls."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_provider: dict[str, dict[str, float]] = {}

    def add(self, provider: str, model: str, input_tokens: int, output_tokens: int) -> None:
        in_rate, out_rate = price_per_million(model)
        cost = (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000
        with self._lock:
            agg = self._by_provider.setdefault(
                provider,
                {"input_tokens": 0.0, "output_tokens": 0.0, "total_tokens": 0.0,
                 "cost_usd": 0.0, "calls": 0.0},
            )
            agg["input_tokens"] += input_tokens
            agg["output_tokens"] += output_tokens
            agg["total_tokens"] += input_tokens + output_tokens
            agg["cost_usd"] += cost
            agg["calls"] += 1

    def reset(self) -> None:
        with self._lock:
            self._by_provider = {}

    def snapshot(self) -> dict[str, dict[str, float]]:
        with self._lock:
            return {p: dict(v) for p, v in self._by_provider.items()}
