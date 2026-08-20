"""Tests for LLM disk-cache wiring (router._maybe_cache) and its K>1 safety guard."""
from __future__ import annotations

import semanticflow.llm.cache as cache_mod
from semanticflow.config import Settings
from semanticflow.llm.base import BaseLLMClient
from semanticflow.llm.router import _maybe_cache


class _CountingClient(BaseLLMClient):
    """Records how many real completions happened."""

    def __init__(self):
        self._model = "stub"
        self.calls = 0

    def complete(self, system_prompt, user_prompt, temperature=None, json_mode=False):
        self.calls += 1
        return f"resp-{self.calls}"


def _fresh_cache(tmp_path):
    # Reset the module global so each test gets an isolated on-disk cache.
    cache_mod._global_cache = cache_mod.LLMCache(cache_dir=tmp_path, enabled=True)


def test_cache_off_by_default():
    c = _CountingClient()
    wrapped = _maybe_cache(c, "anthropic", Settings())
    assert wrapped is c  # not wrapped


def test_cache_hits_avoid_second_call(tmp_path):
    _fresh_cache(tmp_path)
    c = _CountingClient()
    wrapped = _maybe_cache(c, "anthropic", Settings(llm_cache=True, self_consistency_k=1))
    a = wrapped.complete("sys", "user", temperature=0.0, json_mode=True)
    b = wrapped.complete("sys", "user", temperature=0.0, json_mode=True)
    assert a == b
    assert c.calls == 1  # second call served from cache


def test_different_prompt_misses(tmp_path):
    _fresh_cache(tmp_path)
    c = _CountingClient()
    wrapped = _maybe_cache(c, "anthropic", Settings(llm_cache=True, self_consistency_k=1))
    wrapped.complete("sys", "user-A")
    wrapped.complete("sys", "user-B")
    assert c.calls == 2


def test_k_gt_1_disables_cache(tmp_path):
    # Safety guard: with self-consistency on, the cache must NOT engage (would zero out
    # aleatoric uncertainty by returning identical samples).
    _fresh_cache(tmp_path)
    c = _CountingClient()
    wrapped = _maybe_cache(c, "anthropic", Settings(llm_cache=True, self_consistency_k=3))
    assert wrapped is c
    wrapped.complete("sys", "user")
    wrapped.complete("sys", "user")
    assert c.calls == 2  # every sample is a real call
