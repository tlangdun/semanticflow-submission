"""LLM response caching for development efficiency.

Caches LLM responses to disk to speed up iteration during development
and reduce API costs.

Wired into the client call path via ``router._maybe_cache`` (gated by
``settings.llm_cache`` / ``SEMANTICFLOW_LLM_CACHE``). That wrapper FORCE-DISABLES the cache
when ``self_consistency_k > 1`` — a cache would return the same response for every sample,
collapsing aleatoric uncertainty to zero. Safe (and intended) for K=1 development runs to
cut API spend on repeated executions.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from semanticflow.log import get_logger

logger = get_logger("semanticflow.llm.cache")


class LLMCache:
    """Simple file-based cache for LLM responses."""
    
    def __init__(
        self,
        cache_dir: str | Path | None = None,
        enabled: bool = True,
        ttl_hours: int = 24,
    ) -> None:
        """Initialize the cache.
        
        Args:
            cache_dir: Directory to store cache files. Defaults to .llm_cache
            enabled: Whether caching is enabled
            ttl_hours: Cache time-to-live in hours (0 = no expiry)
        """
        self.enabled = enabled
        self.ttl_hours = ttl_hours
        
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.cwd() / ".llm_cache"
        
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._hits = 0
        self._misses = 0
    
    def _make_key(
        self,
        provider: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> str:
        """Create a cache key from request parameters.

        ``json_mode`` is part of the key because it changes the request sent to the
        provider (e.g. OpenAI ``response_format``) and therefore the output; omitting
        it would let JSON and non-JSON calls collide.
        """
        content = json.dumps({
            "provider": provider,
            "model": model,
            "system": system_prompt,
            "user": user_prompt,
            "temp": temperature,
            "json_mode": json_mode,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:32]
    
    def _get_cache_path(self, key: str) -> Path:
        """Get the file path for a cache key."""
        return self.cache_dir / f"{key}.json"
    
    def get(
        self,
        provider: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> str | None:
        """Get a cached response if available.
        
        Args:
            provider: LLM provider name
            model: Model name
            system_prompt: System prompt
            user_prompt: User prompt
            temperature: Temperature setting
        
        Returns:
            Cached response or None if not found/expired
        """
        if not self.enabled:
            return None
        
        key = self._make_key(provider, model, system_prompt, user_prompt, temperature, json_mode)
        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            self._misses += 1
            return None
        
        try:
            with cache_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Check TTL
            if self.ttl_hours > 0:
                import time
                cached_at = data.get("cached_at", 0)
                age_hours = (time.time() - cached_at) / 3600
                if age_hours > self.ttl_hours:
                    logger.debug(f"Cache expired for key {key[:8]}...")
                    self._misses += 1
                    return None
            
            self._hits += 1
            logger.debug(f"Cache hit for key {key[:8]}...", provider=provider, model=model)
            return data.get("response")
        
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
            self._misses += 1
            return None
    
    def set(
        self,
        provider: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        response: str,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> None:
        """Store a response in the cache.
        
        Args:
            provider: LLM provider name
            model: Model name
            system_prompt: System prompt
            user_prompt: User prompt
            response: The LLM response to cache
            temperature: Temperature setting
        """
        if not self.enabled:
            return
        
        key = self._make_key(provider, model, system_prompt, user_prompt, temperature, json_mode)
        cache_path = self._get_cache_path(key)

        try:
            import time
            data = {
                "provider": provider,
                "model": model,
                "cached_at": time.time(),
                "response": response,
            }
            
            with cache_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Cached response for key {key[:8]}...", provider=provider)
        
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
    
    def clear(self) -> int:
        """Clear all cached responses.
        
        Returns:
            Number of cache entries removed
        """
        if not self.cache_dir.exists():
            return 0
        
        count = 0
        for path in self.cache_dir.glob("*.json"):
            try:
                path.unlink()
                count += 1
            except Exception:
                pass
        
        logger.info(f"Cleared {count} cache entries")
        return count
    
    def stats(self) -> dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with hit/miss counts and cache size
        """
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        
        cache_size = 0
        cache_count = 0
        if self.cache_dir.exists():
            for path in self.cache_dir.glob("*.json"):
                cache_size += path.stat().st_size
                cache_count += 1
        
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "cache_count": cache_count,
            "cache_size_kb": cache_size / 1024,
        }


# Global cache instance
_global_cache: LLMCache | None = None


def get_cache(enabled: bool = True) -> LLMCache:
    """Get or create the global cache instance.
    
    Args:
        enabled: Whether caching should be enabled
    
    Returns:
        LLMCache instance
    """
    global _global_cache
    
    # Check environment variable
    env_enabled = os.getenv("SEMANTICFLOW_LLM_CACHE", "").lower()
    if env_enabled == "false" or env_enabled == "0":
        enabled = False
    
    if _global_cache is None:
        _global_cache = LLMCache(enabled=enabled)
    
    return _global_cache
