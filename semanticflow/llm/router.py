from __future__ import annotations

import time
from dataclasses import dataclass

from semanticflow.config import Settings
from semanticflow.llm.base import BaseLLMClient, LLMError
from semanticflow.llm.usage import UsageTracker
from semanticflow.log import get_logger

logger = get_logger("semanticflow.llm.router")


@dataclass
class ProviderSpec:
    name: str
    default_model: str


PROVIDERS: dict[str, ProviderSpec] = {
    "openai": ProviderSpec("openai", "gpt-4o"),
    "anthropic": ProviderSpec("anthropic", "claude-sonnet-4-6"),
    "gemini": ProviderSpec("gemini", "gemini-2.0-flash"),
    "mock": ProviderSpec("mock", "mock"),
}


class OpenAIClient(BaseLLMClient):
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        usage_tracker: UsageTracker | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - optional dependency
            raise LLMError("openai package is not installed") from exc
        client_kwargs: dict[str, object] = {"api_key": api_key, "base_url": base_url}
        if timeout is not None:
            client_kwargs["timeout"] = timeout
        if max_retries is not None:
            client_kwargs["max_retries"] = max_retries
        self._client = OpenAI(**client_kwargs)
        self._model = model
        self._temperature_supported: bool | None = None
        self._usage = usage_tracker

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> str:
        params = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if temperature is not None and self._temperature_supported is not False:
            params["temperature"] = temperature
        if json_mode:
            params["response_format"] = {"type": "json_object"}
        try:
            response = self._client.chat.completions.create(**params)
        except Exception as exc:  # pragma: no cover - network/API dependent
            message = str(exc)
            if temperature is not None and "temperature" in message and "unsupported" in message.lower():
                self._temperature_supported = False
                params.pop("temperature", None)
                response = self._client.chat.completions.create(**params)
            else:
                raise
        else:
            if temperature is not None and self._temperature_supported is None:
                self._temperature_supported = True
        if self._usage is not None:
            u = getattr(response, "usage", None)
            if u is not None:
                self._usage.add(
                    "openai", self._model,
                    int(getattr(u, "prompt_tokens", 0) or 0),
                    int(getattr(u, "completion_tokens", 0) or 0),
                )
        return response.choices[0].message.content or "{}"


class AnthropicClient(BaseLLMClient):
    def __init__(
        self,
        model: str,
        api_key: str,
        timeout: float | None = None,
        max_retries: int | None = None,
        max_tokens: int = 4096,
        usage_tracker: UsageTracker | None = None,
    ) -> None:
        try:
            from anthropic import Anthropic
        except Exception as exc:  # pragma: no cover - optional dependency
            raise LLMError("anthropic package is not installed") from exc
        client_kwargs: dict[str, object] = {"api_key": api_key}
        if timeout is not None:
            client_kwargs["timeout"] = timeout
        if max_retries is not None:
            client_kwargs["max_retries"] = max_retries
        try:
            self._client = Anthropic(**client_kwargs)
        except TypeError:
            self._client = Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._usage = usage_tracker

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=temperature,
        )
        # A truncated response yields invalid/partial JSON that can still parse into a
        # silently-wrong spec — surface it as an error instead.
        if getattr(response, "stop_reason", None) == "max_tokens":
            raise LLMError(
                f"Anthropic response hit max_tokens={self._max_tokens}; "
                "raise llm_max_tokens to avoid truncated specs"
            )
        if self._usage is not None:
            u = getattr(response, "usage", None)
            if u is not None:
                self._usage.add(
                    "anthropic", self._model,
                    int(getattr(u, "input_tokens", 0) or 0),
                    int(getattr(u, "output_tokens", 0) or 0),
                )
        text = "".join(block.text for block in response.content if hasattr(block, "text"))
        return text or "{}"


class GeminiClient(BaseLLMClient):
    def __init__(
        self,
        model: str,
        api_key: str,
        timeout: float | None = None,
        usage_tracker: UsageTracker | None = None,
    ) -> None:
        self._types = None
        self._usage = usage_tracker
        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise LLMError("google-genai package is not installed") from exc
        client_kwargs: dict[str, object] = {"api_key": api_key}
        if timeout is not None:
            # Google GenAI SDK timeout behavior is causing issues (Manually set deadline 1s...).
            # We will rely on asyncio.wait_for in the caller to handle timeouts.
            pass
            # if timeout < 10.0:
            #     timeout = 10.0
            # client_kwargs["http_options"] = {"timeout": timeout}
        try:
            self._client = genai.Client(**client_kwargs)
        except TypeError:
            self._client = genai.Client(api_key=api_key)
        self._model = model
        self._types = types

    def _extract_text(self, response: object) -> str:
        text = getattr(response, "text", None)
        if text:
            return text
        candidates = getattr(response, "candidates", None)
        if candidates:
            first = candidates[0]
            content = getattr(first, "content", None)
            parts = getattr(content, "parts", None) if content else None
            if parts:
                part_text = getattr(parts[0], "text", None)
                if part_text:
                    return part_text
        return "{}"

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> str:
        config_payload: dict[str, object] = {}
        if temperature is not None:
            config_payload["temperature"] = temperature
        if system_prompt:
            config_payload["system_instruction"] = system_prompt
        config = None
        if config_payload:
            if self._types is None:
                raise LLMError("google-genai types unavailable; cannot set temperature/system")
            try:
                config = self._types.GenerateContentConfig(**config_payload)
            except Exception as exc:
                # Do NOT fall back to passing the raw dict: the SDK may silently ignore it,
                # dropping temperature and collapsing the aleatoric signal to zero.
                raise LLMError(f"failed to build Gemini config: {exc}") from exc
        # Gemini has no built-in retry (unlike the OpenAI/Anthropic clients). Retry on
        # transient 429 RESOURCE_EXHAUSTED with backoff so a per-minute rate-limit window
        # does not fail an otherwise-valid call. Non-429 errors are raised immediately.
        backoffs = [3.0, 6.0, 10.0, 15.0, 20.0]
        attempt = 0
        while True:
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=user_prompt,
                    config=config,
                )
                break
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                is_rate = "429" in msg or "RESOURCE_EXHAUSTED" in msg
                if not is_rate or attempt >= len(backoffs):
                    raise
                logger.warning(
                    f"gemini 429 rate-limit, retrying in {backoffs[attempt]:.0f}s "
                    f"(attempt {attempt + 1})"
                )
                time.sleep(backoffs[attempt])
                attempt += 1
        if self._usage is not None:
            m = getattr(response, "usage_metadata", None)
            if m is not None:
                self._usage.add(
                    "gemini", self._model,
                    int(getattr(m, "prompt_token_count", 0) or 0),
                    int(getattr(m, "candidates_token_count", 0) or 0),
                )
        return self._extract_text(response)


class MockClient(BaseLLMClient):
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> str:
        return "{}"


def _maybe_cache(client: BaseLLMClient, provider: str, settings: Settings) -> BaseLLMClient:
    """Wrap ``client.complete`` with a disk cache when ``llm_cache`` is on.

    SAFETY: force-disabled when ``self_consistency_k > 1`` — identical cached responses
    would collapse the aleatoric uncertainty signal to zero. A cache hit also costs nothing,
    so it is correctly not recorded by the usage tracker.
    """
    if not getattr(settings, "llm_cache", False):
        return client
    if int(getattr(settings, "self_consistency_k", 1) or 1) > 1:
        logger.warning(
            "SEMANTICFLOW_LLM_CACHE ignored: self_consistency_k>1 would collapse the "
            "aleatoric signal. Set self_consistency_k=1 to use the cache."
        )
        return client

    from semanticflow.llm.cache import get_cache

    cache = get_cache(enabled=True)
    model = getattr(client, "_model", provider)
    inner = client.complete

    def cached_complete(system_prompt, user_prompt, temperature=None, json_mode=False):
        hit = cache.get(provider, model, system_prompt, user_prompt, temperature, json_mode)
        if hit is not None:
            return hit
        out = inner(system_prompt, user_prompt, temperature=temperature, json_mode=json_mode)
        cache.set(provider, model, system_prompt, user_prompt, out, temperature, json_mode)
        return out

    client.complete = cached_complete  # type: ignore[method-assign]
    return client


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# OpenRouter model ids are "<vendor>/<model>"; map our provider names to OpenRouter vendors.
_OPENROUTER_VENDOR = {"openai": "openai", "anthropic": "anthropic", "gemini": "google"}


def _openrouter_slug(provider: str, model: str) -> str:
    """Map a provider+model to an OpenRouter model id; pass through ids already namespaced."""
    if "/" in model:
        return model
    return f"{_OPENROUTER_VENDOR.get(provider, provider)}/{model}"


def build_provider_client(
    provider: str,
    settings: Settings,
    usage_tracker: UsageTracker | None = None,
) -> BaseLLMClient:
    provider = provider.lower()
    spec = PROVIDERS.get(provider)
    if not spec:
        raise LLMError(f"Unsupported LLM provider: {provider}")
    model = spec.default_model
    if provider == "mock":
        return MockClient()
    # Route real providers through OpenRouter (OpenAI-compatible) when enabled, to sidestep
    # per-provider rate limits. The same logical model is used, mapped to an OpenRouter slug.
    if (getattr(settings, "use_openrouter", False) and settings.openrouter_api_key
            and provider in ("openai", "anthropic", "gemini")):
        model = ({"openai": settings.openai_model, "anthropic": settings.anthropic_model,
                  "gemini": settings.gemini_model}.get(provider) or spec.default_model)
        client = OpenAIClient(
            model=_openrouter_slug(provider, model),
            api_key=settings.openrouter_api_key,
            base_url=OPENROUTER_BASE_URL,
            timeout=settings.llm_timeout_sec,
            max_retries=settings.llm_max_retries,
            usage_tracker=usage_tracker,
        )
        return _maybe_cache(client, provider, settings)
    if provider == "openai":
        if not settings.openai_api_key:
            raise LLMError("OPENAI_API_KEY is required for OpenAI")
        model = settings.openai_model or model
        client: BaseLLMClient = OpenAIClient(
            model=model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            timeout=settings.llm_timeout_sec,
            max_retries=settings.llm_max_retries,
            usage_tracker=usage_tracker,
        )
    elif provider == "anthropic":
        if not settings.anthropic_api_key:
            raise LLMError("ANTHROPIC_API_KEY is required for Anthropic")
        model = settings.anthropic_model or model
        client = AnthropicClient(
            model=model,
            api_key=settings.anthropic_api_key,
            timeout=settings.llm_timeout_sec,
            max_retries=settings.llm_max_retries,
            max_tokens=settings.llm_max_tokens,
            usage_tracker=usage_tracker,
        )
    elif provider == "gemini":
        if not settings.gemini_api_key:
            raise LLMError("GEMINI_API_KEY is required for Gemini")
        model = settings.gemini_model or model
        client = GeminiClient(
            model=model,
            api_key=settings.gemini_api_key,
            timeout=settings.llm_timeout_sec,
            usage_tracker=usage_tracker,
        )
    else:
        raise LLMError(f"Unsupported LLM provider: {provider}")
    return _maybe_cache(client, provider, settings)


def build_llm_client(settings: Settings, usage_tracker: UsageTracker | None = None) -> BaseLLMClient:
    return build_provider_client(settings.llm_provider.lower(), settings, usage_tracker)


# Cheapest sensible model per provider for low-stakes calls (simulated-human answers,
# last-resort execution repair, clarifying-question generation). Overridable via
# settings.cheap_llm_model / SEMANTICFLOW_CHEAP_MODEL.
CHEAP_MODELS: dict[str, str] = {
    "anthropic": "claude-haiku-4-5-20251001",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
}


def build_cheap_client(
    settings: Settings,
    usage_tracker: UsageTracker | None = None,
) -> BaseLLMClient | None:
    """Build a cheap-tier client for low-stakes calls, or None if no provider is usable.

    Provider preference: settings.cheap_llm_provider, then the single-provider
    llm_provider, then the ensemble providers in order. The model is downgraded to the
    provider's CHEAP_MODELS entry unless settings.cheap_llm_model overrides it. Goes
    through build_provider_client so the disk cache (llm_cache) and usage tracking apply
    exactly as for full-tier clients. Never returns a MockClient: callers treat None as
    "skip the LLM step" rather than burning a call on a placeholder.
    """
    from dataclasses import replace

    candidates: list[str] = []
    for prov in (
        settings.cheap_llm_provider,
        settings.llm_provider,
        *settings.ensemble_providers,
    ):
        prov = (prov or "").lower()
        if prov and prov != "mock" and prov in PROVIDERS and prov not in candidates:
            candidates.append(prov)

    for prov in candidates:
        model = settings.cheap_llm_model or CHEAP_MODELS.get(prov, PROVIDERS[prov].default_model)
        cheap_settings = replace(settings, **{f"{prov}_model": model})
        try:
            return build_provider_client(prov, cheap_settings, usage_tracker)
        except LLMError:
            continue
    return None
