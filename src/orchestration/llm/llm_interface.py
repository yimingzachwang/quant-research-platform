"""Provider abstraction for LLM calls.

This module is the only place that imports any LLM vendor SDK.
All other orchestration code references this interface only.

Supported providers:
- "anthropic"  — Anthropic Claude (claude-sonnet-4-6 default)
- "openai"     — OpenAI Chat Completions (gpt-4o default)
- "stub"       — Returns a static placeholder; useful for tests and CI

Provider selection is explicit — no auto-detection, no environment magic.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from src.orchestration.llm.review_schema import (
    PROVIDER_ANTHROPIC,
    PROVIDER_OPENAI,
    PROVIDER_STUB,
)

logger = logging.getLogger(__name__)


class LLMResponse:
    """Normalised response across providers."""

    def __init__(self, text: str, model: str, provider: str, usage: dict[str, int]):
        self.text = text
        self.model = model
        self.provider = provider
        self.usage = usage


def call_llm(
    prompt: str,
    provider: str = PROVIDER_ANTHROPIC,
    model: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.2,
    system: str | None = None,
    base_url: str | None = None,
) -> LLMResponse:
    """Call the selected provider and return a normalised LLMResponse.

    Args:
        prompt:      The full user-turn prompt (context + instructions).
        provider:    One of "anthropic", "openai", "stub".
        model:       Provider-specific model ID; uses sensible default if None.
        max_tokens:  Maximum completion tokens.
        temperature: Sampling temperature (low = more deterministic).
        system:      Optional system prompt prepended before the user turn.
        base_url:    Override base URL for OpenAI-compatible endpoints (e.g.
                     LM Studio at "http://127.0.0.1:1234/v1").  Ignored for
                     the Anthropic provider.

    Returns:
        LLMResponse with .text, .model, .provider, .usage.

    Raises:
        ValueError: Unknown provider.
        RuntimeError: Provider SDK unavailable or API key missing.
    """
    if provider == PROVIDER_STUB:
        return _call_stub(prompt, model or "stub")

    if provider == PROVIDER_ANTHROPIC:
        return _call_anthropic(prompt, model, max_tokens, temperature, system)

    if provider == PROVIDER_OPENAI:
        return _call_openai(prompt, model, max_tokens, temperature, system, base_url)

    raise ValueError(
        f"Unknown provider {provider!r}. Use one of: "
        f"{PROVIDER_ANTHROPIC}, {PROVIDER_OPENAI}, {PROVIDER_STUB}"
    )


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------


def _call_anthropic(
    prompt: str,
    model: str | None,
    max_tokens: int,
    temperature: float,
    system: str | None,
) -> LLMResponse:
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError(
            "anthropic SDK not installed. Run: pip install anthropic"
        ) from exc

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set.")

    resolved_model = model or "claude-sonnet-4-6"
    client = anthropic.Anthropic(api_key=api_key)

    kwargs: dict[str, Any] = {
        "model": resolved_model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system

    response = client.messages.create(**kwargs)
    text = response.content[0].text if response.content else ""
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    return LLMResponse(text=text, model=resolved_model, provider=PROVIDER_ANTHROPIC, usage=usage)


def _call_openai(
    prompt: str,
    model: str | None,
    max_tokens: int,
    temperature: float,
    system: str | None,
    base_url: str | None = None,
) -> LLMResponse:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "openai SDK not installed. Run: pip install openai"
        ) from exc

    api_key = os.getenv("OPENAI_API_KEY")
    # Local OpenAI-compatible servers (e.g. LM Studio) do not require a real
    # API key — accept a placeholder when base_url is provided.
    if not api_key:
        if base_url:
            api_key = "lm-studio"
        else:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    resolved_model = model or "gpt-4o"
    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=resolved_model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    text = response.choices[0].message.content or ""
    usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }
    return LLMResponse(text=text, model=resolved_model, provider=PROVIDER_OPENAI, usage=usage)


def _call_stub(prompt: str, model: str) -> LLMResponse:
    """Deterministic stub for testing — never calls an external API."""
    text = (
        "[STUB LLM RESPONSE]\n\n"
        "This is a placeholder returned by the stub provider.\n"
        "Set provider='anthropic' or provider='openai' to call a real model.\n\n"
        f"Prompt length: {len(prompt)} characters."
    )
    return LLMResponse(text=text, model=model, provider=PROVIDER_STUB, usage={})


# ---------------------------------------------------------------------------
# Embeddings (local OpenAI-compatible endpoint, e.g. LM Studio)
# ---------------------------------------------------------------------------

# Default local embedding model (LM Studio / nomic). Embeddings never call a
# chat/completion model — only the embeddings endpoint.
DEFAULT_EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5"
_STUB_EMBED_DIM = 64


class EmbeddingResponse:
    """Normalised embedding response across providers."""

    def __init__(self, vectors: list[list[float]], model: str, provider: str):
        self.vectors = vectors
        self.model = model
        self.provider = provider
        self.dim = len(vectors[0]) if vectors else 0


def embed_texts(
    texts: list[str],
    provider: str = PROVIDER_OPENAI,
    model: str | None = None,
    base_url: str | None = None,
) -> EmbeddingResponse:
    """Embed a list of texts with a local embedding model. Never calls chat/LLM.

    Args:
        texts:    The compact texts to embed (one vector returned per text).
        provider: "openai" (OpenAI-compatible embeddings endpoint, incl. LM
                  Studio) or "stub" (deterministic offline pseudo-embeddings).
        model:    Embedding model id; defaults to DEFAULT_EMBEDDING_MODEL.
        base_url: Override base URL for OpenAI-compatible endpoints (e.g. LM
                  Studio at "http://127.0.0.1:1234/v1").

    Returns:
        EmbeddingResponse with .vectors, .model, .provider, .dim.

    Raises:
        ValueError:  Unknown provider.
        RuntimeError: SDK unavailable, API key missing, or endpoint failure.
    """
    if provider == PROVIDER_STUB:
        return _embed_stub(texts, model or "stub-embed")
    if provider == PROVIDER_OPENAI:
        return _embed_openai(texts, model, base_url)
    raise ValueError(
        f"Unknown embedding provider {provider!r}. Use one of: "
        f"{PROVIDER_OPENAI}, {PROVIDER_STUB}"
    )


def _embed_openai(
    texts: list[str],
    model: str | None,
    base_url: str | None,
) -> EmbeddingResponse:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai SDK not installed. Run: pip install openai") from exc

    api_key = os.getenv("OPENAI_API_KEY")
    # Local OpenAI-compatible servers (e.g. LM Studio) do not require a real key.
    if not api_key:
        if base_url:
            api_key = "lm-studio"
        else:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    resolved_model = model or DEFAULT_EMBEDDING_MODEL
    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    response = client.embeddings.create(model=resolved_model, input=texts)
    vectors = [list(item.embedding) for item in response.data]
    return EmbeddingResponse(vectors=vectors, model=resolved_model, provider=PROVIDER_OPENAI)


def _embed_stub(texts: list[str], model: str) -> EmbeddingResponse:
    """Deterministic offline pseudo-embeddings (token-hash bag), L2-normalised.

    Lets the semantic pipeline run and be tested without a live endpoint:
    identical text -> identical vector, shared tokens -> higher cosine.
    """
    import hashlib
    import math

    def vec(text: str) -> list[float]:
        acc = [0.0] * _STUB_EMBED_DIM
        for tok in (text or "").lower().split():
            h = int(hashlib.sha256(tok.encode()).hexdigest(), 16)
            acc[h % _STUB_EMBED_DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in acc)) or 1.0
        return [v / norm for v in acc]

    return EmbeddingResponse(
        vectors=[vec(t) for t in texts], model=model, provider=PROVIDER_STUB
    )
