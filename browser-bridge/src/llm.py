"""LLM provider routing for browser-bridge.

Reads `tools_config.llm_provider` from PostgREST at request time so the bridge
follows whichever provider n8n-claw is currently configured for. The provider
config (provider name, model, api_key) lives in a single jsonb row in the
`tools_config` table written by setup.sh — that's the source of truth.

Env vars (ANTHROPIC_API_KEY etc.) are only a backwards-compat fallback for
local dev or instances that pre-date the centralized config.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import httpx

log = logging.getLogger(__name__)

SUPABASE_URL = os.environ.get("SUPABASE_URL", "http://kong:8000")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")

DEFAULT_PROVIDER = "anthropic"
DEFAULT_MODEL_BY_PROVIDER = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o",
    "openrouter": "anthropic/claude-sonnet-4-6",
    "deepseek": "deepseek-chat",
    "gemini": "gemini-2.0-flash",
    "mistral": "mistral-large-latest",
    "ollama": "qwen2.5:14b",
    "groq": "llama-3.3-70b-versatile",
}


@dataclass
class LLMConfig:
    provider: str
    model: str
    api_key: str | None = None
    endpoint: str | None = None  # for openai-compatible / ollama


async def fetch_active_provider() -> LLMConfig:
    """Query PostgREST for the active LLM provider configured in n8n-claw.

    Source of truth is `tools_config.llm_provider` (jsonb config with keys:
    provider, model, api_key, endpoint). Falls back to ANTHROPIC default
    only if the row is missing AND no env-based provider hint is available.
    """
    provider = None
    model = None
    api_key = None
    endpoint = None

    if SUPABASE_KEY:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    f"{SUPABASE_URL}/rest/v1/tools_config",
                    headers={
                        "apikey": SUPABASE_KEY,
                        "Authorization": f"Bearer {SUPABASE_KEY}",
                    },
                    params={"tool_name": "eq.llm_provider", "select": "config"},
                )
                resp.raise_for_status()
                rows = resp.json()
                if rows:
                    cfg = rows[0].get("config") or {}
                    provider = (cfg.get("provider") or "").strip().lower() or None
                    model = cfg.get("model")
                    api_key = cfg.get("api_key")
                    endpoint = cfg.get("endpoint")
                    log.info("Loaded llm_provider from tools_config: provider=%s, model=%s, has_key=%s",
                             provider, model, bool(api_key))
                else:
                    log.warning("tools_config.llm_provider row not found — falling back to env/defaults")
        except Exception as e:
            log.warning("Could not read tools_config from PostgREST (%s) — falling back to env/defaults", e)
    else:
        log.warning("SUPABASE_SERVICE_KEY not set — falling back to env/defaults")

    # Fall back to env if DB had nothing
    if not provider:
        # Pick whichever provider has an env-key set (least surprise on fresh installs)
        for p in ("anthropic", "openai", "openrouter", "gemini", "groq", "mistral", "deepseek"):
            if os.environ.get(f"{p.upper()}_API_KEY", "").strip():
                provider = p
                break
        if not provider:
            provider = DEFAULT_PROVIDER

    if not model:
        model = DEFAULT_MODEL_BY_PROVIDER.get(provider, DEFAULT_MODEL_BY_PROVIDER[DEFAULT_PROVIDER])

    if not api_key:
        api_key = os.environ.get(f"{provider.upper()}_API_KEY", "").strip() or None

    return LLMConfig(provider=provider, model=model, api_key=api_key, endpoint=endpoint)


def build_llm(cfg: LLMConfig):
    """Construct a browser_use.llm chat model for the given provider."""
    p = cfg.provider.lower()
    if not cfg.api_key and p != "ollama":
        raise RuntimeError(
            f"No API key available for provider {cfg.provider!r}. "
            f"Set tools_config.llm_provider.api_key in n8n-claw, or pass "
            f"{cfg.provider.upper()}_API_KEY as a container env var."
        )
    if p == "anthropic":
        from browser_use.llm import ChatAnthropic
        return ChatAnthropic(model=cfg.model, api_key=cfg.api_key)
    if p == "openai":
        from browser_use.llm import ChatOpenAI
        return ChatOpenAI(model=cfg.model, api_key=cfg.api_key)
    if p == "openrouter":
        from browser_use.llm import ChatOpenRouter
        return ChatOpenRouter(model=cfg.model, api_key=cfg.api_key)
    if p == "gemini":
        from browser_use.llm import ChatGoogle
        return ChatGoogle(model=cfg.model, api_key=cfg.api_key)
    if p == "ollama":
        from browser_use.llm import ChatOllama
        host = cfg.endpoint or os.environ.get("OLLAMA_HOST", "http://host.docker.internal:11434")
        return ChatOllama(model=cfg.model, host=host)
    if p == "groq":
        from browser_use.llm import ChatGroq
        return ChatGroq(model=cfg.model, api_key=cfg.api_key)
    if p == "mistral":
        from browser_use.llm import ChatMistral
        return ChatMistral(model=cfg.model, api_key=cfg.api_key)
    if p == "deepseek":
        from browser_use.llm import ChatDeepSeek
        return ChatDeepSeek(model=cfg.model, api_key=cfg.api_key)
    raise ValueError(f"Unsupported LLM provider for browser-bridge: {cfg.provider!r}")
