"""
providers.py
------------
Provider configuration for LLM API clients.

Supported free providers (★ = recommended):
  deepseek    – DeepSeek ★ (free tier, OpenAI-compatible)
                https://platform.deepseek.com
                Env: DEEPSEEK_API_KEY
                Default model: deepseek-chat

  qwen        – Alibaba Qwen via DashScope (free tier, OpenAI-compatible)
                https://dashscope.aliyuncs.com
                Env: DASHSCOPE_API_KEY
                Default model: qwen-turbo

  groq        – Groq (free tier, ultra-fast inference, OpenAI-compatible)
                https://console.groq.com
                Env: GROQ_API_KEY
                Default model: llama-3.3-70b-versatile

  gemini      – Google Gemini (free tier, OpenAI-compatible endpoint)
                https://aistudio.google.com/apikey
                Env: GEMINI_API_KEY
                Default model: gemini-2.0-flash

  mistral     – Mistral AI (free tier, OpenAI-compatible)
                https://console.mistral.ai
                Env: MISTRAL_API_KEY
                Default model: mistral-small-latest

  together    – Together AI (free $25 credits, OpenAI-compatible)
                https://api.together.ai
                Env: TOGETHER_API_KEY
                Default model: meta-llama/Llama-3.3-70B-Instruct-Turbo-Free

  openrouter  – OpenRouter (many free models, OpenAI-compatible)
                https://openrouter.ai
                Env: OPENROUTER_API_KEY
                Default model: meta-llama/llama-3.1-8b-instruct:free

  openai      – OpenAI (paid)
                Env: OPENAI_API_KEY
                Default model: gpt-4o-mini
"""

import os
import sys
from dataclasses import dataclass, field

from openai import OpenAI


@dataclass
class ProviderConfig:
    name: str
    env_var: str
    base_url: str | None          # None → use OpenAI default
    default_model: str
    supports_seed: bool = True    # some providers ignore the seed param
    supports_json_mode: bool = True  # whether response_format=json_object is supported
    extra_kwargs: dict = field(default_factory=dict)


PROVIDERS: dict[str, ProviderConfig] = {
    # ── Free providers ───────────────────────────────────────────────────────
    "deepseek": ProviderConfig(
        name="DeepSeek",
        env_var="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com",
        default_model="deepseek-chat",
        supports_seed=True,
        supports_json_mode=True,
    ),
    "qwen": ProviderConfig(
        name="Qwen (Alibaba DashScope)",
        env_var="DASHSCOPE_API_KEY",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        default_model="qwen-turbo",
        supports_seed=False,      # DashScope ignores seed; temperature=0 for determinism
        supports_json_mode=True,
    ),
    "groq": ProviderConfig(
        name="Groq",
        env_var="GROQ_API_KEY",
        base_url="https://api.groq.com/openai/v1",
        default_model="llama-3.3-70b-versatile",
        supports_seed=False,
        supports_json_mode=True,
    ),
    "gemini": ProviderConfig(
        name="Google Gemini",
        env_var="GEMINI_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        default_model="gemini-2.0-flash",
        supports_seed=False,
        supports_json_mode=True,
    ),
    "mistral": ProviderConfig(
        name="Mistral AI",
        env_var="MISTRAL_API_KEY",
        base_url="https://api.mistral.ai/v1",
        default_model="mistral-small-latest",
        supports_seed=False,
        supports_json_mode=True,
    ),
    "together": ProviderConfig(
        name="Together AI",
        env_var="TOGETHER_API_KEY",
        base_url="https://api.together.xyz/v1",
        default_model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        supports_seed=False,
        supports_json_mode=False,  # free Llama models on Together don't support json_object
    ),
    "openrouter": ProviderConfig(
        name="OpenRouter",
        env_var="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        default_model="meta-llama/llama-3.1-8b-instruct:free",
        supports_seed=False,
        supports_json_mode=False,  # free models vary; skip to stay safe
    ),
    # ── Paid fallback ────────────────────────────────────────────────────────
    "openai": ProviderConfig(
        name="OpenAI",
        env_var="OPENAI_API_KEY",
        base_url=None,
        default_model="gpt-4o-mini",
        supports_seed=True,
        supports_json_mode=True,
    ),
}

# Detection priority: cheapest/fastest free providers first
_DETECT_ORDER = ["deepseek", "qwen", "groq", "gemini", "mistral", "together", "openrouter", "openai"]


def get_client(provider_name: str) -> tuple[OpenAI, ProviderConfig]:
    """
    Return an (OpenAI-compatible client, ProviderConfig) tuple for the given provider.
    Reads the API key from the appropriate environment variable and exits with a
    clear error message if it is missing.
    """
    provider_name = provider_name.lower()
    if provider_name not in PROVIDERS:
        sys.exit(
            f"ERROR: Unknown provider '{provider_name}'.\n"
            f"Supported providers: {', '.join(PROVIDERS)}"
        )

    cfg = PROVIDERS[provider_name]
    api_key = os.getenv(cfg.env_var)
    if not api_key:
        free_keys = (
            "  deepseek   → https://platform.deepseek.com      (DEEPSEEK_API_KEY)\n"
            "  qwen       → https://dashscope.aliyuncs.com     (DASHSCOPE_API_KEY)\n"
            "  groq       → https://console.groq.com           (GROQ_API_KEY)\n"
            "  gemini     → https://aistudio.google.com/apikey (GEMINI_API_KEY)\n"
            "  mistral    → https://console.mistral.ai         (MISTRAL_API_KEY)\n"
            "  together   → https://api.together.ai            (TOGETHER_API_KEY)\n"
            "  openrouter → https://openrouter.ai              (OPENROUTER_API_KEY)\n"
        )
        sys.exit(
            f"ERROR: {cfg.env_var} environment variable is not set.\n"
            f"Get a free API key from one of these providers:\n{free_keys}"
            f"Then add it to your .env file (see .env.example)."
        )

    client_kwargs: dict = {"api_key": api_key}
    if cfg.base_url:
        client_kwargs["base_url"] = cfg.base_url

    return OpenAI(**client_kwargs), cfg


def auto_detect_provider() -> str:
    """
    Return the first provider whose API key is found in the environment.
    Priority: deepseek → qwen → groq → gemini → mistral → together → openrouter → openai
    """
    for name in _DETECT_ORDER:
        cfg = PROVIDERS[name]
        if os.getenv(cfg.env_var):
            return name
    return "deepseek"   # will fail with a clear message if key is missing
