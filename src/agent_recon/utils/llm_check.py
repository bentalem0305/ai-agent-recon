"""Pre-flight LLM-credentials check.

Runs *before* any CrewAI agent is constructed, so we can:

  * Detect a missing API key cleanly (e.g. ``OPENAI_API_KEY`` not set).
  * Print a one-line user-friendly warning instead of letting CrewAI
    fail mid-kickoff with a ~200-line traceback.
  * Hand the orchestrator a clean signal so it can fall back to its
    deterministic / rule-based path.

The check is provider-aware: it knows which env var the configured
provider expects, and it reports the exact var name in the error
message so the user can fix it in one step.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from ..config import LLMConfig


# Map of CrewAI provider name -> env var name expected by that provider.
# Add new providers here as the project grows.
_PROVIDER_ENV: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "azure": "AZURE_OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "cohere": "COHERE_API_KEY",
}


@dataclass(slots=True)
class LLMAvailability:
    """Outcome of the pre-flight check."""

    available: bool
    provider: str
    env_var: str | None
    reason: str | None  # human-readable single-line reason if not available


def check_llm_available(cfg: LLMConfig) -> LLMAvailability:
    """Return whether the configured LLM has credentials available.

    Does NOT make any network call. Only inspects environment
    variables. Safe to call before any CrewAI imports.
    """
    provider = (cfg.provider or "openai").strip().lower()
    env_var = _PROVIDER_ENV.get(provider)

    if env_var is None:
        # Unknown provider - we can't pre-check, so assume available
        # and let CrewAI handle it.
        return LLMAvailability(
            available=True,
            provider=provider,
            env_var=None,
            reason=None,
        )

    key = os.getenv(env_var, "").strip()
    if not key:
        return LLMAvailability(
            available=False,
            provider=provider,
            env_var=env_var,
            reason=f"{env_var} is not set in the environment",
        )

    return LLMAvailability(
        available=True,
        provider=provider,
        env_var=env_var,
        reason=None,
    )
