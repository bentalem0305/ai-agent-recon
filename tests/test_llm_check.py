"""Tests for the pre-flight LLM-credentials check.

Locks in the contract that:

  * A missing API key for the configured provider is detected cleanly,
    without making any network call.
  * The reason / env-var name are surfaced for the orchestrator to put
    in user-facing log lines.
  * Unknown providers don't false-fail — we assume available and let
    CrewAI handle whatever happens.
"""
from __future__ import annotations

from agent_recon.config import LLMConfig
from agent_recon.utils.llm_check import check_llm_available


def test_openai_missing_key_is_detected(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = LLMConfig(provider="openai", model="gpt-4o-mini")

    result = check_llm_available(cfg)

    assert result.available is False
    assert result.provider == "openai"
    assert result.env_var == "OPENAI_API_KEY"
    assert "OPENAI_API_KEY" in result.reason


def test_openai_present_key_is_available(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-redacted")
    cfg = LLMConfig(provider="openai", model="gpt-4o-mini")

    result = check_llm_available(cfg)

    assert result.available is True
    assert result.env_var == "OPENAI_API_KEY"
    assert result.reason is None


def test_anthropic_missing_key_is_detected(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = LLMConfig(provider="anthropic", model="claude-opus-4")

    result = check_llm_available(cfg)

    assert result.available is False
    assert result.env_var == "ANTHROPIC_API_KEY"


def test_anthropic_present_key_is_available(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-test-redacted")
    cfg = LLMConfig(provider="anthropic", model="claude-opus-4")

    result = check_llm_available(cfg)

    assert result.available is True


def test_empty_key_is_treated_as_missing(monkeypatch) -> None:
    """A whitespace-only or empty env var should NOT count as configured."""
    monkeypatch.setenv("OPENAI_API_KEY", "   ")
    cfg = LLMConfig(provider="openai", model="gpt-4o-mini")

    result = check_llm_available(cfg)

    assert result.available is False
    assert "not set" in result.reason


def test_unknown_provider_is_assumed_available() -> None:
    """If we don't know the provider, don't false-fail - let CrewAI handle it."""
    cfg = LLMConfig(provider="some-future-provider", model="x")

    result = check_llm_available(cfg)

    assert result.available is True
    assert result.env_var is None
    assert result.reason is None
