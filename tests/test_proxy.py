"""Tests for the proxy + verify-TLS plumbing.

These tests don't start a real proxy — they intercept ``httpx.Client``
construction and assert the right kwargs flow through from
:class:`TargetClientConfig` to the underlying HTTP client.

The config-layering test exercises env-var precedence end to end.
"""
from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent_recon.config import LLMConfig, ScanConfig, _apply_env_overrides
from agent_recon.config import AppConfig
from agent_recon.models import Probe, ProbeType
from agent_recon.target_client import TargetClient, TargetClientConfig


def _make_probe() -> Probe:
    return Probe(
        id="P-001",
        category="identity_and_role",
        probe_type=ProbeType.direct,
        prompt="hello",
        goal="test",
    )


# ---------------------------------------------------------------------------
# TargetClientConfig: defaults are conservative (no proxy, verify enabled)
# ---------------------------------------------------------------------------

def test_default_config_has_no_proxy_and_verify_on() -> None:
    cfg = TargetClientConfig(url="http://example.com")
    assert cfg.proxy is None
    assert cfg.verify_tls is True


# ---------------------------------------------------------------------------
# httpx.Client receives the proxy kwarg only when one is configured
# ---------------------------------------------------------------------------

def test_proxy_is_passed_to_httpx_client_when_set() -> None:
    cfg = TargetClientConfig(
        url="http://example.com/chat",
        proxy="http://127.0.0.1:8080",
        max_retries=0,
    )
    client = TargetClient(cfg)

    # Patch httpx.Client at the module level the target_client uses.
    with patch("agent_recon.target_client.httpx.Client") as mock_client_cls:
        mock_client_cls.return_value.__enter__.return_value.request = MagicMock(
            return_value=MagicMock(
                status_code=200,
                text='{"response": "ok"}',
                json=lambda: {"response": "ok"},
            )
        )
        client.send_probe(_make_probe())

    # httpx.Client called exactly once with proxy= set
    assert mock_client_cls.call_count == 1
    _, kwargs = mock_client_cls.call_args
    assert kwargs.get("proxy") == "http://127.0.0.1:8080"
    # verify is NOT passed when verify_tls is the default True
    assert "verify" not in kwargs


def test_no_proxy_kwarg_when_not_configured() -> None:
    cfg = TargetClientConfig(url="http://example.com/chat", max_retries=0)
    client = TargetClient(cfg)

    with patch("agent_recon.target_client.httpx.Client") as mock_client_cls:
        mock_client_cls.return_value.__enter__.return_value.request = MagicMock(
            return_value=MagicMock(
                status_code=200,
                text='{"response": "ok"}',
                json=lambda: {"response": "ok"},
            )
        )
        client.send_probe(_make_probe())

    _, kwargs = mock_client_cls.call_args
    assert "proxy" not in kwargs


def test_verify_false_is_passed_when_insecure() -> None:
    cfg = TargetClientConfig(
        url="https://example.com/chat",
        proxy="http://127.0.0.1:8080",
        verify_tls=False,
        max_retries=0,
    )
    client = TargetClient(cfg)

    with patch("agent_recon.target_client.httpx.Client") as mock_client_cls:
        mock_client_cls.return_value.__enter__.return_value.request = MagicMock(
            return_value=MagicMock(
                status_code=200,
                text='{"response": "ok"}',
                json=lambda: {"response": "ok"},
            )
        )
        client.send_probe(_make_probe())

    _, kwargs = mock_client_cls.call_args
    assert kwargs.get("proxy") == "http://127.0.0.1:8080"
    assert kwargs.get("verify") is False


def test_verify_kwarg_omitted_when_verify_tls_is_default() -> None:
    """When verify_tls=True (default), we don't pass verify= at all so we
    pick up httpx's own default (which is also True). This keeps the call
    signature minimal and forward-compatible."""
    cfg = TargetClientConfig(url="https://example.com/chat", max_retries=0)
    client = TargetClient(cfg)

    with patch("agent_recon.target_client.httpx.Client") as mock_client_cls:
        mock_client_cls.return_value.__enter__.return_value.request = MagicMock(
            return_value=MagicMock(
                status_code=200,
                text='{"response": "ok"}',
                json=lambda: {"response": "ok"},
            )
        )
        client.send_probe(_make_probe())

    _, kwargs = mock_client_cls.call_args
    assert "verify" not in kwargs


# ---------------------------------------------------------------------------
# Config-layering: env vars override YAML defaults
# ---------------------------------------------------------------------------

def test_env_var_overrides_yaml_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_RECON_PROXY", "http://10.0.0.1:9999")
    cfg = AppConfig(scan=ScanConfig(proxy="http://yaml-default:8080"))
    cfg = _apply_env_overrides(cfg)
    assert cfg.scan.proxy == "http://10.0.0.1:9999"


def test_env_var_overrides_yaml_verify_tls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_RECON_VERIFY_TLS", "false")
    cfg = AppConfig(scan=ScanConfig(verify_tls=True))
    cfg = _apply_env_overrides(cfg)
    assert cfg.scan.verify_tls is False


def test_verify_tls_env_var_accepts_truthy_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any of 1/true/yes/on enables; 0/false/no/off disables."""
    for truthy in ("1", "true", "True", "TRUE", "yes", "on"):
        monkeypatch.setenv("AGENT_RECON_VERIFY_TLS", truthy)
        cfg = _apply_env_overrides(AppConfig(scan=ScanConfig(verify_tls=False)))
        assert cfg.scan.verify_tls is True, f"truthy value {truthy!r} should enable"

    for falsy in ("0", "false", "False", "no", "off"):
        monkeypatch.setenv("AGENT_RECON_VERIFY_TLS", falsy)
        cfg = _apply_env_overrides(AppConfig(scan=ScanConfig(verify_tls=True)))
        assert cfg.scan.verify_tls is False, f"falsy value {falsy!r} should disable"


def test_no_proxy_env_var_means_yaml_value_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENT_RECON_PROXY", raising=False)
    cfg = AppConfig(scan=ScanConfig(proxy="http://yaml-default:8080"))
    cfg = _apply_env_overrides(cfg)
    assert cfg.scan.proxy == "http://yaml-default:8080"
