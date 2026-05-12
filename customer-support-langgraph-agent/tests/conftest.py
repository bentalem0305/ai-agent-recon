"""Pytest configuration.

* Force every test run to use an isolated temp directory for memory and
  audit logs so tests don't pollute real ``data/`` or ``logs/`` files.
* Provide a fresh ``Settings`` object per test via the ``settings`` fixture.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

# Tell pytest to add src/ to sys.path so 'import supportmate' works without
# an editable install.
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture()
def temp_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Copy data + config into a tmp dir and re-root the app there."""
    # Copy the real data + config so KB / customers / orders are available.
    shutil.copytree(REPO_ROOT / "data", tmp_path / "data")
    shutil.copytree(REPO_ROOT / "config", tmp_path / "config")
    (tmp_path / "logs").mkdir()

    # Re-import config with the new root.
    monkeypatch.setenv("SUPPORTMATE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SUPPORTMATE_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("SUPPORTMATE_CONFIG_PATH", str(tmp_path / "config" / "config.yaml"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # Reset memory file to empty dict so we don't inherit prior state.
    mem = tmp_path / "data" / "session_memory.json"
    mem.write_text("{}", encoding="utf-8")

    # Reset tickets.
    tickets = tmp_path / "data" / "tickets.json"
    tickets.write_text("[]", encoding="utf-8")

    # Reset the cached Settings + the cached compiled graph.
    from supportmate.config import reset_settings_cache
    from supportmate.graph import get_compiled_graph

    reset_settings_cache()
    # Override project_root via monkeypatching get_settings's resolver.
    # The cleanest approach is to set SUPPORTMATE_DATA_DIR/LOG_DIR/CONFIG_PATH
    # (already done) and additionally point the Settings.project_root at tmp.
    # Re-cache:
    get_compiled_graph.cache_clear()

    # Monkeypatch project_root so that resolve() lands in tmp_path:
    from supportmate import config as cfgmod

    real = cfgmod.get_settings

    def patched_get_settings():
        s = real()
        s.project_root = tmp_path
        # Also rewrite memory path to absolute since we changed root mid-flight:
        s.memory.store_path = str(tmp_path / "data" / "session_memory.json")
        s.audit.log_path = str(tmp_path / "logs" / "audit.jsonl")
        s.knowledge_base.dir = str(tmp_path / "data" / "knowledge_base")
        s.data_dir = str(tmp_path / "data")
        return s

    monkeypatch.setattr(cfgmod, "get_settings", patched_get_settings)
    # Patch get_settings everywhere it's already bound by reference.
    import supportmate.audit as audit_mod
    import supportmate.memory as memory_mod
    import supportmate.tools.customer_tools as ct
    import supportmate.tools.order_tools as ot
    import supportmate.tools.ticket_tools as tt
    import supportmate.tools.kb_tools as kt
    import supportmate.nodes.router as router_mod
    import supportmate.api as api_mod

    for mod in (audit_mod, memory_mod, ct, ot, tt, kt, router_mod, api_mod):
        monkeypatch.setattr(mod, "get_settings", patched_get_settings)

    # Force the graph cache to refresh with the patched settings.
    get_compiled_graph.cache_clear()

    return tmp_path


@pytest.fixture()
def chat_client(temp_project):
    from fastapi.testclient import TestClient

    from supportmate.api import create_app

    return TestClient(create_app())


def _read_audit(tmp_path: Path) -> list[dict]:
    path = tmp_path / "logs" / "audit.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


@pytest.fixture()
def read_audit(temp_project):
    return lambda: _read_audit(temp_project)
