"""Application configuration.

Loads settings in this order (lowest -> highest precedence):
    1. defaults defined here
    2. config/config.yaml
    3. environment variables / .env

Configuration is exposed as a single ``Settings`` Pydantic model. Path-style
attributes are always resolved against the *project root* (the directory
containing this package's parent) so the app behaves identically when launched
from the repo root, from `src/`, or as an installed package.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


def _project_root() -> Path:
    """Return the project root (the directory containing `src/`)."""
    # This file is .../src/supportmate/config.py  ->  parents[2] is the project root.
    return Path(__file__).resolve().parents[2]


class AppInfo(BaseModel):
    name: str = "SupportMate"
    version: str = "1.0.0"
    purpose: str = "LangGraph-based customer support AI agent."


class ServerSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000


class LLMSettings(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_tokens: int = 600
    api_key: str | None = None
    base_url: str | None = None


class MemorySettings(BaseModel):
    store_path: str = "data/session_memory.json"
    max_summary_chars: int = 800


class AuditSettings(BaseModel):
    log_path: str = "logs/audit.jsonl"


class KBSettings(BaseModel):
    dir: str = "data/knowledge_base"
    max_snippets: int = 3
    snippet_chars: int = 600


class Settings(BaseModel):
    project_root: Path = Field(default_factory=_project_root)
    app: AppInfo = Field(default_factory=AppInfo)
    server: ServerSettings = Field(default_factory=ServerSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    audit: AuditSettings = Field(default_factory=AuditSettings)
    knowledge_base: KBSettings = Field(default_factory=KBSettings)

    data_dir: str = "data"
    log_dir: str = "logs"

    # ---- helpers ---------------------------------------------------------

    def resolve(self, rel_or_abs: str | os.PathLike[str]) -> Path:
        """Resolve a path relative to the project root if not absolute."""
        p = Path(rel_or_abs)
        return p if p.is_absolute() else (self.project_root / p)

    @property
    def memory_path(self) -> Path:
        return self.resolve(self.memory.store_path)

    @property
    def audit_path(self) -> Path:
        return self.resolve(self.audit.log_path)

    @property
    def kb_dir(self) -> Path:
        return self.resolve(self.knowledge_base.dir)

    @property
    def customers_path(self) -> Path:
        return self.resolve(self.data_dir) / "customers.json"

    @property
    def orders_path(self) -> Path:
        return self.resolve(self.data_dir) / "orders.json"

    @property
    def tickets_path(self) -> Path:
        return self.resolve(self.data_dir) / "tickets.json"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        return {}
    return data


def _apply_env_overrides(settings: Settings) -> Settings:
    """Layer environment variables on top of the YAML/default config."""
    env_host = os.getenv("SUPPORTMATE_HOST")
    env_port = os.getenv("SUPPORTMATE_PORT")
    if env_host:
        settings.server.host = env_host
    if env_port:
        try:
            settings.server.port = int(env_port)
        except ValueError:
            pass

    env_data_dir = os.getenv("SUPPORTMATE_DATA_DIR")
    env_log_dir = os.getenv("SUPPORTMATE_LOG_DIR")
    if env_data_dir:
        settings.data_dir = env_data_dir
    if env_log_dir:
        settings.log_dir = env_log_dir
        # Re-point audit log if it still points at the default folder.
        if settings.audit.log_path.startswith("logs/"):
            settings.audit.log_path = f"{env_log_dir.rstrip('/')}/audit.jsonl"

    settings.llm.api_key = os.getenv("OPENAI_API_KEY") or None
    if os.getenv("OPENAI_MODEL"):
        settings.llm.model = os.environ["OPENAI_MODEL"]
    if os.getenv("OPENAI_BASE_URL"):
        settings.llm.base_url = os.environ["OPENAI_BASE_URL"]
    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached, fully-resolved Settings instance."""
    root = _project_root()
    yaml_path = Path(os.getenv("SUPPORTMATE_CONFIG_PATH", root / "config" / "config.yaml"))
    if not yaml_path.is_absolute():
        yaml_path = root / yaml_path
    raw = _load_yaml(yaml_path)
    settings = Settings(project_root=root, **raw)
    return _apply_env_overrides(settings)


def reset_settings_cache() -> None:
    """Useful in tests when env vars or YAML have been monkey-patched."""
    get_settings.cache_clear()
