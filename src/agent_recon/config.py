"""Configuration loading for AI Agent Recon.

Configuration is layered:

  1. defaults baked into this module
  2. optional YAML file (config/config.yaml)
  3. environment variables (loaded from .env if present)
  4. CLI flags (applied by cli.py)

Lower numbers are overridden by higher numbers.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    provider: str = "openai"
    model: str = "gpt-4o-mini"
    # ``None`` means: do not pass a temperature override to the LLM (use the
    # model's default). This is required for the GPT-5 family and OpenAI
    # reasoning models (o1, o3, o4-mini, ...) which only accept the default
    # temperature value of 1.
    temperature: float | None = 0.1


class ScanConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    timeout: float = 30.0
    rate_limit_seconds: float = 1.0
    max_retries: int = 2


class TargetConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    method: str = "POST"
    body_template: dict[str, Any] = Field(default_factory=lambda: {"message": "{{prompt}}"})
    response_path: str | None = "response"


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    default_dir: str = "reports"
    default_format: str = "all"  # json | markdown | html | both | all


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    llm: LLMConfig = Field(default_factory=LLMConfig)
    scan: ScanConfig = Field(default_factory=ScanConfig)
    target: TargetConfig = Field(default_factory=TargetConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


DEFAULT_CONFIG_PATHS = (
    Path("config/config.yaml"),
    Path("config/config.example.yaml"),
)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a mapping at the top level.")
    return data


def _apply_env_overrides(cfg: AppConfig) -> AppConfig:
    """Apply select environment overrides on top of YAML."""

    provider = os.getenv("AGENT_RECON_LLM_PROVIDER")
    model = os.getenv("AGENT_RECON_LLM_MODEL") or os.getenv("OPENAI_MODEL_NAME")
    temp = os.getenv("AGENT_RECON_LLM_TEMPERATURE")

    if provider:
        cfg.llm.provider = provider
    if model:
        cfg.llm.model = model
    if temp is not None:
        t = temp.strip().lower()
        if t in {"", "none", "null", "default"}:
            cfg.llm.temperature = None
        else:
            try:
                cfg.llm.temperature = float(t)
            except ValueError:
                pass

    return cfg


def load_config(path: str | os.PathLike[str] | None = None) -> AppConfig:
    """Load configuration, layering YAML + env on top of defaults."""

    load_dotenv(override=False)

    if path is not None:
        yaml_data = _load_yaml(Path(path))
    else:
        yaml_data = {}
        for candidate in DEFAULT_CONFIG_PATHS:
            if candidate.exists():
                yaml_data = _load_yaml(candidate)
                break

    cfg = AppConfig(**yaml_data) if yaml_data else AppConfig()
    cfg = _apply_env_overrides(cfg)
    return cfg
