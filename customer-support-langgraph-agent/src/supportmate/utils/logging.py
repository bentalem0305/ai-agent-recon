"""Lightweight Rich-based logging configuration."""

from __future__ import annotations

import logging

from rich.logging import RichHandler

_CONFIGURED = False


# Third-party libraries that emit INFO-level chatter we don't want
# polluting the CLI / FastAPI logs. Each is downgraded to WARNING
# unless the caller explicitly asks for verbose mode.
#
# We list every reasonable sub-logger because some of these libraries
# emit on the parent logger (e.g. `langchain`) and others on a child
# (e.g. `langchain_core.tracers`). Downgrading the parent alone isn't
# always enough.
_NOISY_LIBRARY_LOGGERS = (
    # HTTP
    "httpx",
    "httpcore",
    "urllib3",
    # OpenAI SDK + LiteLLM (LangChain uses one of these under the hood)
    "openai",
    "openai._base_client",
    "litellm",
    "LiteLLM",
    # LangChain / LangChain Core
    "langchain",
    "langchain.chat_models",
    "langchain.llms",
    "langchain.callbacks",
    "langchain_core",
    "langchain_core.tracers",
    "langchain_core.callbacks",
    "langchain_openai",
    # LangGraph
    "langgraph",
    "langgraph.graph",
    "langgraph.prebuilt",
    "langgraph.checkpoint",
    # Uvicorn access-log (we keep error log readable in serve mode)
    "uvicorn.access",
)


def configure_logging(level: int = logging.INFO) -> None:
    """Idempotent global logging setup.

    Installs a Rich handler at INFO by default and silences the
    library loggers listed in :data:`_NOISY_LIBRARY_LOGGERS` so that
    our own ``supportmate.*`` lines stand out.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = RichHandler(rich_tracebacks=True, show_path=False, markup=False)
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[handler],
    )

    # Route uvicorn through the Rich handler so its output matches ours.
    # We KEEP uvicorn.error at default level (we want startup / shutdown
    # / 5xx messages) and only quiet down access logs.
    logging.getLogger("uvicorn.error").handlers = [handler]
    logging.getLogger("uvicorn.access").handlers = [handler]

    # Silence noisy library INFO chatter.
    for noisy in _NOISY_LIBRARY_LOGGERS:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
