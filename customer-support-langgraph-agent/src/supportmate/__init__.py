"""SupportMate — a LangGraph-based customer support AI agent.

The package implements a multi-node LangGraph state machine for customer
support tasks: input guardrails, intent routing, tenant-scoped
authorization, session memory, knowledge-base retrieval, tool execution,
escalation, response generation, and audit logging. Data sources are JSON
files in ``data/`` today and can be swapped for a real backing store.
"""

__version__ = "1.0.0"
__all__ = ["__version__"]
