"""AI Agent Recon — safe, authorized reconnaissance for AI-agent applications.

v2.0 surface
------------

For library users who want to drive the tool programmatically (rather than
through the CLI), the top-level package re-exports the v2.0 building blocks:

  * :class:`Probe`, :class:`ProbeResult` and the v2 schema additions
    (:class:`ProbeTurn`, :class:`TurnResponse`, :class:`DifferentialRun`,
    :class:`DifferentialResult`, :class:`VerifiedDefense`,
    :class:`TransportKind`).
  * :class:`Transport` Protocol with the two shipped implementations
    (:class:`HttpTransport`, :class:`SseTransport`) plus
    :func:`build_transport`.
  * :class:`ProbeExecutor` — the seam between :class:`Probe` and a transport
    that handles multi-turn, differential, and transport selection.
  * Adaptive follow-up selection: :class:`LlmFollowUpSelector`,
    :class:`FollowUpSelector` Protocol, :class:`FollowUpPlan`,
    :func:`plan_follow_up`.

The CLI imports the same surface internally, so anything documented here is
guaranteed to stay stable through v2.x.
"""

__version__ = "2.0.0"

# Models — schemas exchanged between every layer.
from .models import (
    CapabilityFinding,
    ClassificationResult,
    Confidence,
    DifferentialResult,
    DifferentialRun,
    FinalReport,
    FindingStatus,
    Probe,
    ProbeResult,
    ProbeTurn,
    ProbeType,
    RiskFinding,
    Severity,
    TargetInfo,
    TransportKind,
    TurnResponse,
    ValidationResult,
    VerifiedDefense,
)

# Transport + executor — v2.0 A.
from .probe_executor import ProbeExecutor
from .transports import (
    HttpTransport,
    SseTransport,
    Transport,
    TransportResponse,
    build_transport,
)

# Adaptive follow-ups — v2.0 B.
from .follow_ups import (
    FollowUpPlan,
    FollowUpSelector,
    LlmFollowUpSelector,
    plan_follow_up,
)

__all__ = [
    "__version__",
    # models
    "CapabilityFinding",
    "ClassificationResult",
    "Confidence",
    "DifferentialResult",
    "DifferentialRun",
    "FinalReport",
    "FindingStatus",
    "Probe",
    "ProbeResult",
    "ProbeTurn",
    "ProbeType",
    "RiskFinding",
    "Severity",
    "TargetInfo",
    "TransportKind",
    "TurnResponse",
    "ValidationResult",
    "VerifiedDefense",
    # transport + executor
    "HttpTransport",
    "ProbeExecutor",
    "SseTransport",
    "Transport",
    "TransportResponse",
    "build_transport",
    # follow-ups
    "FollowUpPlan",
    "FollowUpSelector",
    "LlmFollowUpSelector",
    "plan_follow_up",
]
