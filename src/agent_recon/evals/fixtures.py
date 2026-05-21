"""Ground-truth fixture schemas + loader for the mapper eval harness.

A *fixture* is a pair of files in the same directory:

  * ``<name>.json``           — a NormalizedRecon (or Phase-1 FinalReport)
                                input the mapper will consume. Same shape
                                you'd hand to ``ai-agent-recon pt``.
  * ``<name>.expected.json``  — the hand-labeled ground truth for that
                                input: per-ASI applicability + expected
                                priority, optional required signal
                                substrings, and per-fixture tolerance
                                settings.

The loader pairs them up by filename stem and yields :class:`EvalFixture`
objects. We deliberately store the truth alongside the input rather than
inside the input JSON: the input must remain a valid mapper input that
the production pipeline can also consume.

Why two priority tolerances?

The mapper's priority assignment is the noisiest signal — small changes
to the scoring weights can shift "Medium" ↔ "High" without changing
whether the *category* is correctly flagged. We support three modes:

  * ``exact``     — priority must match exactly. Use this for canonical
                    cases you want to nail.
  * ``one_step``  — predicted priority may differ from expected by one
                    step on the ladder (Informational < Low < Medium <
                    High < Critical). This is the default; it captures
                    "right ballpark" without being brittle.
  * ``any``       — priority is not scored at all; only applicability
                    is checked. Use for cases where you genuinely don't
                    care.
"""
from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..pt.schema import Priority


# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------

# Canonical ASI ladder used everywhere — the mapper emits all ten in
# this order, and the fixture loader requires expected entries to cover
# every category exactly once.
ASI_IDS: tuple[str, ...] = (
    "ASI01", "ASI02", "ASI03", "ASI04", "ASI05",
    "ASI06", "ASI07", "ASI08", "ASI09", "ASI10",
)

# Priority ladder, low → high. Index distance defines the "one step"
# tolerance.
_PRIORITY_LADDER: tuple[Priority, ...] = (
    "Informational", "Low", "Medium", "High", "Critical",
)


def priority_distance(a: Priority, b: Priority) -> int:
    """Absolute distance between two priorities on the ladder.

    Returns a large sentinel if either value is off-ladder (shouldn't
    happen for validated models, but we don't want to crash the metric
    pass if it does).
    """
    try:
        return abs(_PRIORITY_LADDER.index(a) - _PRIORITY_LADDER.index(b))
    except ValueError:
        return 999


class PriorityTolerance(str, Enum):
    """How strictly to compare predicted vs expected priority.

    See module docstring for semantics. ``one_step`` is the default
    because it captures "the mapper got the right ballpark" without
    punishing the kind of weight-tuning that doesn't actually change
    operator behavior.

    ``any`` here is the enum member, NOT the Python builtin — the
    builtin remains shadowed only inside this class scope, which is
    fine because we don't call ``any()`` from this enum's methods.
    """

    exact = "exact"
    one_step = "one_step"
    any = "any"


# ---------------------------------------------------------------------------
# Per-ASI expectation
# ---------------------------------------------------------------------------

class ExpectedAsiMapping(BaseModel):
    """Ground-truth expectation for a single OWASP ASI category.

    The fields are intentionally minimal: we score what the mapper is
    *supposed* to communicate to operators — "is this category in scope
    for this target, and roughly how urgent is it" — not the internal
    score breakdown.
    """

    model_config = ConfigDict(extra="forbid")

    asi_id: str
    applicable: bool
    # Only meaningful when ``applicable=True``. Required in that case;
    # the validator below enforces it.
    expected_priority: Priority | None = None
    # Optional: substrings that must appear in at least one of the
    # mapper's emitted signals for this category. Lets us assert that
    # the mapper picked up specific evidence (e.g. "has_mcp" for an MCP
    # target), not just the right applicability bit.
    expected_signals_must_contain: list[str] = Field(default_factory=list)
    note: str = ""

    @field_validator("asi_id")
    @classmethod
    def _valid_asi_id(cls, v: str) -> str:
        if v not in ASI_IDS:
            raise ValueError(
                f"asi_id must be one of {ASI_IDS}, got {v!r}"
            )
        return v

    @field_validator("expected_priority")
    @classmethod
    def _priority_when_applicable(cls, v: Priority | None) -> Priority | None:
        # We can't cross-validate against ``applicable`` here in v2
        # without a model_validator; the model_validator below does it.
        return v


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

class EvalFixture(BaseModel):
    """One hand-labeled mapper test case.

    A fixture pairs a recon input (the JSON file the mapper would
    actually receive) with the ground-truth expectations for what the
    mapper should produce.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    # Absolute path to the input JSON the mapper will consume. The
    # loader sets this from the on-disk pair so tests don't have to
    # hand-construct it.
    recon_path: Path
    # Must contain exactly one entry per ASI category (ASI01..ASI10).
    expected: list[ExpectedAsiMapping]
    priority_tolerance: PriorityTolerance = PriorityTolerance.one_step

    @field_validator("name")
    @classmethod
    def _non_empty_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("fixture name must be a non-empty string")
        return v.strip()

    @field_validator("expected")
    @classmethod
    def _covers_all_categories(
        cls, v: list[ExpectedAsiMapping]
    ) -> list[ExpectedAsiMapping]:
        seen = [e.asi_id for e in v]
        if sorted(seen) != list(ASI_IDS):
            missing = sorted(set(ASI_IDS) - set(seen))
            extra = sorted(set(seen) - set(ASI_IDS))
            dupes = sorted({a for a in seen if seen.count(a) > 1})
            raise ValueError(
                "fixture 'expected' must contain exactly one entry per ASI "
                "category (ASI01..ASI10). "
                f"missing={missing} extra={extra} duplicate={dupes}"
            )
        # Cross-field check: applicable=True ⇒ expected_priority required.
        for e in v:
            if e.applicable and e.expected_priority is None:
                raise ValueError(
                    f"{e.asi_id}: expected_priority is required when "
                    "applicable=True"
                )
        return v


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _iter_fixture_files(directory: Path) -> Iterator[tuple[Path, Path]]:
    """Yield (recon_path, expected_path) pairs from ``directory``.

    Pairing rule: for every ``<stem>.json`` that is NOT itself an
    expected file (no ``.expected.json`` suffix), look for the matching
    ``<stem>.expected.json`` in the same directory. A recon file
    without a matching expected file is skipped with no error (so
    operators can drop new inputs into the dir before labeling them);
    an expected file without a matching recon raises.
    """
    expected_files: dict[str, Path] = {}
    recon_files: dict[str, Path] = {}

    for p in sorted(directory.glob("*.json")):
        name = p.name
        if name.endswith(".expected.json"):
            stem = name[: -len(".expected.json")]
            expected_files[stem] = p
        else:
            recon_files[p.stem] = p

    orphans = set(expected_files) - set(recon_files)
    if orphans:
        raise FileNotFoundError(
            "Found expected files with no matching recon input: "
            + ", ".join(sorted(orphans))
        )

    for stem, recon_path in recon_files.items():
        if stem in expected_files:
            yield recon_path, expected_files[stem]


def load_fixtures(fixtures_dir: str | Path) -> list[EvalFixture]:
    """Load every labeled fixture pair from ``fixtures_dir``.

    Each pair becomes an :class:`EvalFixture`. The loader does NOT
    validate the recon input itself — that's the runner's job — but it
    does fully validate the expected JSON via Pydantic.

    Raises FileNotFoundError if the directory doesn't exist or contains
    an ``.expected.json`` file with no matching recon input.
    """
    directory = Path(fixtures_dir)
    if not directory.exists():
        raise FileNotFoundError(f"Fixtures directory not found: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    fixtures: list[EvalFixture] = []
    for recon_path, expected_path in _iter_fixture_files(directory):
        with expected_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            raise ValueError(
                f"{expected_path}: expected file must be a JSON object"
            )
        # The on-disk file shouldn't repeat the recon_path (the loader
        # supplies it). If the user did supply one, ignore it.
        raw.pop("recon_path", None)
        raw.setdefault("name", recon_path.stem)
        raw["recon_path"] = recon_path
        fixtures.append(EvalFixture.model_validate(raw))

    return fixtures


__all__ = [
    "ASI_IDS",
    "EvalFixture",
    "ExpectedAsiMapping",
    "PriorityTolerance",
    "load_fixtures",
    "priority_distance",
]
