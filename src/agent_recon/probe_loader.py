"""Probe dataset loading and validation."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import yaml
from pydantic import ValidationError

from .models import Probe


class ProbeLoadError(Exception):
    """Raised when the probe dataset cannot be loaded or validated."""


def load_probes(path: str | Path) -> list[Probe]:
    """Load and validate the probe dataset.

    The file is expected to contain a YAML list of probe mappings.
    Each entry is validated against the :class:`Probe` model.

    Args:
        path: Path to the YAML probe dataset.

    Returns:
        A list of validated Probe objects, preserving file order.

    Raises:
        ProbeLoadError: If the file is missing, malformed, or contains
            entries that fail schema validation.
    """

    p = Path(path)
    if not p.exists():
        raise ProbeLoadError(f"Probe file not found: {p}")

    try:
        with p.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ProbeLoadError(f"Probe file is not valid YAML: {e}") from e

    if not isinstance(raw, list):
        raise ProbeLoadError(
            f"Probe file {p} must be a YAML list of probe objects, got {type(raw).__name__}."
        )

    probes: list[Probe] = []
    seen_ids: set[str] = set()

    for idx, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ProbeLoadError(f"Probe at index {idx} is not a mapping: {entry!r}")
        try:
            probe = Probe(**entry)
        except ValidationError as e:
            raise ProbeLoadError(
                f"Probe at index {idx} (id={entry.get('id', '<missing>')}) failed schema validation:\n{e}"
            ) from e
        if probe.id in seen_ids:
            raise ProbeLoadError(f"Duplicate probe id: {probe.id}")
        seen_ids.add(probe.id)
        probes.append(probe)

    if not probes:
        raise ProbeLoadError(f"Probe file {p} is empty.")

    return probes


def probes_by_category(probes: Iterable[Probe]) -> dict[str, list[Probe]]:
    """Group probes by their ``category`` field, preserving order."""

    grouped: dict[str, list[Probe]] = {}
    for probe in probes:
        grouped.setdefault(probe.category, []).append(probe)
    return grouped
