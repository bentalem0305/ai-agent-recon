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

    # v2.0: cross-check follow_up_ids references against the dataset.
    # The runtime selector already filters unknown / self IDs and the
    # depth cap, but surfacing typos at load time gives the operator
    # immediate feedback instead of silent "selector_chose_skip" at
    # scan time.
    probes_by_id = {probe.id: probe for probe in probes}
    for probe in probes:
        for fid in probe.follow_up_ids:
            if fid == probe.id:
                raise ProbeLoadError(
                    f"Probe {probe.id!r} declares a self-referential "
                    f"follow_up_id ({fid!r}). Remove it."
                )
            if fid not in probes_by_id:
                raise ProbeLoadError(
                    f"Probe {probe.id!r} declares follow_up_id {fid!r}, "
                    f"but no probe with that id is defined in {p}. "
                    f"Add the probe or remove the reference."
                )
            # Depth cap: a follow-up never has its own follow-up. The
            # runtime selector already enforces this, but rejecting at
            # load time is the right place to fail loud — operators
            # write YAML once, scans run many times.
            if probes_by_id[fid].follow_up_ids:
                raise ProbeLoadError(
                    f"Probe {probe.id!r} declares follow_up_id {fid!r}, "
                    f"but that probe itself declares follow_up_ids "
                    f"({probes_by_id[fid].follow_up_ids!r}). Adaptive "
                    f"follow-ups are capped at depth 1 — chains are "
                    f"not allowed. Remove follow_up_ids from {fid!r} "
                    f"or change the reference."
                )

    return probes


def probes_by_category(probes: Iterable[Probe]) -> dict[str, list[Probe]]:
    """Group probes by their ``category`` field, preserving order."""

    grouped: dict[str, list[Probe]] = {}
    for probe in probes:
        grouped.setdefault(probe.category, []).append(probe)
    return grouped
