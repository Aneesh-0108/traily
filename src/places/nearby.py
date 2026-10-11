"""Local place loading, validation, and nearest-N filtering.

Phase 1 uses the fictional `data/places.json` fixture (no OSM, no GPS).
The LLM never sees more than MAX_PLACES items and never computes
distances itself: sorting happens here, in code, per AGENTS.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src import config

REQUIRED_FIELDS = ("name", "type", "distance_m")


def load_places(path: str | Path = config.PLACES_FILE) -> list[dict[str, Any]]:
    """Load and validate places; return nearest-first valid entries.

    Invalid entries are skipped (not fatal). Raises ValueError if the
    file is missing, not JSON, or contains zero valid places.
    """
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"places file not found: {path}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"places file is not valid JSON: {path}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("places"), list):
        raise ValueError(f"places file must contain a top-level 'places' list: {path}")
    valid = [p for p in data["places"] if is_valid_place(p)]
    if not valid:
        raise ValueError(f"places file contains no valid entries: {path}")
    valid.sort(key=lambda p: p["distance_m"])
    return valid


def is_valid_place(place: Any) -> bool:
    """A place needs a non-empty name/type and a non-negative distance."""
    if not isinstance(place, dict):
        return False
    if not all(f in place for f in REQUIRED_FIELDS):
        return False
    if not isinstance(place["name"], str) or not place["name"].strip():
        return False
    if not isinstance(place["type"], str) or not place["type"].strip():
        return False
    dist = place["distance_m"]
    if isinstance(dist, bool) or not isinstance(dist, (int, float)):
        return False
    return dist >= 0


def nearest_places(places: list[dict[str, Any]], limit: int = config.MAX_PLACES) -> list[dict[str, Any]]:
    """Return at most the `limit` nearest places (input must be sorted)."""
    return list(places[: max(0, limit)])
