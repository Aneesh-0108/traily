"""Offline trail-pack schema (v1): validated OSM places for one fixed area.

A pack is a plain JSON file stored under `data/packs/`, built once with
`scripts/build_pack.py` (online preparation) and read at runtime with no
network access. It never contains fictional data: only validated
OpenStreetMap elements.

Schema (schema_version == 1):
{
  "schema_version": 1,
  "area_name": "human label supplied at build time",
  "bbox": {"min_lat":.., "min_lon":.., "max_lat":.., "max_lon":..},
  "generated_at": "ISO-8601 UTC timestamp",
  "source": "OpenStreetMap Overpass API",
  "attribution": "© OpenStreetMap contributors",
  "license": "Open Database License (ODbL)",
  "license_url": "https://www.openstreetmap.org/copyright",
  "places": [
    {"name":.., "type":.., "lat":.., "lon":..,
     "osm_id":.., "osm_type": "node"|"way"|"relation"}
  ]
}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
ATTRIBUTION = "© OpenStreetMap contributors"
LICENSE = "Open Database License (ODbL)"
LICENSE_URL = "https://www.openstreetmap.org/copyright"
SOURCE = "OpenStreetMap Overpass API"

REQUIRED_TOP_FIELDS = (
    "schema_version",
    "area_name",
    "bbox",
    "generated_at",
    "places",
)
REQUIRED_BBOX_FIELDS = ("min_lat", "min_lon", "max_lat", "max_lon")
REQUIRED_PLACE_FIELDS = ("name", "type", "lat", "lon", "osm_id", "osm_type")
VALID_OSM_TYPES = ("node", "way", "relation")


def is_valid_lat(lat: Any) -> bool:
    return isinstance(lat, (int, float)) and not isinstance(lat, bool) and -90.0 <= lat <= 90.0


def is_valid_lon(lon: Any) -> bool:
    return isinstance(lon, (int, float)) and not isinstance(lon, bool) and -180.0 <= lon <= 180.0


def is_valid_bbox(bbox: Any) -> bool:
    if not isinstance(bbox, dict) or not all(f in bbox for f in REQUIRED_BBOX_FIELDS):
        return False
    vals = [bbox[f] for f in REQUIRED_BBOX_FIELDS]
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
        return False
    return bbox["min_lat"] <= bbox["max_lat"] and bbox["min_lon"] <= bbox["max_lon"]


def is_valid_pack_place(place: Any) -> bool:
    if not isinstance(place, dict) or not all(f in place for f in REQUIRED_PLACE_FIELDS):
        return False
    if not isinstance(place["name"], str) or not place["name"].strip():
        return False
    if not isinstance(place["type"], str) or not place["type"].strip():
        return False
    if not is_valid_lat(place["lat"]) or not is_valid_lon(place["lon"]):
        return False
    if not isinstance(place["osm_id"], int) or isinstance(place["osm_id"], bool):
        return False
    return place["osm_type"] in VALID_OSM_TYPES


def load_pack(path: str | Path) -> dict[str, Any]:
    """Load and validate a trail pack. Raises ValueError on any problem."""
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"trail pack not found: {path}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"trail pack is not valid JSON: {path}: {exc}") from exc
    problems = validate_pack(data)
    if problems:
        raise ValueError(f"invalid trail pack {path}: " + "; ".join(problems))
    return data


def validate_pack(data: Any) -> list[str]:
    """Return a list of schema problems (empty == valid). Never raises."""
    problems: list[str] = []
    if not isinstance(data, dict):
        return ["top level must be a JSON object"]
    for field in REQUIRED_TOP_FIELDS:
        if field not in data:
            problems.append(f"missing field: {field}")
    if data.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"unsupported schema_version (expected {SCHEMA_VERSION})")
    if "area_name" in data and (
        not isinstance(data["area_name"], str) or not data["area_name"].strip()
    ):
        problems.append("area_name must be a non-empty string")
    if "bbox" in data and not is_valid_bbox(data["bbox"]):
        problems.append("bbox must hold min_lat/min_lon/max_lat/max_lon with min <= max")
    places = data.get("places")
    if isinstance(places, list):
        bad = sum(1 for p in places if not is_valid_pack_place(p))
        if bad:
            problems.append(f"{bad} invalid place entries (dropped at build time, kept never)")
    elif "places" in data:
        problems.append("places must be a list")
    return problems


def in_bbox(lat: float, lon: float, bbox: dict[str, float]) -> bool:
    """True when coordinates fall inside the pack coverage area."""
    return bbox["min_lat"] <= lat <= bbox["max_lat"] and bbox["min_lon"] <= lon <= bbox["max_lon"]
