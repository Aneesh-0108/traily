"""Online preparation-time fetch of OSM places via the Overpass API.

Used ONLY by `scripts/build_pack.py`. Never imported at runtime: the
runtime reads the saved local pack and makes zero network requests
(tests enforce this by blocking sockets during lookup).

No new dependency: stdlib `urllib` + `json` do the job (one POST, one
response), which keeps the offline runtime untouched per AGENTS.md.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
USER_AGENT = "Traily/Phase2 offline-pack-builder (Hacktoberfest demo; single query)"
REQUEST_TIMEOUT_S = 90
SERVER_TIMEOUT_S = 60

# (osm_key, osm_value, traily_type) — named features useful for "what's nearby?".
TAG_FILTERS: tuple[tuple[str, str, str], ...] = (
    ("leisure", "park", "park"),
    ("leisure", "garden", "garden"),
    ("amenity", "cafe", "cafe"),
    ("amenity", "library", "library"),
    ("amenity", "place_of_worship", "place_of_worship"),
    ("amenity", "restaurant", "restaurant"),
    ("tourism", "museum", "museum"),
    ("tourism", "viewpoint", "viewpoint"),
    ("tourism", "attraction", "attraction"),
    ("historic", "monument", "monument"),
    ("historic", "memorial", "memorial"),
    ("natural", "viewpoint", "viewpoint"),
)


class OverpassError(RuntimeError):
    """Fetch or response problems (network, HTTP status, bad JSON)."""


def build_query(
    min_lat: float, min_lon: float, max_lat: float, max_lon: float
) -> str:
    """One Overpass QL request: nodes/ways/relations matching TAG_FILTERS.

    A single request (no parallel calls) respecting public-service limits.
    Ways/relations use `out center` so every element yields coordinates.
    """
    bbox = f"{min_lat},{min_lon},{max_lat},{max_lon}"
    parts = []
    for key, value, _ in TAG_FILTERS:
        filt = f'["{key}"="{value}"]'
        parts.append(f"node{filt}({bbox});")
        parts.append(f"way{filt}({bbox});")
        parts.append(f"relation{filt}({bbox});")
    return f"[out:json][timeout:{SERVER_TIMEOUT_S}];({''.join(parts)});out center;"


def fetch_elements(query: str) -> list[dict[str, Any]]:
    """POST one query; return the raw `elements` list. Raises OverpassError."""
    req = urllib.request.Request(
        OVERPASS_URL,
        data=query.encode("utf-8"),
        headers={"User-Agent": USER_AGENT, "Content-Type": "text/plain"},
    )
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise OverpassError(f"Overpass HTTP {exc.code}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise OverpassError(f"Overpass unreachable: {exc.reason}") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise OverpassError(f"Overpass returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("elements"), list):
        raise OverpassError("Overpass response has no 'elements' list")
    return payload["elements"]


def _element_coords(el: dict[str, Any]) -> tuple[float, float] | None:
    if el.get("type") == "node":
        lat, lon = el.get("lat"), el.get("lon")
    else:
        center = el.get("center") or {}
        lat, lon = center.get("lat"), center.get("lon")
    if (
        isinstance(lat, (int, float))
        and isinstance(lon, (int, float))
        and not isinstance(lat, bool)
        and not isinstance(lon, bool)
        and -90.0 <= lat <= 90.0
        and -180.0 <= lon <= 180.0
    ):
        return float(lat), float(lon)
    return None


def _element_type(el: dict[str, Any], tags: dict[str, Any]) -> str | None:
    for key, value, traily_type in TAG_FILTERS:
        if tags.get(key) == value:
            return traily_type
    return None


def extract_places(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize elements to pack places. Drops (never invents):

    - elements without a non-empty name (cannot be spoken about);
    - elements matching none of TAG_FILTERS;
    - elements without usable coordinates (incl. ways/relations lacking center);
    - duplicates by (osm_type, osm_id), keeping the first occurrence.
    """
    places: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for el in elements:
        if not isinstance(el, dict):
            continue
        osm_type = el.get("type")
        osm_id = el.get("id")
        if osm_type not in ("node", "way", "relation"):
            continue
        if not isinstance(osm_id, int) or isinstance(osm_id, bool):
            continue
        if (osm_type, osm_id) in seen:
            continue
        tags = el.get("tags") or {}
        if not isinstance(tags, dict):
            continue
        name = tags.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        traily_type = _element_type(el, tags)
        if traily_type is None:
            continue
        coords = _element_coords(el)
        if coords is None:
            continue
        lat, lon = coords
        seen.add((osm_type, osm_id))
        places.append(
            {
                "name": name.strip(),
                "type": traily_type,
                "lat": lat,
                "lon": lon,
                "osm_id": osm_id,
                "osm_type": osm_type,
            }
        )
    return places
