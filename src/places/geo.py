"""Local geographic math and nearby-place lookup over an offline trail pack.

Everything here is deterministic and offline: haversine distances in
meters, sorting, and limiting. The LLM never sees coordinates and never
computes distances — it receives the same {name, type, distance_m}
contract as the Phase 1 fixture, so `build_prompt` is unchanged.
"""

from __future__ import annotations

import math
from typing import Any

from src import config
from src.places.pack import in_bbox, is_valid_lat, is_valid_lon

EARTH_RADIUS_M = 6_371_000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters between two WGS84 points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


class GeoError(ValueError):
    """Invalid coordinates or pack coverage problems."""


def nearest_in_pack(
    pack: dict[str, Any],
    lat: float,
    lon: float,
    limit: int = config.MAX_PLACES,
) -> list[dict[str, Any]]:
    """Return up to `limit` nearest places as {name, type, distance_m}.

    Raises GeoError for invalid coordinates or coordinates outside the
    pack bbox. Returns [] when the pack holds no usable places (caller
    reports that honestly instead of asking the LLM to invent any).
    """
    if not is_valid_lat(lat) or not is_valid_lon(lon):
        raise GeoError(f"invalid coordinates: lat={lat!r} lon={lon!r}")
    if not in_bbox(lat, lon, pack["bbox"]):
        raise GeoError(
            f"coordinates ({lat}, {lon}) are outside pack '{pack['area_name']}' coverage"
        )
    ranked: list[dict[str, Any]] = []
    for place in pack.get("places", []):
        plat, plon = place.get("lat"), place.get("lon")
        if not is_valid_lat(plat) or not is_valid_lon(plon):
            continue  # validated at build/load; skip defensively
        ranked.append(
            {
                "name": place["name"],
                "type": place["type"],
                "distance_m": int(round(haversine_m(lat, lon, plat, plon))),
            }
        )
    ranked.sort(key=lambda p: p["distance_m"])
    return ranked[: max(0, limit)]
