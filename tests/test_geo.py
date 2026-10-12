"""Tests for local geo math and pack-based nearby lookup.

Deterministic: no network, no GPS. Network access is actively blocked
during lookup tests to prove the offline guarantee.
"""

import socket
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.places.geo import GeoError, haversine_m, nearest_in_pack  # noqa: E402

PACK = {
    "schema_version": 1,
    "area_name": "Test Area",
    "bbox": {"min_lat": 12.9, "min_lon": 77.5, "max_lat": 13.0, "max_lon": 77.6},
    "generated_at": "2026-01-01T00:00:00+00:00",
    "places": [
        {"name": "Far Park", "type": "park", "lat": 12.99, "lon": 77.59,
         "osm_id": 1, "osm_type": "node"},
        {"name": "Near Cafe", "type": "cafe", "lat": 12.9505, "lon": 77.5505,
         "osm_id": 2, "osm_type": "node"},
        {"name": "Mid Library", "type": "library", "lat": 12.96, "lon": 77.56,
         "osm_id": 3, "osm_type": "way"},
    ],
}


class TestHaversine(unittest.TestCase):
    def test_known_distance(self):
        # ~111.19 m per 0.001 deg latitude at the equator.
        d = haversine_m(0.0, 0.0, 0.001, 0.0)
        self.assertAlmostEqual(d, 111.19, delta=0.5)

    def test_zero_distance(self):
        self.assertAlmostEqual(haversine_m(12.5, 77.5, 12.5, 77.5), 0.0)

    def test_symmetric(self):
        a = haversine_m(12.9, 77.5, 13.0, 77.6)
        b = haversine_m(13.0, 77.6, 12.9, 77.5)
        self.assertAlmostEqual(a, b)


class TestNearestInPack(unittest.TestCase):
    def test_ordering_and_contract(self):
        places = nearest_in_pack(PACK, 12.95, 77.55, limit=10)
        self.assertEqual([p["name"] for p in places], ["Near Cafe", "Mid Library", "Far Park"])
        for p in places:  # same contract as the Phase 1 fixture
            self.assertEqual(set(p), {"name", "type", "distance_m"})
            self.assertIsInstance(p["distance_m"], int)
        dists = [p["distance_m"] for p in places]
        self.assertEqual(dists, sorted(dists))
        self.assertLess(places[0]["distance_m"], 100)  # ~78 m away

    def test_limit(self):
        places = nearest_in_pack(PACK, 12.95, 77.55, limit=2)
        self.assertEqual(len(places), 2)

    def test_empty_pack_returns_empty(self):
        pack = dict(PACK, places=[])
        self.assertEqual(nearest_in_pack(pack, 12.95, 77.55), [])

    def test_outside_coverage(self):
        with self.assertRaises(GeoError):
            nearest_in_pack(PACK, 14.0, 77.55)

    def test_invalid_coords(self):
        for lat, lon in [(float("nan"), 77.5), (12.9, 999.0), ("x", 77.5)]:
            with self.subTest(lat=lat, lon=lon), self.assertRaises(GeoError):
                nearest_in_pack(PACK, lat, lon)  # type: ignore[arg-type]

    def test_lookup_needs_no_network(self):
        def _no_network(*a, **k):
            raise AssertionError("runtime lookup attempted network access")

        with mock.patch.object(socket, "socket", side_effect=_no_network), mock.patch(
            "socket.create_connection", side_effect=_no_network
        ):
            places = nearest_in_pack(PACK, 12.95, 77.55)
        self.assertEqual(len(places), 3)


if __name__ == "__main__":
    unittest.main()
