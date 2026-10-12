"""Tests for trail-pack schema validation and OSM element extraction.

Deterministic: fixtures only, Overpass HTTP mocked. No network, no GPS.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.places import overpass as ov  # noqa: E402
from src.places import pack as pk  # noqa: E402


def _good_pack() -> dict:
    return {
        "schema_version": 1,
        "area_name": "Test Area",
        "bbox": {"min_lat": 12.9, "min_lon": 77.5, "max_lat": 13.0, "max_lon": 77.6},
        "generated_at": "2026-01-01T00:00:00+00:00",
        "source": pk.SOURCE,
        "attribution": pk.ATTRIBUTION,
        "license": pk.LICENSE,
        "license_url": pk.LICENSE_URL,
        "places": [
            {"name": "A", "type": "park", "lat": 12.95, "lon": 77.55,
             "osm_id": 1, "osm_type": "node"},
        ],
    }


class TestPackSchema(unittest.TestCase):
    def test_valid_pack_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pack.json"
            path.write_text(json.dumps(_good_pack()), encoding="utf-8")
            pack = pk.load_pack(path)
        self.assertEqual(pack["area_name"], "Test Area")

    def test_missing_file(self):
        with self.assertRaises(ValueError):
            pk.load_pack(REPO_ROOT / "data" / "packs" / "nope.json")

    def test_bad_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.json"
            path.write_text("{nope", encoding="utf-8")
            with self.assertRaises(ValueError):
                pk.load_pack(path)

    def test_wrong_schema_version(self):
        data = _good_pack()
        data["schema_version"] = 99
        self.assertTrue(any("schema_version" in p for p in pk.validate_pack(data)))

    def test_bad_bbox(self):
        data = _good_pack()
        data["bbox"] = {"min_lat": 13.5, "min_lon": 77.5, "max_lat": 13.0, "max_lon": 77.6}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError):
                pk.load_pack(path)

    def test_bad_place_rejected(self):
        data = _good_pack()
        data["places"] = [{"name": "", "type": "park", "lat": 91.0, "lon": 0.0,
                           "osm_id": 1, "osm_type": "node"}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError):
                pk.load_pack(path)

    def test_in_bbox(self):
        bbox = _good_pack()["bbox"]
        self.assertTrue(pk.in_bbox(12.95, 77.55, bbox))
        self.assertFalse(pk.in_bbox(14.0, 77.55, bbox))


OVERPASS_FIXTURE = {
    "elements": [
        {"type": "node", "id": 1, "lat": 12.95, "lon": 77.55,
         "tags": {"name": "Central Park", "leisure": "park"}},
        {"type": "way", "id": 2,
         "center": {"lat": 12.96, "lon": 77.56},
         "tags": {"name": "Corner Cafe", "amenity": "cafe"}},
        {"type": "relation", "id": 3,
         "center": {"lat": 12.97, "lon": 77.57},
         "tags": {"name": "Old Library", "amenity": "library"}},
        # duplicates, unnamed, untagged, centerless — all dropped:
        {"type": "node", "id": 1, "lat": 12.95, "lon": 77.55,
         "tags": {"name": "Central Park Copy", "leisure": "park"}},
        {"type": "node", "id": 4, "lat": 12.95, "lon": 77.55, "tags": {"leisure": "park"}},
        {"type": "node", "id": 5, "lat": 12.95, "lon": 77.55,
         "tags": {"name": "Mystery", "shop": "books"}},
        {"type": "way", "id": 6, "tags": {"name": "No Center", "amenity": "cafe"}},
    ]
}


class TestOverpass(unittest.TestCase):
    def test_build_query_single_request(self):
        q = ov.build_query(12.9, 77.5, 13.0, 77.6)
        self.assertIn("[out:json]", q)
        self.assertIn("12.9,77.5,13.0,77.6", q)
        self.assertIn("out center", q)
        self.assertIn('["leisure"="park"]', q)
        self.assertEqual(q.count("out center"), 1)  # one request, no parallel calls

    def test_extract_places(self):
        places = ov.extract_places(OVERPASS_FIXTURE["elements"])
        self.assertEqual(
            [(p["name"], p["type"], p["osm_type"]) for p in places],
            [("Central Park", "park", "node"),
             ("Corner Cafe", "cafe", "way"),
             ("Old Library", "library", "relation")],
        )
        self.assertEqual((places[0]["lat"], places[0]["lon"]), (12.95, 77.55))

    def test_fetch_mocked(self):
        body = json.dumps(OVERPASS_FIXTURE).encode()
        fake_resp = mock.MagicMock()
        fake_resp.read.return_value = body
        fake_resp.__enter__.return_value = fake_resp
        with mock.patch("src.places.overpass.urllib.request.urlopen", return_value=fake_resp) as urlopen:
            elements = ov.fetch_elements("query")
            _, kwargs = urlopen.call_args
            req = urlopen.call_args[0][0]
        self.assertEqual(len(elements), 7)
        self.assertIn("Traily", req.get_header("User-agent"))
        self.assertEqual(kwargs.get("timeout"), ov.REQUEST_TIMEOUT_S)

    def test_fetch_http_error(self):
        import urllib.error

        err = urllib.error.HTTPError("http://x", 429, "Too Many", {}, None)
        with mock.patch("src.places.overpass.urllib.request.urlopen", side_effect=err):
            with self.assertRaises(ov.OverpassError):
                ov.fetch_elements("query")


if __name__ == "__main__":
    unittest.main()
