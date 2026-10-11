"""Tests for place loading, validation, sorting, and limiting.

Deterministic: uses temp files and the real fixture. No models, no mic.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.places.nearby import is_valid_place, load_places, nearest_places  # noqa: E402


def _write_places(payload: dict, path: Path) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestPlaces(unittest.TestCase):
    def test_fixture_loads_sorted(self):
        places = load_places(REPO_ROOT / "data" / "places.json")
        self.assertEqual(len(places), 5)
        distances = [p["distance_m"] for p in places]
        self.assertEqual(distances, sorted(distances))

    def test_nearest_limit(self):
        places = [{"name": f"P{i}", "type": "park", "distance_m": i} for i in range(20)]
        self.assertEqual(len(nearest_places(places, 10)), 10)
        self.assertEqual(nearest_places(places, 10)[-1]["distance_m"], 9)
        self.assertEqual(nearest_places(places, 0), [])

    def test_invalid_entries_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_places(
                {
                    "places": [
                        {"name": "Good", "type": "park", "distance_m": 100},
                        {"name": "", "type": "park", "distance_m": 50},  # empty name
                        {"name": "NoType", "distance_m": 60},  # missing type
                        {"name": "Neg", "type": "cafe", "distance_m": -5},  # negative
                        {"name": "Str", "type": "cafe", "distance_m": "far"},  # wrong type
                        "not-a-dict",
                    ]
                },
                Path(tmp) / "places.json",
            )
            places = load_places(path)
        self.assertEqual([p["name"] for p in places], ["Good"])

    def test_missing_file(self):
        with self.assertRaises(ValueError):
            load_places(REPO_ROOT / "data" / "does-not-exist.json")

    def test_bad_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "places.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_places(path)

    def test_no_valid_places(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_places({"places": [{"name": "", "type": "", "distance_m": -1}]}, Path(tmp) / "p.json")
            with self.assertRaises(ValueError):
                load_places(path)

    def test_missing_places_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_places({"nearby": []}, Path(tmp) / "p.json")
            with self.assertRaises(ValueError):
                load_places(path)

    def test_is_valid_place(self):
        self.assertTrue(is_valid_place({"name": "A", "type": "park", "distance_m": 0}))
        self.assertTrue(is_valid_place({"name": "A", "type": "park", "distance_m": 12.5}))
        self.assertFalse(is_valid_place({"name": "A", "type": "park", "distance_m": True}))
        self.assertFalse(is_valid_place(None))


if __name__ == "__main__":
    unittest.main()
