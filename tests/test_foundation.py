"""Phase 0 foundation tests (stdlib unittest only, no extra deps)."""

import json
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestFoundation(unittest.TestCase):
    def test_project_structure_exists(self):
        for rel in [
            "src/main.py",
            "src/audio/__init__.py",
            "src/stt/__init__.py",
            "src/llm/__init__.py",
            "src/tts/__init__.py",
            "data/places.json",
            "scripts/timing.py",
            "requirements.txt",
            "README.md",
            ".env.example",
            "AGENTS.md",
        ]:
            with self.subTest(path=rel):
                self.assertTrue((REPO_ROOT / rel).is_file(), f"missing: {rel}")

    def test_places_fixture_valid(self):
        with (REPO_ROOT / "data" / "places.json").open(encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("places", data)
        self.assertEqual(len(data["places"]), 5)
        for place in data["places"]:
            self.assertIn("name", place)
            self.assertIn("type", place)
            self.assertIn("distance_m", place)

    def test_main_entry_point(self):
        import sys

        sys.path.insert(0, str(REPO_ROOT / "src"))
        try:
            import main

            self.assertEqual(main.main(), 0)
            self.assertEqual(len(main.load_places()), 5)
        finally:
            sys.path.remove(str(REPO_ROOT / "src"))


if __name__ == "__main__":
    unittest.main()
