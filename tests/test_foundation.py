"""Phase 0/1 foundation tests (stdlib unittest only, no extra deps)."""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


class TestFoundation(unittest.TestCase):
    def test_project_structure_exists(self):
        for rel in [
            "src/main.py",
            "src/config.py",
            "src/audio/__init__.py",
            "src/audio/record.py",
            "src/audio/play.py",
            "src/stt/__init__.py",
            "src/stt/transcribe.py",
            "src/places/__init__.py",
            "src/places/nearby.py",
            "src/llm/__init__.py",
            "src/llm/answer.py",
            "src/tts/__init__.py",
            "src/tts/speak.py",
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
        from src.places.nearby import load_places

        places = load_places(REPO_ROOT / "data" / "places.json")
        self.assertEqual(len(places), 5)
        for place in places:
            self.assertIn("name", place)
            self.assertIn("type", place)
            self.assertIn("distance_m", place)

    def test_cli_parses_text_mode(self):
        from src.main import parse_args

        args = parse_args(["--text", "What's nearby?", "--no-playback"])
        self.assertEqual(args.text, "What's nearby?")
        self.assertTrue(args.no_playback)


if __name__ == "__main__":
    unittest.main()
