"""Trail Companion Phase 0 entry point.

Minimal placeholder only. The Phase 1 voice pipeline
(mic -> STT -> LLM -> TTS) is NOT implemented yet.
"""

from __future__ import annotations

import json
from pathlib import Path

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "places.json"


def load_places(path: Path = DATA_FILE) -> list[dict]:
    """Load the temporary Phase 0 places fixture."""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data.get("places", [])


def main() -> int:
    print("Trail Companion (Phase 0 foundation)")
    print("Phase 1 voice pipeline (mic -> STT -> LLM -> TTS) is not yet implemented.")
    try:
        places = load_places()
    except FileNotFoundError:
        print("No places fixture found at data/places.json (expected in Phase 1).")
        return 0
    print(f"Loaded {len(places)} fixture place(s) from data/places.json (temporary test data).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
