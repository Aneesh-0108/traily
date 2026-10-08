"""Phase 0 placeholder timing benchmark.

Establishes the per-stage timing output format required by AGENTS.md
(STT, LLM, TTS separately) so Phase 1 can fill in real measurements.
Target for the real pipeline: ~8-10 s total, question to spoken answer.

Usage: python3 scripts/timing.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PLACES_FILE = REPO_ROOT / "data" / "places.json"


def stage(name: str, func) -> tuple[float, object]:
    start = time.perf_counter()
    result = func()
    elapsed = time.perf_counter() - start
    return elapsed, result


def load_places_stub() -> list:
    """Placeholder for future context loading (nearest 5-10 items)."""
    with PLACES_FILE.open(encoding="utf-8") as f:
        return json.load(f).get("places", [])


def main() -> int:
    print("Trail Companion timing placeholder (Phase 0 — no real STT/LLM/TTS yet)")
    stt_s, _ = stage("stt", lambda: None)  # Phase 1: speech-to-text
    llm_s, places = stage("llm", load_places_stub)  # Phase 1: LLM + place context
    tts_s, _ = stage("tts", lambda: None)  # Phase 1: text-to-speech
    total_s = stt_s + llm_s + tts_s
    print(f"STT: {stt_s:.3f} s (placeholder)")
    print(f"LLM: {llm_s:.3f} s (placeholder, loaded {len(places)} fixture places)")
    print(f"TTS: {tts_s:.3f} s (placeholder)")
    print(f"Total: {total_s:.3f} s (Phase 1 target: ~8-10 s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
