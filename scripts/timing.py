"""Phase 1 pipeline benchmark (non-interactive, no microphone).

Runs the full local pipeline in --text mode (places -> Qwen3.5-4B ->
Piper) and prints the same stage-timing summary as `run`. The microphone
and STT stages are skipped here; STT was measured separately at ~1.3 s
for a short question on this machine (see README).

Usage:
  python3 scripts/timing.py [--text "What's nearby?"] [--no-playback]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.main import run_pipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Traily Phase 1 benchmark.")
    parser.add_argument("--text", default="What's nearby?")
    parser.add_argument("--no-playback", action="store_true")
    args = parser.parse_args()
    print(f"Benchmark question: {args.text!r} (mic/STT skipped; STT ~1.3 s separately)")
    with tempfile.TemporaryDirectory(prefix="traily-bench-") as tmp:
        return run_pipeline(args.text, 0.0, not args.no_playback, Path(tmp))


if __name__ == "__main__":
    raise SystemExit(main())
