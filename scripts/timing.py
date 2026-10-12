"""Phase 2 pipeline benchmark (non-interactive, no microphone).

Runs the full local pipeline in --text mode (places -> Qwen3.5-4B ->
Piper) and prints the same stage-timing summary as `run`. The microphone
and STT stages are skipped here; STT was measured separately at ~1.3 s
for a short question on this machine (see README).

Usage:
  python3 scripts/timing.py [--text "What's nearby?"] [--no-playback]
      [--pack data/packs/area.json --lat 12.34 --lon 56.78]
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
    parser = argparse.ArgumentParser(description="Traily Phase 2 benchmark.")
    parser.add_argument("--text", default="What's nearby?")
    parser.add_argument("--no-playback", action="store_true")
    parser.add_argument("--pack", default=None)
    parser.add_argument("--lat", type=float, default=None)
    parser.add_argument("--lon", type=float, default=None)
    parser.add_argument("--location-source", choices=("manual", "gpsd"), default="manual")
    parser.add_argument("--max-age", type=float, default=30.0)
    cli = parser.parse_args()
    pipe_args = argparse.Namespace(
        text=cli.text,
        record_seconds=0.0,
        no_playback=cli.no_playback,
        workdir=None,
        pack=cli.pack,
        lat=cli.lat,
        lon=cli.lon,
        location_source=cli.location_source,
        max_age=cli.max_age,
    )
    print(f"Benchmark question: {cli.text!r} (mic/STT skipped; STT ~1.3 s separately)")
    with tempfile.TemporaryDirectory(prefix="traily-bench-") as tmp:
        return run_pipeline(cli.text, 0.0, not cli.no_playback, Path(tmp), pipe_args)


if __name__ == "__main__":
    raise SystemExit(main())
