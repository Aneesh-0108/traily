"""Local audio playback of a WAV file via `paplay` (PulseAudio).

Kept separate from synthesis so TTS time and playback-start time can be
measured independently, as required by AGENTS.md.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from src import config


class PlaybackError(RuntimeError):
    """Raised when local playback fails."""


def play_wav(path: Path, play_bin: str = config.APLAY_BIN) -> None:
    """Play `path` on the local speaker. Blocks until playback finishes."""
    if shutil.which(play_bin) is None:
        raise PlaybackError(
            f"player '{play_bin}' not found. "
            "Install PulseAudio client tools or set TRAILY_APLAY_BIN."
        )
    try:
        subprocess.run(
            [play_bin, str(path)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError as exc:
        raise PlaybackError(f"player '{play_bin}' failed: {exc}") from exc
