"""Text-to-speech via the project-local Piper (`en_US-lessac-medium`).

- Synthesis: `.venv/bin/piper` (piper-tts, pre-installed; reused, not
  reinstalled) reads text on stdin and writes a WAV file. CPU-only
  (no --cuda flag). Fully offline.
- Playback is NOT done here: the caller plays the WAV separately so
  synthesis time and playback-start time are measured independently.
"""

from __future__ import annotations

import shutil
import subprocess
import time
import wave
from pathlib import Path

from src import config


class TTSError(RuntimeError):
    """Raised for missing binaries, missing voices, or failed synthesis."""


def synthesize(
    text: str,
    out_wav: Path,
    voice_path: str = config.PIPER_VOICE,
    piper_bin: str = config.PIPER_BIN,
) -> float:
    """Synthesize `text` to `out_wav`; return synthesis seconds."""
    if not text.strip():
        raise TTSError("refusing to synthesize empty text.")
    if shutil.which(piper_bin) is None:
        raise TTSError(
            f"TTS binary '{piper_bin}' not found. "
            "Activate the project .venv or set TRAILY_PIPER_BIN."
        )
    if not Path(voice_path).is_file():
        raise TTSError(f"Piper voice not found: {voice_path}")
    start = time.perf_counter()
    proc = subprocess.run(
        [piper_bin, "-m", voice_path, "-f", str(out_wav)],
        input=text,
        stdin=None,  # `input=` supplies stdin
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        raise TTSError(f"'{piper_bin}' failed with exit code {proc.returncode}.")
    if not out_wav.is_file():
        raise TTSError(f"Piper produced no output file: {out_wav}")
    try:
        with wave.open(str(out_wav), "rb") as wav:
            frames, rate = wav.getnframes(), wav.getframerate()
    except (wave.Error, EOFError, OSError) as exc:
        raise TTSError(f"Piper output is not a readable WAV: {out_wav} ({exc})") from exc
    if frames <= 0 or rate <= 0:
        raise TTSError(f"Piper rendered empty audio: {out_wav}")
    return elapsed
