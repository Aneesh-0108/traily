"""Speech-to-text via the installed `whisper-cli` and local `base.en` model.

Verified CLI behaviour (whisper.cpp, `--help` output):
- `whisper-cli -m MODEL -f WAV -l en --no-prints --no-timestamps`
  prints only the transcription text to stdout.
- No network access; model file is read from disk.

Empty/whitespace-only output means "nothing recognized" and is reported
to the caller (never passed to the LLM as a question).
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

from src import config


class STTError(RuntimeError):
    """Raised for missing executables, missing models, or failed runs."""


def is_empty_transcription(text: str) -> bool:
    """True when there is nothing usable to send to the LLM."""
    return not text.strip()


def transcribe(
    wav_path: Path,
    model_path: str = config.WHISPER_MODEL,
    whisper_bin: str = config.WHISPER_BIN,
    language: str = config.LANGUAGE,
) -> tuple[str, float]:
    """Transcribe `wav_path`; return (text, elapsed_seconds).

    Raises STTError if the binary/model/audio is missing or whisper fails.
    Returns "" text (not an exception) when audio contains no speech, so
    the caller can exit gracefully with a helpful message.
    """
    if shutil.which(whisper_bin) is None:
        raise STTError(
            f"STT binary '{whisper_bin}' not found. Set TRAILY_WHISPER_BIN."
        )
    if not Path(model_path).is_file():
        raise STTError(f"STT model not found: {model_path}")
    if not Path(wav_path).is_file():
        raise STTError(f"recording not found: {wav_path}")

    start = time.perf_counter()
    proc = subprocess.run(
        [
            whisper_bin,
            "-m", model_path,
            "-f", str(wav_path),
            "-l", language,
            "--no-prints",
            "--no-timestamps",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        raise STTError(f"'{whisper_bin}' failed with exit code {proc.returncode}.")
    return proc.stdout.strip(), elapsed
