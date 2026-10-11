"""Microphone recording to 16 kHz mono 16-bit PCM WAV.

Uses the PulseAudio `parec` capture utility (already present, no new
Python dependency) plus the standard-library `wave` module to write the
WAV header. No wake-word, VAD, or continuous listening in Phase 1:
one fixed-duration recording per interaction.
"""

from __future__ import annotations

import shutil
import subprocess
import time
import wave
from pathlib import Path

from src import config


class AudioError(RuntimeError):
    """Raised when recording fails (missing tool, no mic, bad output)."""


def record_question(
    out_wav: Path,
    seconds: float = config.RECORD_SECONDS,
    rate_hz: int = config.RECORD_RATE_HZ,
    record_bin: str = config.ARECORD_BIN,
) -> float:
    """Record `seconds` of audio and write a 16 kHz mono 16-bit WAV.

    `parec` captures until killed, so we stop it after the fixed duration
    and wrap the raw PCM in a WAV header with stdlib `wave`.
    Returns the wall-clock recording duration in seconds.
    Raises AudioError on missing tools, failed capture, or empty audio.
    """
    if shutil.which(record_bin) is None:
        raise AudioError(
            f"recorder '{record_bin}' not found. "
            "Install PulseAudio client tools or set TRAILY_ARECORD_BIN."
        )
    raw_path = out_wav.with_suffix(".raw")
    start = time.perf_counter()
    try:
        proc = subprocess.Popen(
            [
                record_bin,
                f"--rate={rate_hz}",
                "--format=s16le",
                "--channels=1",
                "--record",
                "--device=@DEFAULT_SOURCE@",
                str(raw_path),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError as exc:
        raise AudioError(f"recorder '{record_bin}' could not start: {exc}") from exc
    try:
        time.sleep(seconds)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
    elapsed = time.perf_counter() - start

    expected_frames = int(seconds * rate_hz)
    try:
        raw = raw_path.read_bytes()
    except OSError:
        raw = b""
    finally:
        raw_path.unlink(missing_ok=True)

    if not raw.strip(b"\x00"):
        raise AudioError(
            "no audio captured (is a microphone available?). "
            "Tip: `python3 src/main.py --text \"What's nearby?\"` skips the mic."
        )
    # Trim (or zero-pad, if the mic vanished mid-recording) to the fixed size.
    frames = raw[: expected_frames * 2].ljust(expected_frames * 2, b"\x00")
    with wave.open(str(out_wav), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate_hz)
        wav.writeframes(frames)
    return elapsed
