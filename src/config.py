"""Local-only configuration for Traily Phase 1.

All defaults are local file paths. No network addresses, no cloud
endpoints. Every value can be overridden with a TRAILY_* environment
variable (see .env.example).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _venv_bin(name: str) -> str:
    """Binary next to the running Python (e.g. .venv/bin/piper).

    NOTE: no os.path.realpath()/Path.resolve() here — .venv/bin/python is
    a symlink to the system interpreter, and resolving it would point at
    /usr/bin instead of the venv.
    """
    return str(Path(sys.executable).absolute().parent / name)


# Model files (verified present on the dev machine; never downloaded at runtime)
WHISPER_MODEL = _env("TRAILY_WHISPER_MODEL", str(REPO_ROOT / "models/whisper/ggml-base.en.bin"))
LLM_MODEL = _env("TRAILY_LLM_MODEL", str(REPO_ROOT / "models/llm/Qwen3.5-4B-Q4_K_M.gguf"))
PIPER_VOICE = _env("TRAILY_PIPER_VOICE", str(REPO_ROOT / "models/piper/en_US-lessac-medium.onnx"))

# Executables (installed system tools + piper from the project .venv)
WHISPER_BIN = _env("TRAILY_WHISPER_BIN", "whisper-cli")
LLAMA_BIN = _env("TRAILY_LLAMA_BIN", "llama-cli")
PIPER_BIN = _env("TRAILY_PIPER_BIN", _venv_bin("piper"))
ARECORD_BIN = _env("TRAILY_ARECORD_BIN", "parec")  # PulseAudio capture
APLAY_BIN = _env("TRAILY_APLAY_BIN", "paplay")  # PulseAudio playback

# Data and pipeline settings
PLACES_FILE = _env("TRAILY_PLACES_FILE", str(REPO_ROOT / "data/places.json"))
LANGUAGE = _env("TRAILY_LANGUAGE", "en")
RECORD_SECONDS = float(_env("TRAILY_RECORD_SECONDS", "8"))
RECORD_RATE_HZ = int(_env("TRAILY_RECORD_RATE_HZ", "16000"))
MAX_PLACES = int(_env("TRAILY_MAX_PLACES", "10"))

# Phase 2 offline pack + location (empty == unset; manual coords are TEST
# coordinates, never live location). CLI flags override these.
PACK_FILE = _env("TRAILY_PACK_FILE", "") or None
DEF_LAT = _env("TRAILY_LAT", "")
DEF_LON = _env("TRAILY_LON", "")
LOCATION_SOURCE = _env("TRAILY_LOCATION_SOURCE", "manual")
MAX_LOCATION_AGE_S = float(_env("TRAILY_MAX_LOCATION_AGE_S", "30"))

# LLM inference settings. Defaults are the known-good values from the
# standalone laptop measurement (~4.13 s, ~4.2 GiB peak RSS):
#   -ngl 0 --reasoning off --jinja -st -c 2048 -b 128 -ub 64 -n 48
# Rationale: the app previously omitted -c/-b/-ub, leaving llama.cpp
# defaults (model-sized context, batch 2048/ubatch 512), which inflated
# KV-cache allocation and prompt processing on 16 GB CPU-only RAM.
# -n 48 caps output at two short spoken sentences (AGENTS.md: limit
# output tokens before touching quantization or model choice).
# -n 96 caps output for up to 10 supplied names + distances in two short
# spoken sentences (five places need ~60 tokens; 48 cut answers off
# mid-sentence). The cap only costs time when the model uses the tokens.
LLM_CTX_SIZE = int(_env("TRAILY_LLM_CTX_SIZE", "2048"))
LLM_BATCH_SIZE = int(_env("TRAILY_LLM_BATCH_SIZE", "128"))
LLM_UBATCH_SIZE = int(_env("TRAILY_LLM_UBATCH_SIZE", "64"))
LLM_MAX_TOKENS = int(_env("TRAILY_LLM_MAX_TOKENS", "96"))
