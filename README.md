# Trail Companion (Traily)

Screen-off voice assistant for Hacktoberfest 2026 ("Touch Grass"):
mic → speech-to-text → small local LLM (with nearby OSM places) → text-to-speech.
Laptop demo, fixed area, English only, one question type ("what's nearby?").

> **Phase 0 status:** project foundation only. The voice pipeline
> (microphone, STT, llama.cpp, Piper, OSM processing) is **not** implemented yet.
> `src/main.py` runs and reports this explicitly.

## Constraints (from `AGENTS.md`, source of truth)

- **CPU only** (Intel Iris Xe, 16 GB RAM). **No CUDA.**
- Models (Phase 1): whisper.cpp / faster-whisper (small or base, int8),
  ≤3B GGUF via llama.cpp, Piper TTS.
- **Fully offline at runtime.** No network calls in the inference path.
  Prep/pack building may be online.
- Per-stage timings (STT, LLM, TTS) printed; total target **~8–10 s**
  from question to spoken answer.
- Keep place context to nearest 5–10 items. Simple code over frameworks.

## Project structure

```text
├── src/
│   ├── audio/
│   ├── stt/
│   ├── llm/
│   ├── tts/
│   └── main.py
├── data/
│   └── places.json      # Phase 0: 5 fictional fixture places (not real OSM data)
├── models/
│   ├── whisper/         # Phase 1 model files (git-ignored)
│   ├── llm/
│   └── piper/
├── scripts/
│   └── timing.py        # Phase 0 placeholder benchmark
├── tests/
├── .env.example
├── requirements.txt     # Phase 0: stdlib only, no third-party deps
├── README.md
└── AGENTS.md
```

## Prerequisites

- Python 3.11+ (developed/tested with the system Python 3; repo specifies
  3.11, code is 3.11-compatible stdlib only). Python 3.11 was not available
  on this machine, so validation used the installed Python 3.x.
- No third-party packages required in Phase 0.

## Commands

All commands run from the repo root, no task runner required.

### setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

(Phase 0 `requirements.txt` is intentionally dependency-free; this just
verifies the environment resolves.)

### run

```bash
python3 src/main.py
```

Runs the minimal entry point. Expected output: Phase 0 banner stating the
Phase 1 voice pipeline is not yet implemented, plus fixture place count.

### test

```bash
python3 -m unittest discover -s tests -v
```

Runs the basic foundation tests with stdlib `unittest` (no pytest needed).

### timing

```bash
python3 scripts/timing.py
```

Placeholder benchmark. Prints `STT / LLM / TTS / Total` lines (all
placeholders in Phase 0). Phase 1 will extend it with real measurements;
the total target is ~8–10 s.

## Dependencies

Phase 0 adds **zero** third-party dependencies. `requirements.txt` contains
only comments explaining this choice and listing what is deliberately
deferred (faster-whisper, llama-cpp-python, Piper, OSM libs, web
frameworks, DBs, cloud SDKs) and why (CPU-only/offline constraints,
Phase 0 needs stdlib only). Each future dependency must be justified per
`AGENTS.md` before adding.

## Notes

- Runtime inference will be **offline and CPU-only**. No runtime network
  calls, no CUDA, English only, fixed-area scope.
- `data/places.json` holds temporary fictional fixtures for the Phase 1
  prototype, not real OSM data.
- See `AGENTS.md` for the authoritative constraints, rules, and scope.
