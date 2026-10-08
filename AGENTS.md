# Traily(Hacktoberfest 2026, Touch Grass)
Screen-off voice assistant: mic -> speech-to-text -> small local LLM
(with nearby OSM places) -> text-to-speech. Fully offline at runtime.

## Constraints
- CPU only (Intel Iris Xe, 16 GB RAM). No CUDA.
- Models: whisper.cpp / faster-whisper (small or base, int8),
  3B or smaller GGUF via llama.cpp, Piper TTS.
- Target under ~8-10 s from question to spoken answer.
- No network calls at runtime. Prep (pack building) can be online.

## Scope
Fixed area, English only, one question type ("what's nearby?").
Laptop demo only. Prefer simple code over frameworks.

## Commands
(run: ..., test: ..., timing: ...)

## Rules
- Print per-stage timings (STT, LLM, TTS).
- Keep the place context to the nearest 5-10 items.
- Explain any new dependency before adding it.