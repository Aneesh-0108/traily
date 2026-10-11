# Trail Companion (Traily)

Screen-off voice assistant for Hacktoberfest 2026 ("Touch Grass"):
mic → speech-to-text → small local LLM (with nearby places) → text-to-speech.
Laptop demo, fixed area, English only, one question type ("what's nearby?").

> **Phase 1 status:** working local baseline with fictional places.
> `python src/main.py` runs one full interaction
> (Enter → record → Whisper → places → Qwen3.5-4B → Piper → speaker).
> OSM packs, GPS, and fine-tuning are explicitly out of scope for this phase.

## Constraints (from `AGENTS.md`, source of truth)

- **CPU only** (Intel Iris Xe, 16 GB RAM). **No CUDA** (`-ngl 0`, no `--cuda` anywhere).
- STT: `whisper.cpp` `base.en` via installed `whisper-cli`.
- LLM: Qwen3.5-4B GGUF Q4_K_M via installed `llama-cli`, **reasoning always off**
  (`--reasoning off` is hard-coded into every invocation in `src/llm/answer.py`,
  plus defensive `<think>` stripping; unit-tested).
- TTS: Piper voice `en_US-lessac-medium` via the project `.venv`.
- **Fully offline at runtime.** No network calls in `src/` or `scripts/`
  (verified by inspection); models/binaries are reused from disk, never downloaded.
- Per-stage timings printed with `time.perf_counter()`; target **~8–10 s**
  from question-end to playback-start (NOT met in the dev container — see Benchmarks).

## Project structure

```text
├── src/
│   ├── config.py        # local-only paths/settings (TRAILY_* env overrides)
│   ├── main.py          # pipeline orchestration + timing summary (--text/--no-playback)
│   ├── audio/record.py  # fixed-duration mic capture (parec + stdlib wave)
│   ├── audio/play.py    # local playback (paplay), timed separately from synthesis
│   ├── stt/transcribe.py# whisper-cli wrapper, empty-transcription handling
│   ├── places/nearby.py # fixture load/validate/sort/limit (max 10, in code)
│   ├── llm/answer.py    # prompt builder + llama-cli runner (reasoning OFF mandatory)
│   └── tts/speak.py     # piper synthesis to WAV (CPU-only)
├── data/
│   └── places.json      # Phase 1: 5 FICTIONAL fixture places (not real OSM data)
├── models/              # git-ignored binaries (reused, never committed/downloaded at runtime)
│   ├── whisper/ggml-base.en.bin
│   ├── llm/Qwen3.5-4B-Q4_K_M.gguf
│   └── piper/en_US-lessac-medium.onnx{,.json}
├── scripts/
│   └── timing.py        # non-interactive benchmark (full pipeline, mic skipped)
├── tests/               # 30 deterministic unittest tests (subprocess mocked)
├── notes/failures.md    # mishearings, hallucinations, slow responses
├── .env.example
├── requirements.txt     # one dep: piper-tts (see Dependencies)
└── AGENTS.md
```

## Prerequisites

- Python 3.11+ (validated with 3.14; code is 3.11-compatible stdlib except piper-tts).
- Installed and verified: `whisper-cli`, `llama-cli` (0.4.0-dev, supports
  `-ngl 0 --reasoning off --jinja -st`), `parec`/`paplay`, model files above.
- Nothing is downloaded or installed by the app at runtime.

## Commands

All commands run from the repo root, no task runner.

### setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Already satisfied if the `.venv` (with `piper-tts`) exists; nothing is reinstalled.

### run

```bash
.venv/bin/python src/main.py
```

Press Enter, ask your question (default 8 s recording), hear the reply.
Useful variants:

```bash
.venv/bin/python src/main.py --text "What's nearby?" --no-playback   # no mic/speaker needed
.venv/bin/python src/main.py --record-seconds 5 --workdir /tmp/traily-run
```

### test

```bash
python3 -m unittest discover -s tests -v
```

30 tests, all offline: places validation/sort/limit, prompt contents,
empty-transcription handling, invalid-place handling, `--reasoning off`
presence in every LLM argv, CLI-output parsing, mocked pipeline paths.
No model downloads or inference in tests.

### timing

```bash
.venv/bin/python scripts/timing.py --no-playback
```

Non-interactive end-to-end benchmark (default question "What's nearby?";
mic/STT skipped — STT measured separately at ~1.3 s for a short question).
Prints record / STT / LLM / TTS / post-recording / total timings.

## Dependencies

One added vs Phase 0, justified per AGENTS.md (see `requirements.txt` header
for the full rationale):

- `piper-tts==1.8.0` — the AGENTS.md-selected TTS engine; used only via its
  `piper` binary (CPU-only, offline, local voice file). Already present in
  `.venv`; recorded here for reproducibility.

Deliberately **not** added: Python bindings for Whisper/llama (pre-installed
system CLIs used via subprocess instead), audio libraries (`parec`/`paplay` +
stdlib `wave` suffice), OSM/GPS/web/cloud/fine-tuning packages (out of scope).

## Benchmarks

Laptop (reported): STT ~0.92 s ("What is nearby?"), Piper ~2.05 s, LLM
17.71 s with the app's OLD flags; standalone aligned command
(`-ngl 0 --reasoning off --jinja -st -c 2048 -b 128 -ub 64 -n 48`) ~4.13 s
at ~4.2 GiB peak RSS.

Dev container (measured 2026-10-11, text mode, after the latency fix):

| Stage | Measured |
|---|---|
| STT (whisper base.en, ~2 s speech sample) | **~1.3 s** |
| LLM (Qwen3.5-4B-Q4_K_M, aligned flags above, ~40-token answer) | **9.8–10.2 s** (was 66–99 s before the latency fix) |
| TTS synthesis (Piper, short clean reply) | **~1.8–2.0 s** (4.06 s on laptop was speaking CLI contamination) |
| Post-recording latency (record-finish → playback-start) | **~11.8–12.1 s** — target NOT met here |
| Bare LLM peak RSS (`/usr/bin/time -v`) | **~4.24 GiB** — no memory issue on 16 GB |

The fix was a configuration mismatch, not the model: the app had omitted
`-c/-b/-ub` (llama.cpp defaults: model-sized context, batch 2048), and used
`-n 80`. Previously the old `question-end → playback start` figure also
wrongly INCLUDED the 8 s recording window (e.g. laptop 28.69 s) — the app now
reports post-recording and total interaction separately. Per AGENTS.md no model
switch or quantization change was made; **re-run `timing` on the laptop** to
confirm (expect near the ~4 s standalone figure there, plus ~1 s STT + ~2 s TTS).

Sample grounded answer after the fix (no think trace, 2 sentences):
> "The nearest places are Example Temple at 300 meters and Green Park at 450 meters.
> You could also reach Old Library at 600 meters if you need something else."

LLM output handling (2026-10-11 fix): `llama-cli -st` prints banner, model
metadata, command menu, prompt echo, answer, and perf lines ALL to stdout
(`--log-disable` does not suppress them; dropping `-st` hangs forever), so
`src/llm/answer.py` extracts the answer between the echoed prompt line and
the perf/`Exiting...` terminators and VALIDATES it (rejects CLI markers,
echoed prompt bullets, empty/oversize text) — raising `LLMError` instead of
sending raw CLI logs to Piper. stdout/stderr are captured separately and
never concatenated.

LLM inference settings (`src/config.py`, overridable via `TRAILY_LLM_*`):
`CTX_SIZE=2048`, `BATCH_SIZE=128`, `UBATCH_SIZE=64`, `MAX_TOKENS=48`.

## Known limitations

- Full microphone-to-speaker run unverified in this container (no mic/audio
  server; mic path exits gracefully with exit 2). Must test on the laptop.
- Audible playback unverified headless (`paplay` exits 0, no speaker to confirm).
- `llama-cli` cold-starts per question: model-load time is included in the LLM
  figure (no load/inference split exposed); first-token latency reported as N/A.
- Whisper hallucinates short words (e.g. `" you"`) on pure silence — see
  `notes/failures.md`.
- `data/places.json` is fictional; real OSM pack is a later phase.
- Tinker credit untouched; no training/fine-tuning in this phase.
