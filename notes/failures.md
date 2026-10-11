# Traily failure log (per AGENTS.md: record mishearings, hallucinations, slow responses)

## 2026-10-11 — LLM output contamination (fixed)

- **Symptom (laptop).** `Traily:` answer contained llama.cpp chrome
  (Loading model, build info, command menu, prompt echo); Piper spoke it
  all (TTS 4.06 s, post-recording 14.66 s).
- **Root causes found by inspecting real stdout/stderr.**
  (1) With piped stdout, llama-cli prints banner+menu+echo+answer+perf
  ALL to stdout (`--log-disable` changes nothing; stderr has only GPU
  warnings). (2) The old parser defaulted to `start = 0` when the echo
  anchor was missing, returning the entire banner as the answer.
  (3) simple-io TRUNCATES long echoed lines
  (`User: What are the nearby  ... (truncated)`), so exact-match
  anchoring fails on real prompts. (4) Dropping `-st` is NOT an option:
  without it the CLI loops `> ` prompts forever (produced a 6.4 GB log).
- **Fix (src/llm/answer.py, no model/config change).** Keep `-st` and all
  known-good flags; anchor on the echoed last prompt line with exact OR
  24-char-prefix match; stop at perf/`Exiting...`/`> `/blank lines; strip
  think tags; then VALIDATE (reject CLI markers, prompt bullets, >1500
  chars, empty) and raise LLMError instead of speaking logs. Plus a 300 s
  subprocess timeout. Verified: 39/39 unit tests, clean text-mode runs
  (LLM 9.8–10.2 s, TTS ~1.8–2.0 s on short answers, post-recording
  ~11.8–12.1 s here).
- **New observation (not fixed here).** One `-n 48` answer ended without
  terminal punctuation ("...600 meters away"), i.e. likely hit the token
  cap mid-sentence. Acceptable for TTS; revisit if frequent.

## 2026-10-11 — Latency fix (same dev container)

- **Root cause of slow LLM found: config mismatch, not the model.**
  The app omitted `-c/-b/-ub`, leaving llama.cpp defaults (model-sized
  context, batch 2048/ubatch 512). Aligning with the known-good
  standalone flags (`-c 2048 -b 128 -ub 64`, `-n 80` -> `-n 48`) cut
  in-app LLM time from 66–99 s to **9.5–10.6 s** for ~40-token answers.
  No model/quantization change (per AGENTS.md ordering: prompt + output
  tokens first).
- **8–10 s target still NOT met in this container.** Post-recording
  latency 11.3–12.5 s (LLM ~10 s + Piper ~1.8 s). No persistent server
  introduced (out of scope for this task). Next options, separately:
  (a) re-measure on the laptop (prior app result there: 17.71 s LLM with
  old flags — expect improvement with new flags); (b) warm reuse / server;
  (c) shorter answers. No claim made without measurements.
- **Brevity vs grounding tension at -n 48 (observed, mitigated).**
  First `-n 48` runs dropped either place names or distances ("a temple",
  "a park"; "listed with their exact distances" without giving them).
  Fixed by instructing "Mention the nearest two or three places by name
  with their distances" — verified run: "The nearest places are Example
  Temple at 300 meters and Green Park at 450 meters. You could also reach
  Old Library at 600 meters..." All 5 places are still SUPPLIED; the model
  names a grounded subset, which AGENTS.md permits.
- **Timing metric corrected.** The old `question-end -> playback start`
  figure summed stage durations INCLUDING the 8 s recording window
  (laptop report: 28.69 s total, not post-recording latency). Now the app
  records distinct perf_counter timestamps and reports post-recording
  latency (record-finish -> playback-start) plus total interaction time.
  Labelled as post-recording, not end-of-speech (no VAD).
- **Memory: no issue.** Bare aligned invocation: peak RSS 4,447,132 kB
  (~4.24 GiB), matching the prior ~4.2 GiB measurement. Comfortable in
  16 GB alongside Whisper/Piper (not concurrent).
- **Mic test still impossible here.** `parec` captures only noise floor
  (max sample 15/32768) — no live mic in this environment. Laptop mic
   test (Bluetooth earbuds) must confirm the full loop there.

## 2026-10-10 — Phase 1 baseline (dev container, not the demo laptop)

- **Whisper hallucinates on pure silence.** 1 s of digital silence
  transcribed as `" you"` (base.en, `--no-prints --no-timestamps`).
  Impact: a silent recording does not take the empty-transcription path;
  it proceeds as the question "you". Mitigation for demo: record in a
  quiet-but-not-silent room and confirm the "Heard: ..." echo before
  trusting the answer. No code change yet; revisit if it affects the demo.
- **LLM generation far above the 8–10 s target in this container.**
  Qwen3.5-4B-Q4_K_M, `-ngl 0 --reasoning off --jinja -st`, short grounded
  prompt, `-n 80`: 66–99 s for ~40–60 tokens (Generation: 0.6 t/s).
  The container shows <1 effective CPU core (user+sys ≈ 85 s over ~99 s
  wall on a 16-thread i5-12500H host), so this looks like environment
  throttling, not model choice. DO NOT switch models on this evidence
  (AGENTS.md). Action: re-measure on the actual demo laptop before any
  quantization/prompt-length changes.
- **No microphone or audio server in the container.** `parec` captures
  zero bytes; the app exits gracefully (`Recording failed: no audio
  captured...`, exit 2). Full mic-to-speaker run still unverified —
  must be tested on the laptop.
- **Playback unverified audibly.** `paplay` exits 0 in the container but
  there is no speaker. Must be tested on the laptop.
- **Answer style (minor).** One run produced "Just keep in mind the
  distances are 300 meters and 450 meters respectively." — grounded and
  within two sentences, but stilted. Acceptable for the baseline;
  fine-tuning experiment may address style later.
