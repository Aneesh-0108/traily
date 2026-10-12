# Traily failure log (per AGENTS.md: record mishearings, hallucinations, slow responses)

## 2026-10-12 — Tinker future/checkpoint fix (no new billing)

- **Failure:** tiny run died at `save_state(...).path` with
  `AttributeError: 'AwaitableConcurrentFuture' object has no attribute 'path'`.
  Every training call returns an APIFuture; the script never resolved any.
- **Fix:** each `forward_backward`/`optim_step` future is now resolved inline
  via documented `.result()` (preserves order, surfaces per-step errors);
  export uses `save_weights_for_sampler()` (resolved, 600 s bound, path
  validated non-empty) — NOT `save_state()`, which persists resume-state
  while `weights.download`/`build_lora_adapter` need the
  `tinker://.../sampler_weights/...` path. Resume-state saving dropped
  from the 2-step smoke chain (nothing worth resuming). Remote failures
  raise `TinkerRunError` → exit 5.
- **Spend analysis (honest, no zero-claim):** the failed attempt executed
  estimate/subset/preflight/key-check, then submitted 2× forward_backward
  + 2× optim_step futures (unresolved) before crashing on the save line.
  No checkpoint was saved, nothing was exported/converted. Whether the
  server completed (and billed) the submitted steps is UNKNOWN from here —
  verify via `tinker billing` or the console with the key before the next
  attempt. Upper bound if both steps ran: ~1.3k tokens ≈ $0.001.

## 2026-10-12 — Dry-run env separation fix (no billing)

- **Failure:** first dry-run attempt died with
  `ModuleNotFoundError: No module named 'torch'` at
  `from tinker_cookbook import weights` — line 160, BEFORE `ServiceClient()`
  (line 167) and all training calls, so no session, no tokens, no billing.
- **Root cause:** batch rendering (`FromConversationFileBuilder`) and PEFT
  export (`weights.*`) both ran in the lightweight train venv, whose
  `--no-deps` cookbook install has no torch (full cookbook needs ~1 GB disk,
  quota-blocked there). Torch was only ever in the convert venv.
- **Fix:** train interpreter now imports `tinker` ONLY; new
  `scripts/tinker_batch_helper.py` (`dump-data`, `export`) runs under
  `--convert-python`, passing plain-JSON Datum payloads rebuilt with
  torch-free `tinker.types` (round-trip verified live in both venvs).
  Convert env gained only justified packages (tinker, cookbook --no-deps,
  blobfile/chz/datasets/pillow); train env and runtime `.venv` unchanged.
  Preflight now verifies per-env imports + helper presence before billing.
- **Credit untouched.** No key configured; 120/120 tests pass.

## 2026-10-11 — Phase 3B.1: LoRA converter prepared + verified (no training)

- **Required commit pinned, not assumed.** Installed llama-cli is commit
  5266f24da7 (2026-09-04); the Qwen3.5 LoRA V-head fix is upstream commit
  **b016f461be57** (2026-09-30, `conversion/qwen.py`, +15 lines) — the
  installed toolchain predates it, so the converter comes from upstream
  at exactly that commit (`tools/lora-converter/`, git-ignored,
  reproducible via `tools/fetch-converter.sh`).
- **Dependency footprint (isolated `/tmp/traily-convert-venv`, ~1.2 GB).**
  torch CPU + safetensors + transformers + numpy + huggingface_hub from
  pip; gguf MUST come from the repo checkout (PyPI max 0.19.0 lacks
  MODEL_ARCH.DFLASH and fails to import the pinned converter).
  Runtime `.venv` and system packages untouched.
- **Synthetic-adapter proof (real Qwen3.5-4B dims/hparams).** V-head
  out_proj/in_proj_qkv + classic o_proj + MLP tensors convert cleanly;
  output arch `qwen35` matches base; installed `llama-cli --lora` loads
  (rc=0) and generates. Found en route: full-attention o_proj input dim
  is 4096, not 2560 (hybrid quirk) — the initial load failure was my
  synthetic shape, correctly rejected by the runtime.
- **Residual uncertainty (honest).** Random weights prove format and
  loadability, not numerical correctness — the real trained adapter is
  the remaining proof. Preflight (sources + fix marker + interpreter
  deps) now runs in `tinker_dryrun.py` BEFORE any billing (exit 4).
- **Credit untouched.** No key, no job, no adapter download (none exists).

## 2026-10-11 — Dry-run review: minimal smoke test (not executed)

- **Shrunk to the floor.** No documented minimum batch/steps exists, so the
  smoke test is 2 examples, batch 2, 2 steps, rank 4, seed 7 (≥1 optimizer
  update + exportable checkpoint, with one spare step): ~1.3k tokens ≈
  **$0.001** vs the prior 8-ex/40-step ≈$0.08. Full 36-ex run ≈ $0.03.
- **SDK verified in an isolated venv** (`tinker` 0.33.1 installs cleanly;
  runtime `.venv` untouched). Signature introspection corrected three real
  bugs in the prepared script: `save_checkpoint` → `save_state` (+`.path`),
  `optim_step(lr=)` → `optim_step(adam_params=AdamParams(...))`,
  `train_ds.sample()` → `get_batch(i)`; `seed` now passed to
  `create_lora_training_client`. Full cookbook needs torch (~1 GB,
  quota-blocked here).
- **Conversion, confirmed vs unverified.** CONFIRMED: cookbook
  `weights.download`/`build_lora_adapter`/`build_hf_model`,
  `llama-cli --lora` flag, `llama-quantize`, `TRAILY_LLM_MODEL` override.
  UNVERIFIED: `convert_lora_to_gguf.py` (exists upstream, not installed),
  `convert_hf_to_gguf.py` (absent), actual adapter load. Script stops
  cleanly (exit 4) if the converter is missing — after train+export.
- **Credit untouched.** No key configured here; script refuses without
  `TINKER_API_KEY`; tinker SDK absent from runtime env. Eval file intact
  (20 held-out); all 109 tests pass.

## 2026-10-11 — Phase 3B: dry run prepared, not executed

- **Verified today:** `Qwen/Qwen3.5-4B` on Models & Pricing (Train $0.737/M);
  no fallback model. Dataset re-validated (36/20/20, all targets complete,
  zero overlap). Tokens measured via `llama-tokenize` (~294/ex).
  Estimates: dry run ≈ $0.08, full 36-ex run ≈ $0.03. Credit untouched.
- **Blocker before any paid run:** NO local conversion path verified.
  `llama-cli --lora` flag exists and `llama-quantize` is installed, but
  PEFT→GGUF-LoRA and `convert_hf_to_gguf.py` are absent (no transformers/
  peft/torch/tinker SDK locally) and no checkpoint exists to test against.
  Export script prepared with every unverified step marked; guarded so it
  cannot run or bill without explicit approval + TINKER_API_KEY in env.

## 2026-10-11 — Phase 3A: expanded pack + dataset + baseline

- **Overpass fetch SUCCEEDED (one attempt).** Bbox 74.88,12.89,74.92,12.93
  → 12 real places, all unique/coordinated (7 worship, 2 museum, 1 cafe,
  1 restaurant, 1 attraction). Saved to
  `data/packs/sjec-vamanjoor-expanded.json` (git-ignored); small pack kept.
- **Dataset honesty limit.** 12 clustered places yield ~23 distinct
  unordered nearby-5 sets (400-pt probe: 99 ordered). Strict
  group-disjoint split gives **36 train + 20 eval** — the maximum honest
  size. NOT claimed ready for a 100–300 LoRA experiment; needs 3–5× larger
  area (follow-up fetch; this task's single attempt is used).
- **Base-model baseline (20/20 held-out, local, ~20.8 s/ex): 12/20
  complete, 15/20 grounded.** Failure modes: comma-dropped names (5×
  "Kordappu Sthana, Parari"), one generic-noun answer, one truncation.
  Eval/grader heuristic fixed twice (sentence-word false positives:
  Nearby/Around/Closest/Other) — remaining flags are real mutations.
- **Generator rounding fix.** Query coords now rounded BEFORE lookup so
  stored coords exactly reproduce stored distances (caught by test:
  536 vs 535 m off-by-one at a .5 boundary).

## 2026-10-11 — All-five-places answers (+ heard-vs-shown investigation)

- **Prompt change.** "Mention the nearest two or three places" became
  "Mention EVERY supplied place by name with its distance in meters, each
  place once." `-n` 48 -> 96 (five names+distances need ~60 tokens; 48 cut
  answers mid-sentence, e.g. "...at 1"). Ctx/batch untouched; still
  `-ngl 0 --reasoning off`, still configurable via TRAILY_LLM_MAX_TOKENS.
- **Completeness gate (no retry loop).** `ensure_complete_answer` checks
  every supplied name (case-insensitive) + exact distance (word boundary)
  in the LLM text; on any omission the deterministic `fallback_answer`
  ("Nearby: A at X meters, ..., and E at Y meters.") speaks instead.
  Verified live: one run needed the fallback, Piper rendered all five
  (17.3 s audio, frame-count validated).
- **Heard-fewer-than-shown: likely causes + fixes.** (1) `play_wav` used
  `check=False` and IGNORED the exit code while the report said
  "playback started: yes" — a failed/cut Bluetooth playback looked
  successful; now non-zero exits raise PlaybackError. (2) Same-string
  guarantee confirmed: the printed `Traily:` text IS the Piper stdin
  (unit-tested); `synthesize` now also rejects empty/unreadable WAVs by
  frame count. Bluetooth A2DP start-up clipping cannot be verified
  headlessly — re-check on the laptop with audible playback.
- **Cost.** LLM ~14-14.6 s here (vs ~11 s at -n 48; the cap only costs
  time for tokens actually generated), TTS ~2.2 s, post-recording
  ~16.3-16.8 s. Target still not met; no claim made.

## 2026-10-11 — LLM echo-anchor regression (fixed)

- **Symptom.** Pack-mode run (SJEC pack, 5 OSM places) failed with
  "LLM output missing prompt echo; cannot isolate the answer" although
  the CLI exited 0 with a good answer in stdout.
- **Root cause (inspected real stdout, not guessed).** simple-io does not
  just truncate long echoed lines mid-line — for longer prompts it drops
  the echo TAIL entirely: the output showed everything up to
  `At most t ... (truncated)` and the final `User:` line never appeared.
  The adapter anchored on the LAST prompt line, so no anchor could match.
- **Fix (src/llm/answer.py).** Anchor on the echo BLOCK instead: open at
  the first `> ` line (must match the first prompt line), consume echo
  lines while they match successive prompt lines (exact or
  `... (truncated)`-head match), answer starts at the first non-match.
  Still fail-closed (no echo block / contamination / empty / oversize /
  timeout / bad exit -> LLMError, never to Piper). Flags unchanged.
- **Validated.** 73/73 tests (incl. new Phase 2 tail-truncation
  regression test with SJEC-shaped places); failing command now returns
  "Saint Pio Chapel at 64 meters and The Lassi Corner at 120 meters.
  Bethany Provincialate, Vamanjoor is also close at 184 meters."
  (LLM 11.37 s, TTS 1.78 s here).

## 2026-10-11 — Phase 2 validation

- **Public Overpass unusable during validation.** `overpass-api.de` 504 on
  two full-pack queries AND a trivial one-node probe; `overpass.kumi.systems`
  500. Stopped after 4 attempts (no aggressive retries). Pack builder
  validated end-to-end via `--from-file` with realistic fixture data
  (8 elements -> 5 places; dupe/unnamed/centerless correctly dropped).
  Live `build-pack` fetch still untested — retry on a good connection.
- **No live offline location source on test machines.** No GPSD daemon
  (127.0.0.1:2947 refused), no ModemManager process, no /dev/ttyUSB*,
  /dev/ttyACM*, /dev/gps* devices, no lsusb output available. Manual
  `--lat/--lon` mode works; `--location-source gpsd` correctly reports
  unavailable. Laptop re-check needed before claiming live location.
- **Pack-mode answer quality good.** Fixture pack run: "The nearest places
  are Cubbon Park at 16 meters and Park Cafe at 78 meters..." — exact
  names/distances, no invented directions (haversine cross-checked:
  15.5 m independently).

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
