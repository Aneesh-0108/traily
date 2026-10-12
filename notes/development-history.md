# Traily development history (condensed build diary)

The main README describes the working release. This note preserves how it
was built, phase by phase. Failure-level detail lives in `failures.md`.

## Phase 0 — Project foundation
Repo skeleton (`src/`, `data/`, `models/`, `scripts/`, `tests/`), stdlib-only
`requirements.txt`, `run`/`test`/`timing` commands, 5-place fictional
fixture, placeholder timing script. No pipeline yet.

## Phase 1 — Local voice pipeline
Mic (`parec`) → `whisper-cli` (`base.en`) → fixture places → Qwen3.5-4B via
`llama-cli` (`-ngl 0 --reasoning off --jinja -st`) → Piper
(`en_US-lessac-medium`) → `paplay`. Per-stage `perf_counter` timings.
Key fixes along the way: venv-symlink path bug, llama-cli chat-mode output
parsing (banner/menu/echo stripped, fail-closed `LLMError`), latency
alignment (`-c 2048 -b 128 -ub 64`, `-n` 48→96), prompt-echo tail
truncation, five-place completeness gate with deterministic fallback,
WAV frame validation, playback exit-code checks.

## Phase 2 — Offline OSM packs + location
`scripts/build_pack.py` (one Overpass request per explicit bbox, validate +
dedupe, `--from-file` rebuilds), pack schema v1 under `data/packs/`,
haversine `nearest_in_pack`, manual `--lat/--lon` test coordinates +
localhost-only GPSD client (receiver never found; live GPS unverified).
Real packs built: SJEC Campus Area (5 places), SJEC+Vamanjoor expanded
(12 places). Runtime lookup proven socket-free by tests.

## Phase 3A — Fine-tuning dataset prep (no training)
Deterministic generator (`scripts/prepare_finetune_data.py`, seed 7):
36 train + 20 eval JSONL (`messages` chat format), group-disjoint split,
manifest. Base-model eval harness (`scripts/evaluate_baseline.py`):
12/20 complete, 15/20 grounded on held-out questions. Conclusion recorded
at the time: 36 examples cannot support a 100–300-example LoRA experiment.

## Phase 3B — Tinker dry-run prep (no paid runs in this repo)
Verified `Qwen/Qwen3.5-4B` on Models & Pricing; prepared minimal smoke
config (2 ex, rank 4, 2 steps, ≈$0.001 est.); fixed SDK call shapes
(`save_weights_for_sampler`, resolved futures) and train/convert env
separation; pinned LoRA converter (llama.cpp `b016f461be57`, Qwen3.5
V-head fix) with a synthetic-adapter load test. Then **scope changed**:
fine-tuning continues in a separate repository. The `tinker_*` scripts,
`src/finetune/`, `tools/lora-converter/`, `data/finetune/`, and
`local/traily-smoke/` remain in the tree, ignored or untracked, and are
NOT needed to install or run Traily.

## Release refactor
README rewritten as a user-facing release doc (features actually verified,
real benchmark numbers, honest limitations); phase diary moved here.
`local/` workdirs added to `.gitignore`. No application code changed for
the refactor; full suite re-run green (126 tests).
