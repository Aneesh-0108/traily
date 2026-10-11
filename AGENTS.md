# Traily — Hacktoberfest 2026: Touch Grass

A screen-off, voice-first trail companion:

Microphone → speech-to-text → local LLM with nearby OSM places → text-to-speech.

All inference and place lookup must work locally without network access at runtime.

## Constraints

- Hardware: Intel Iris Xe integrated graphics, 16 GB RAM. CPU-only; no CUDA.
- STT: `whisper.cpp` with `base.en` as the initial model. Upgrade to `small.en` only if accuracy needs it and measured latency allows. Use quantization where supported and beneficial.
- LLM: Qwen3.5-4B, GGUF Q4_K_M, via `llama.cpp`. This is the selected local model.
  - Local model path: `models/llm/Qwen3.5-4B-Q4_K_M.gguf`, configurable when necessary.
  - Disable thinking/reasoning for the runtime. The currently verified configuration uses `--reasoning off`; verify compatibility if llama.cpp changes.
  - Benchmark on the actual CPU.
  - If latency is too high, first limit prompt length and output tokens, then test a lower-bit quantization such as Q3_K_M. Consider a smaller model only if necessary.
  - Do not switch models without recording the reason and benchmark results.
- TTS: Piper, voice `en_US-lessac-medium`.
- Latency target: approximately 8–10 seconds or less from the end of the user's question to the start of spoken output.
- Prefer beginning TTS when the first complete sentence is available. First establish a reliable non-streaming baseline; streaming is an optimization, not a blocker for the initial prototype.
- Answers: at most two short sentences, natural spoken English, no Markdown.
- Runtime must not make network calls. Preparation, model downloads, OSM pack building, and Tinker training may use the network.

## Scope

- One fixed geographic area.
- English only.
- One principal question type: "What's nearby?"
- Laptop demonstration only.
- Prefer simple, modular code over frameworks.
- Do not add features beyond this scope without explicit approval.

## Local runtime

The intended pipeline is:

Microphone → Whisper → local place filtering → Qwen3.5-4B → Piper → speaker.

- Keep STT, place filtering, LLM inference, and TTS in separate functions/modules.
- Pre-filter places in code and provide only the nearest 5–10 relevant items.
- Validate place data before passing it to the LLM.
- Never trust the LLM to calculate distances or establish which places are nearby.
- The LLM must use only supplied places and must not invent names, distances, directions, or other facts.
- Handle empty transcriptions and unavailable model files gracefully.
- Do not require model downloads or network connectivity during a normal interaction.

## Fine-tuning with Tinker

- Primary training model: `Qwen/Qwen3.5-4B`.
- Verify the exact model ID and training availability on Tinker's Models & Pricing page immediately before training.
- Do not automatically fall back to Llama 3.2 3B or 1B for Tinker training; verify current availability before choosing any replacement.
- Dataset: JSONL examples containing a question, 5–10 nearby places (name, type, distance), and a short spoken target answer.
- Aim for 100–300 examples if time permits. Generate examples from realistic place data, inspect them manually, and remove invalid or duplicate examples.
- Reserve 20 held-out questions. Avoid duplicate place contexts and near-identical examples across training and evaluation.
- Compare base and fine-tuned models on the same held-out examples.
- Evaluate factual correctness, grounding, brevity, spoken style, and hallucination.
- Record quantitative results and representative outputs in the README.
- Before full training, run a small end-to-end dry run:
  train → download adapter → convert/merge → run in llama.cpp.
- Do not assume that downloading a LoRA adapter automatically produces a compatible GGUF. Verify the exact conversion and inference path first.
- Use the fine-tuned model for the outdoor demonstration only if conversion succeeds and its latency and output quality are acceptable. Otherwise, use the base model and clearly disclose this in the README.
- Record the model ID, dataset version, training configuration, and checkpoint used.
- Never commit API keys, secrets, or credentials to the repository or training dataset.

## Commands

Keep the actual, tested project commands documented in `README.md` and maintained here as implementation evolves:

- `run`: launch Traily.
- `test`: run deterministic unit tests.
- `timing`: benchmark the local pipeline.
- `build-pack`: prepare the offline OSM place pack.
- `train`: execute the Tinker fine-tuning workflow.
- `evaluate`: compare base and fine-tuned outputs.

Do not document a command as working until it has been tested.

## Rules

- Print separate timings for:
  - audio recording;
  - STT;
  - LLM first-token latency, where measurable;
  - LLM total generation;
  - TTS synthesis;
  - start of spoken playback;
  - total time from the end of the question to the start of spoken output.
- Distinguish model-loading time from warm inference whenever possible.
- Use a monotonic high-resolution timer, such as `time.perf_counter()`.
- Explain any new dependency before adding it.
- Never commit secrets, API keys, personal location data, or model binaries.
- Keep each pipeline stage separate so it can be timed, tested, and replaced independently.
- Log meaningful failures, including mishearings, hallucinations, incorrect distances, and slow responses, in `notes/failures.md`.
- Record actual model versions, quantization, commands, timings, and evaluation results.
- Prefer measured results over assumptions. Never claim a latency or quality improvement without evidence.

## Completion priorities

1. Get the local microphone-to-speaker pipeline working with a hard-coded place list.
2. Record actual latency and validate responses.
3. Replace fictional places with a real, prepared OSM pack for the fixed area.
4. Test outdoors and record failures, photos, and observations.
5. Run the Tinker dry run and full fine-tune only if the conversion and evaluation path is viable within the remaining time.
6. Write the Hacktoberfest post with actual measurements, the role of open-source/open-weight technology, and honest limitations.

The outdoor test and submission write-up take priority over optional integrations or optimizations.

## Definition of done

- A working screen-off voice interaction on the laptop.
- No runtime network dependency.
- Answers grounded in the local place pack.
- Measured latency and documented model choices.
- A real outdoor test with findings and failures recorded.
- Base-versus-fine-tuned evaluation documented if fine-tuning succeeds.
- A final write-up that accurately represents what was built and tested.