"""Local LLM answers via installed `llama-cli` + Qwen3.5-4B GGUF (CPU).

Verified runtime flags (llama.cpp 0.4.0-dev, `--help`):
- `-ngl 0` ............ CPU-only inference, no GPU offload.
- `--reasoning off` ... MANDATORY on every invocation: disables the
  model's thinking/reasoning trace. Never rely on prompt wording alone.
- `--jinja` ............ use the model's chat template.
- `-st` ................ single-turn, non-interactive conversation.
- `-c/-b/-ub` .......... context/batch/ubatch sizes. Known-good laptop
  values (standalone run ~4.13 s, ~4.2 GiB peak RSS): -c 2048 -b 128
  -ub 64. Previously the app omitted these, leaving llama.cpp defaults
  (model-sized context, batch 2048) — the suspected latency cause.
  All three are configurable (TRAILY_LLM_{CTX,BATCH,UBATCH}_SIZE).

Output-hygiene additions (do not change the reasoning configuration):
- `--simple-io --no-display-prompt --no-warmup` for clean subprocess use.
- `-n` caps output tokens (48: two short spoken sentences; latency control).
- Prompt passed via `-f` file; stdin is DEVNULL so the CLI always exits
  after one turn instead of waiting for chat input.

Measurement limits (stated, not fabricated):
- llama-cli runs one cold-start process per question, so model-load time
  is included in the reported LLM wall time; the CLI does not expose a
  load-vs-inference split.
- First-token latency is NOT reliably exposed by the CLI, so it is
  reported as N/A rather than estimated.

Verified stream layout (llama.cpp 0.4.0-dev, build 10809, stdout piped):
- stdout carries EVERYTHING: "Loading model..." banner, ASCII logo,
  build/model/ftype lines, the `available commands` menu, the `> `-prefixed
  prompt echo, the generated answer, a `[ Prompt: ... | Generation: ... ]`
  perf line, then `Exiting...`. Stream separation alone cannot isolate the
  answer, and dropping `-st` makes the CLI loop on `> ` prompts forever.
- stderr carries only warnings (e.g. no-GPU notices); it is ignored.
- Extraction is therefore anchor-based AND fail-closed: the answer is the
  text between the echoed last prompt line and the perf/`Exiting...`/`> `
  terminators. If the anchor is missing, or the extracted text still
  contains CLI markers, run_llm raises LLMError instead of sending raw
  CLI logs to Piper. stdout and stderr are NEVER concatenated.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from src import config

# Mandatory reasoning-disable flag. Every llama-cli invocation built by
# build_llm_argv() must contain these two tokens; tests enforce this.
REASONING_OFF_ARGV = ["--reasoning", "off"]

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def build_prompt(question: str, places: list[dict[str, Any]]) -> str:
    """Build the short grounded prompt. Deterministic; tested without models."""
    lines = [
        "Answer only the user's question, using only these nearby places:",
    ]
    for place in places:
        lines.append(f"- {place['name']} ({place['type']}, {place['distance_m']} m)")
    lines += [
        "Rules: never invent names, distances, directions, or facts. No navigation directions. "
        "Mention the nearest two or three places by name with their distances in meters. "
        "At most two short sentences, natural spoken English, no Markdown, no lists.",
        f"User: {question.strip()}",
    ]
    return "\n".join(lines)


def build_llm_argv(prompt_file: str, model_path: str = config.LLM_MODEL) -> list[str]:
    """Assemble the llama-cli command. Reasoning OFF is always included."""
    return [
        config.LLAMA_BIN,
        "-m", model_path,
        "-ngl", "0",
        *REASONING_OFF_ARGV,
        "--jinja",
        "-st",
        "-c", str(config.LLM_CTX_SIZE),
        "-b", str(config.LLM_BATCH_SIZE),
        "-ub", str(config.LLM_UBATCH_SIZE),
        "--simple-io",
        "--no-display-prompt",
        "--no-warmup",
        "-n", str(config.LLM_MAX_TOKENS),
        "-f", prompt_file,
    ]


class LLMError(RuntimeError):
    """Raised for missing binaries/models, bad exit codes, or unextractable output."""


# Markers that prove CLI chrome (not the assistant answer) leaked through.
# Checked against the extracted answer; any hit means extraction failed.
_CONTAMINATION_MARKERS = (
    "Loading model",
    "available commands",
    "Exiting...",
    "t/s",
    "modalities",
    "ftype",
)

# Generous upper bound: -n 48 caps generation at ~48 tokens (~200 chars).
# Anything far beyond that is runaway CLI output, never a spoken answer.
_MAX_ANSWER_CHARS = 1500

# Wall-clock guard: -st normally exits after one turn, but a CLI that
# loops on prompts (seen without -st) must never hang the pipeline.
_LLM_TIMEOUT_S = 300


def _find_answer_start(lines: list[str], last_prompt_line: str) -> int:
    """Index of the first line after the echoed prompt, or -1 if absent.

    The CLI truncates long echoed input lines, e.g.
    `User: What are the nearby  ... (truncated)`, so besides exact matches
    we also accept a line starting with the echo marker plus the first 24
    characters of the last prompt line. Banner/menu lines never share that
    prefix; the LAST match wins in case the model quotes the question.
    """
    start = -1
    head = last_prompt_line[:24]
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == last_prompt_line or stripped == f"> {last_prompt_line}":
            start = i + 1
        elif head and (stripped.startswith(f"> {head}") or stripped.startswith(head)):
            start = i + 1
    return start


def looks_contaminated(answer: str) -> bool:
    """True if `answer` still contains CLI chrome or echoed prompt lines."""
    if not answer:
        return True
    lowered = answer.lower()
    for marker in _CONTAMINATION_MARKERS:
        if marker.lower() in lowered:
            return True
    for line in answer.splitlines():
        stripped = line.strip()
        # New-turn prompt (runaway `> ` loop) or build-metadata line.
        if stripped == ">" or stripped.startswith("> ") or stripped.startswith("build "):
            return True
        # Echoed place-context bullet ("- Name (type, N m)") is prompt, not answer.
        if re.match(r"^-\s+.+\(.+,\s*\d+(\.\d+)?\s*m\)$", stripped):
            return True
    return False


def parse_llm_output(stdout: str, prompt: str) -> str:
    """Extract the assistant answer from `llama-cli -st` chat output.

    Fail-closed: raises LLMError if the prompt-echo anchor is missing, if
    nothing usable follows it, or if the result still looks like CLI
    output. Never returns raw CLI logs for TTS.
    """
    lines = stdout.splitlines()
    last_prompt_line = prompt.strip().splitlines()[-1].strip() if prompt.strip() else ""
    start = _find_answer_start(lines, last_prompt_line) if last_prompt_line else -1
    if start < 0:
        raise LLMError("LLM output missing prompt echo; cannot isolate the answer.")
    answer_lines: list[str] = []
    for line in lines[start:]:
        stripped = line.strip()
        if stripped.startswith("[") and "t/s" in stripped:
            break
        if stripped == "Exiting...":
            break
        if stripped == ">" or stripped.startswith("> "):
            break  # next turn prompt: stop, do not absorb the loop
        if not stripped:
            if answer_lines:
                break  # blank line ends the answer block
            continue
        answer_lines.append(line)
    text = " ".join(part.strip() for part in answer_lines)
    text = _THINK_RE.sub("", text)
    text = " ".join(text.split())
    if not text:
        raise LLMError("LLM returned no usable text.")
    if len(text) > _MAX_ANSWER_CHARS:
        raise LLMError("LLM answer implausibly long; refusing to speak CLI output.")
    if looks_contaminated(text):
        raise LLMError("LLM answer contains CLI output; refusing to speak it.")
    return text


def run_llm(prompt: str, model_path: str = config.LLM_MODEL) -> tuple[str, float]:
    """Run one local inference; return (answer_text, elapsed_seconds.

    Raises LLMError on missing binary/model, bad exit code, timeout, or
    unextractable/contaminated output. stdout and stderr are captured
    separately; only stdout is parsed, stderr (warnings) is ignored.
    """
    if shutil.which(config.LLAMA_BIN) is None:
        raise LLMError(f"LLM binary '{config.LLAMA_BIN}' not found. Set TRAILY_LLAMA_BIN.")
    if not Path(model_path).is_file():
        raise LLMError(f"LLM model not found: {model_path}")
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
        tmp.write(prompt)
        prompt_file = tmp.name
    try:
        start = time.perf_counter()
        try:
            proc = subprocess.run(
                build_llm_argv(prompt_file, model_path),
                stdin=subprocess.DEVNULL,  # EOF: CLI exits after the single turn
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,  # captured separately, never mixed into the answer
                text=True,
                check=False,
                timeout=_LLM_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired as exc:
            raise LLMError(f"'{config.LLAMA_BIN}' timed out after {_LLM_TIMEOUT_S}s.") from exc
        elapsed = time.perf_counter() - start
    finally:
        Path(prompt_file).unlink(missing_ok=True)
    if proc.returncode != 0:
        raise LLMError(f"'{config.LLAMA_BIN}' failed with exit code {proc.returncode}.")
    return parse_llm_output(proc.stdout, prompt), elapsed
