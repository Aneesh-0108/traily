"""Baseline evaluation: run the BASE local Qwen3.5-4B on held-out questions.

Reads data/finetune/eval.jsonl, sends each user prompt through the SAME
`run_llm` adapter as the runtime (CPU, --reasoning off), and writes
results JSONL with raw outputs plus DETERMINISTIC exact checks:

- completeness: every supplied name + exact distance present;
- grounding: no invented place names (names outside the supplied set);
- brevity/style: <= 2 sentences, single line, no Markdown chars;
- invented directions: flags route words (left/right/turn/head/go straight).

Model-based qualitative grading is deliberately NOT included here —
exact checks first; keep graders separate. The saved raw outputs let a
later fine-tuned model be compared on identical inputs (AGENTS.md).

Usage:
  python3 scripts/evaluate_baseline.py [--eval data/finetune/eval.jsonl]
      [--out data/finetune/baseline_results.jsonl] [--limit N]
`--limit N` runs a representative prefix subset; default runs all.
Slow (~10-15 s/example on CPU): 20 examples take ~4-5 minutes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.llm.answer import LLMError, missing_places, run_llm  # noqa: E402

NON_SPOKEN_CHARS = "*#{}[]|`_"
DIRECTION_WORDS = (
    "turn left", "turn right", "go straight", "head north", "head south",
    "head east", "head west", "on your left", "on your right", "take the",
)


def parse_user_places(user_content: str) -> list[dict]:
    """Recover the supplied places from the runtime-shaped user prompt."""
    places = []
    for line in user_content.splitlines():
        m = re.match(r"^-\s+(.+?)\s+\((.+?),\s*(\d+)\s*m\)$", line.strip())
        if m:
            places.append({"name": m.group(1), "type": m.group(2),
                           "distance_m": int(m.group(3))})
    return places


# Common capitalized sentence words: never place names. Single-word
# matches in this set are heuristic noise, not hallucinations.
STOPWORDS = frozenset({
    "near", "nearby", "around", "close", "there", "these", "those", "this",
    "that", "each", "every", "further", "also", "just", "then", "first",
    "second", "next", "finally", "however", "moreover", "additionally",
    "you", "your", "here", "within", "besides", "closest", "other",
    "another", "first", "second",
})


def invented_names(answer: str, places: list[dict]) -> list[str]:
    """Capitalized phrases in the answer matching no supplied name.

    Only multi-word phrases or non-stopword singles count: bare "Nearby",
    "Around", "There" are sentence scaffolding, not invented places.
    """
    supplied = " | ".join(p["name"] for p in places).lower()
    invented = []
    for phrase in set(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z'’]+){0,3}", answer)):
        if phrase.lower() not in supplied and len(phrase) > 3:
            if " " not in phrase and phrase.lower() in STOPWORDS:
                continue
            invented.append(phrase)
    return sorted(invented)


def grade(answer: str, places: list[dict]) -> dict:
    missing = missing_places(answer, places)
    sentences = [s for s in re.split(r"[.!?]+", answer) if s.strip()]
    return {
        "complete": not missing,
        "missing": missing,
        "invented_names": invented_names(answer, places),
        "grounded": not invented_names(answer, places),
        "n_sentences": len(sentences),
        "brief": len(sentences) <= 2,
        "single_line": "\n" not in answer,
        "no_markdown": not any(c in answer for c in NON_SPOKEN_CHARS),
        "direction_words": [w for w in DIRECTION_WORDS if w in answer.lower()],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Baseline eval on held-out questions.")
    parser.add_argument("--eval", default="data/finetune/eval.jsonl")
    parser.add_argument("--out", default="data/finetune/baseline_results.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="0 = all examples")
    parser.add_argument("--regrade-only", action="store_true",
                        help="recompute metrics from saved model outputs (no inference)")
    args = parser.parse_args(argv)

    examples = [json.loads(l) for l in Path(args.eval).read_text(encoding="utf-8").splitlines()
                if l.strip()]
    if args.limit > 0:
        examples = examples[: args.limit]
    saved_by_index: dict[int, dict] = {}
    if args.regrade_only:
        try:
            for line in Path(args.out).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    saved_by_index[row["index"]] = row
        except OSError as exc:
            print(f"Cannot read saved results for regrade: {exc}")
            return 2
        print(f"regrade-only: recomputing metrics for {len(examples)} saved outputs")
    print(f"baseline: {len(examples)} held-out examples (base Qwen3.5-4B, local)")

    results = []
    for i, ex in enumerate(examples):
        user = next(m["content"] for m in ex["messages"] if m["role"] == "user")
        target = next(m["content"] for m in ex["messages"] if m["role"] == "assistant")
        places = parse_user_places(user)
        if args.regrade_only and i < len(saved_by_index):
            prev = saved_by_index[i]
            output, error, dt = prev["model_output"], prev["error"], prev["seconds"]
        else:
            t0 = time.perf_counter()
            try:
                output, _ = run_llm(user)
                error = None
            except LLMError as exc:
                output, error = "", str(exc)
            dt = time.perf_counter() - t0
        results.append({
            "index": i,
            "places": places,
            "target": target,
            "model_output": output,
            "error": error,
            "seconds": round(dt, 2),
            "metrics": grade(output, places) if output else {"complete": False, "error": error},
        })
        m = results[-1]["metrics"]
        print(f"[{i+1}/{len(examples)}] complete={m.get('complete')} "
              f"grounded={m.get('grounded')} {dt:.1f}s :: {output[:80]}")

    Path(args.out).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n", encoding="utf-8")
    n = len(results)
    complete = sum(1 for r in results if r["metrics"].get("complete"))
    grounded = sum(1 for r in results if r["metrics"].get("grounded"))
    avg_s = sum(r["seconds"] for r in results) / n if n else 0
    print(f"summary: {complete}/{n} complete, {grounded}/{n} grounded, avg {avg_s:.1f}s/example")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
