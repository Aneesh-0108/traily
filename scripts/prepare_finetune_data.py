"""Build deterministic SFT data for the later Tinker LoRA experiment (OFFLINE).

Reads one validated local OSM pack, samples seeded query points inside its
bbox, and takes the 5 nearest valid named places per point using the SAME
`nearest_in_pack` + haversine code as the runtime (no second formula).

Target answers reuse the runtime's deterministic `fallback_answer`
(no LLM invention): every supplied place once, with its distance, one
sentence, no Markdown. Output lines match Tinker's
`FromConversationFileBuilder` schema — {"messages": [{role, content}]}
(see tinker-cookbook supervised API reference; re-verify with a dry run
before any paid training job).

Dedup + split policy (all seeded/deterministic):
- candidates with pairwise query separation < MIN_SEPARATION_M dropped;
- identical ordered place sets dropped (same point sampled twice);
- contexts are split by UNORDERED place-id set: no eval id-set may appear
  in train, in any order, with any question wording;
- (ordered ids, question) unique within each file.

Writes data/finetune/train.jsonl + eval.jsonl + manifest.json.
Refuses to fabricate: exits non-zero when the pack cannot supply the
requested eval size with disjoint contexts.

Usage:
  python3 scripts/prepare_finetune_data.py --pack data/packs/<area>.json \\
      [--seed 7] [--train-target 150] [--eval-size 20] [--places 5] [--out-dir data/finetune]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.llm.answer import build_prompt, fallback_answer, missing_places  # noqa: E402
from src.places.geo import haversine_m, nearest_in_pack  # noqa: E402
from src.places.pack import in_bbox, load_pack  # noqa: E402

GENERATOR_VERSION = "finetune-gen-v1"
MIN_SEPARATION_M = 150.0
GRID_OVERSAMPLE = 40  # candidate points per wanted example
MAX_CANDIDATES = 4000

QUESTIONS = (
    "What's nearby?",
    "What are the nearby places?",
    "What is around here?",
    "Anything interesting nearby?",
    "What's close by?",
)

SYSTEM_MESSAGE = (
    "You are a voice trail companion. Answer only using the supplied nearby "
    "places. Never invent names, distances, directions, or facts. At most "
    "two short sentences, natural spoken English, no Markdown, no lists."
)


def sample_candidates(pack: dict, seed: int, n_want: int, n_places: int) -> list[dict]:
    """Seeded grid-jitter query points -> unique ordered nearby-5 contexts."""
    rng = random.Random(seed)
    bbox = pack["bbox"]
    n_try = min(n_want * GRID_OVERSAMPLE, MAX_CANDIDATES)
    candidates: list[dict] = []
    seen_ordered: set[tuple] = set()
    accepted_pts: list[tuple[float, float]] = []
    for _ in range(n_try):
        lat = bbox["min_lat"] + rng.random() * (bbox["max_lat"] - bbox["min_lat"])
        lon = bbox["min_lon"] + rng.random() * (bbox["max_lon"] - bbox["min_lon"])
        if not in_bbox(lat, lon, bbox):
            continue
        # Round BEFORE use: stored coords exactly reproduce stored distances.
        lat, lon = round(lat, 5), round(lon, 5)
        if any(haversine_m(lat, lon, a, b) < MIN_SEPARATION_M for a, b in accepted_pts):
            continue
        try:
            near = nearest_in_pack(pack, lat, lon, n_places)
        except ValueError:
            continue
        if len(near) < n_places:
            continue  # thin edge of pack: skip, never pad with fiction
        key = tuple((p["name"], p["distance_m"]) for p in near)
        if key in seen_ordered:
            continue
        seen_ordered.add(key)
        accepted_pts.append((lat, lon))
        candidates.append(
            {"lat": lat, "lon": lon, "places": near, "id_set": frozenset(p["name"] for p in near)}
        )
    return candidates


def validate_example(places: list[dict], answer: str, n_places: int) -> list[str]:
    """Deterministic checks; returns problem strings (empty == valid)."""
    problems: list[str] = []
    if len(places) != n_places:
        problems.append(f"expected {n_places} places, got {len(places)}")
    if missing_places(answer, places):
        problems.append(f"answer omits: {missing_places(answer, places)}")
    if answer.count(".") > 2 or "\n" in answer:
        problems.append("answer exceeds two sentences/one line")
    for ch in "*#{}[]|`_":
        if ch in answer:
            problems.append(f"answer contains non-spoken char {ch!r}")
            break
    return problems


def to_jsonl_line(question: str, places: list[dict], answer: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": build_prompt(question, places)},
            {"role": "assistant", "content": answer},
        ]
    }


def split_train_eval(
    candidates: list[dict], seed: int, train_target: int, eval_size: int
) -> tuple[list[dict], list[dict]]:
    """Split by unordered place-id set groups: whole groups go to train XOR
    eval, so no eval 5-set appears in train in any order or wording.
    Eval takes one example from each of the first `eval_size` groups
    (max diversity); train takes all examples of the remaining groups.
    """
    groups: dict[frozenset, list[dict]] = {}
    for cand in candidates:
        groups.setdefault(cand["id_set"], []).append(cand)
    order = list(groups.values())
    random.Random(seed + 1).shuffle(order)
    if len(order) < eval_size + 1:
        raise SystemExit(
            f"insufficient distinct place sets: {len(order)} groups for "
            f"{eval_size} eval + train (pack too small — expand the source area)"
        )
    eval_ctx = [g[0] for g in order[:eval_size]]
    train: list[dict] = []
    for g in order[eval_size:]:
        train.extend(g)
    return train[:train_target], eval_ctx


def write_split(
    path: Path, contexts: list[dict], seed: int, n_places: int, split: str
) -> tuple[int, list[str]]:
    """Validate + write one JSONL file. Returns (count, skipped_reasons)."""
    rng = random.Random(seed + (0 if split == "train" else 999))
    seen: set[tuple] = set()
    skipped: list[str] = []
    written = 0
    with path.open("w", encoding="utf-8") as fh:
        for cand in contexts:
            question = QUESTIONS[rng.randrange(len(QUESTIONS))]
            key = (tuple((p["name"], p["distance_m"]) for p in cand["places"]), question)
            if key in seen:
                skipped.append("dup-input")
                continue
            seen.add(key)
            answer = fallback_answer(cand["places"])
            problems = validate_example(cand["places"], answer, n_places)
            if problems:
                skipped.append(";".join(problems))
                continue
            fh.write(json.dumps(to_jsonl_line(question, cand["places"], answer),
                                ensure_ascii=False) + "\n")
            written += 1
    return written, skipped


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic SFT data (offline).")
    parser.add_argument("--pack", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-target", type=int, default=150)
    parser.add_argument("--eval-size", type=int, default=20)
    parser.add_argument("--places", type=int, default=5)
    parser.add_argument("--out-dir", default="data/finetune")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not 5 <= args.places <= 10:
        print(f"--places must be within the project 5–10 limit (got {args.places})")
        return 2
    try:
        pack = load_pack(args.pack)
    except ValueError as exc:
        print(f"Source pack invalid: {exc}")
        return 2
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = sample_candidates(pack, args.seed, args.train_target + args.eval_size, args.places)
    print(f"candidate unique contexts: {len(candidates)}")
    train_ctx, eval_ctx = split_train_eval(candidates, args.seed, args.train_target, args.eval_size)
    n_train, skip_train = write_split(out_dir / "train.jsonl", train_ctx, args.seed, args.places, "train")
    n_eval, skip_eval = write_split(out_dir / "eval.jsonl", eval_ctx, args.seed, args.places, "eval")
    manifest = {
        "generator": GENERATOR_VERSION,
        "seed": args.seed,
        "source_pack": str(args.pack),
        "pack_places": len(pack["places"]),
        "n_places_per_example": args.places,
        "train_examples": n_train,
        "eval_examples": n_eval,
        "skipped_train": len(skip_train),
        "skipped_eval": len(skip_eval),
        "tinker_schema": "messages[] role/content per line (FromConversationFileBuilder; re-verify at train time)",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"train: {n_train} eval: {n_eval} skipped={len(skip_train) + len(skip_eval)}")
    print(f"wrote {out_dir}/train.jsonl {out_dir}/eval.jsonl {out_dir}/manifest.json")
    # Split-leakage proof: unordered id-sets disjoint.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
