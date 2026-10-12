"""Tinker LoRA conversion smoke test for Qwen/Qwen3.5-4B — PREPARED, NOT EXECUTED.

Smallest practical chain prover (2 examples, batch 2, 2 steps, rank 4,
seed 7): one optimizer update minimum, exportable checkpoint, PEFT
export, PEFT→GGUF-LoRA conversion attempt, and a `llama-cli --lora`
smoke test. API shapes verified against installed tinker 0.33.1 /
tinker-cookbook 0.5.7 signatures (NOT executed — no key, no billing).

Environment separation (remote training stays lightweight):
- TRAIN interpreter (this script): `tinker` SDK ONLY. No torch, no
  cookbook — the ModuleNotFoundError that killed the first attempt came
  from importing cookbook here.
- CONVERT interpreter (--convert-python): torch + transformers +
  tinker_cookbook (data rendering, PEFT export) + pinned converter.
  Batch building (`dump-data`) and checkpoint export (`export`) run there
  via scripts/tinker_batch_helper.py; plain-JSON Datum payloads cross the
  boundary and are rebuilt with torch-free `tinker.types` here.

Converter: pinned upstream checkout tools/lora-converter (commit
b016f461be57 with the Qwen3.5 V-head fix); a preflight check for its
sources + BOTH interpreters' imports runs BEFORE any billable call
(exit 4 on failure). See src/finetune/converter.py and
tools/fetch-converter.sh.

DO NOT RUN without explicit approval: executing bills the $10 credit
(estimate printed first; expected ≈$0.001, see src/finetune/costs.py).

Prerequisites (isolated envs; runtime .venv untouched):
  python3 -m venv /tmp/traily-train-venv
  /tmp/traily-train-venv/bin/pip install tinker
  python3 -m venv /tmp/traily-convert-venv
  /tmp/traily-convert-venv/bin/pip install -r tools/requirements-convert.txt
  tools/fetch-converter.sh   # only if tools/lora-converter/ is absent
  export TINKER_API_KEY=...   # env only; never print or commit it

Exact launch command (from repo root):
  /tmp/traily-train-venv/bin/python scripts/tinker_dryrun.py \\
      --train data/finetune/dryrun_train.jsonl \\
      --convert-python /tmp/traily-convert-venv/bin/python

Dry-run subset (deterministic first 2 training lines, fixed order):
  head -2 data/finetune/train.jsonl > data/finetune/dryrun_train.jsonl
The 20 held-out eval examples are NEVER used here (one is read back only
as a fixed smoke-test prompt AFTER conversion, never for training).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.finetune.costs import DRYRUN, MODEL_ID, dryrun_estimate  # noqa: E402
from src.finetune.converter import (  # noqa: E402
    CONVERTER_ABSENT_EXIT,
    converter_script,
    default_converter_dir,
    preflight,
)

SUBSET_EXAMPLES = 2

# Exit codes: 2 usage/key/local failure, 3 smoke-test failure,
# 4 converter unavailable, 5 billable-phase remote failure.
TRAIN_FAILED_EXIT = 5
# Bound on the sampler-weight save only; per-step futures resolve
# unbounded like the cookbook trainer (a hung save must not hang forever).
SAVE_TIMEOUT_S = 600


class TinkerRunError(RuntimeError):
    """A billable-phase remote operation failed AFTER resolving its future."""


def run_training_steps(train_client, examples: list, batch_size: int,
                       steps: int, learning_rate: float) -> int:
    """Submit forward/backward + optimizer steps, resolving each future inline.

    Every Tinker training call returns an APIFuture; `.result()` blocks until
    the server completes that exact operation (and raises its remote error).
    Resolving inline preserves submit order and surfaces per-step failures
    instead of dying later with AttributeError on an unresolved future.
    """
    n_batches = max(1, math.ceil(len(examples) / batch_size))
    for step in range(steps):
        batch = examples[(step % n_batches) * batch_size:
                         (step % n_batches + 1) * batch_size]
        try:
            train_client.forward_backward(batch, loss_fn="cross_entropy").result()
            train_client.optim_step(
                adam_params=train_client_adam_params(learning_rate)).result()
        except Exception as exc:
            raise TinkerRunError(f"training step {step} failed: {exc}") from exc
    return steps


def train_client_adam_params(learning_rate: float):
    """AdamParams constructor resolved lazily so module import needs no SDK."""
    import tinker  # noqa: E402 (train interpreter only; never the convert env)

    return tinker.AdamParams(learning_rate=learning_rate)


def save_sampler_checkpoint(train_client, name: str = "traily-smoke-sampler",
                            timeout: int = SAVE_TIMEOUT_S) -> str:
    """Save SAMPLER weights for export/inference; return the tinker:// path.

    This is deliberately NOT save_state(): save_state() persists
    training-state for RESUMING training, while weights.download /
    build_lora_adapter expect a `tinker://.../sampler_weights/...` path
    like the one this returns. The 2-step smoke chain has nothing worth
    resuming, so only the sampler save is billed.
    """
    try:
        resp = train_client.save_weights_for_sampler(name=name).result(timeout=timeout)
    except Exception as exc:
        raise TinkerRunError(f"sampler checkpoint save failed: {exc}") from exc
    path = getattr(resp, "path", None)
    if not path or not isinstance(path, str):
        raise TinkerRunError(
            f"sampler checkpoint save returned no usable path: {resp!r}")
    return path

# PEFT→GGUF-LoRA converter: pinned upstream checkout (tools/lora-converter,
# commit b016f461be57 with the Qwen3.5 V-head fix). Override only to point
# at an identically-pinned tree.
CONVERT_LORA_SCRIPT = os.environ.get("TRAILY_CONVERT_LORA",
                                     str(converter_script()))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tinker LoRA smoke test (bills credit!).")
    parser.add_argument("--train", required=True, help="subset JSONL (messages format)")
    parser.add_argument("--steps", type=int, default=DRYRUN["steps"])
    parser.add_argument("--rank", type=int, default=DRYRUN["lora_rank"])
    parser.add_argument("--seed", type=int, default=DRYRUN["seed"])
    parser.add_argument("--learning-rate", type=float, default=DRYRUN["learning_rate"])
    parser.add_argument("--batch-size", type=int, default=DRYRUN["batch_size"])
    parser.add_argument("--eval-prompt-src", default="data/finetune/eval.jsonl",
                        help="held-out file to borrow ONE smoke-test prompt from (never trained on)")
    parser.add_argument("--workdir", default="local/traily-smoke",
                        help="local dir for adapter + converted files (git-ignored)")
    parser.add_argument("--convert-python", default=None,
                        help="interpreter that runs the converter (needs torch/safetensors/ "
                             "transformers; default: TRAILY_CONVERT_PYTHON or this python)")
    parser.add_argument("--dry-print-only", action="store_true",
                        help="print estimate + planned calls WITHOUT billing anything")
    return parser.parse_args(argv)


def validate_subset(path: str) -> int:
    """Check the subset file matches the Tinker conversation schema."""
    n = 0
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if not line.strip():
                continue
            obj = json.loads(line)
            if "messages" not in obj:
                raise ValueError(f"line {i}: missing 'messages' key (DataFormatError)")
            roles = [m["role"] for m in obj["messages"]]
            if "user" not in roles or "assistant" not in roles:
                raise ValueError(f"line {i}: need user + assistant messages")
            n += 1
    if n != SUBSET_EXAMPLES:
        raise ValueError(f"smoke test needs exactly {SUBSET_EXAMPLES} examples, got {n}")
    return n


def first_eval_user_prompt(path: str) -> str:
    """One fixed smoke-test prompt from the held-out file (read-only)."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                obj = json.loads(line)
                return next(m["content"] for m in obj["messages"] if m["role"] == "user")
    raise ValueError(f"no examples in {path}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    est = dryrun_estimate()
    print(f"model: {MODEL_ID} | steps: {args.steps} | rank: {args.rank} | "
          f"batch: {args.batch_size} | seed: {args.seed}")
    print(f"estimate: ~{est['tokens']} train tokens ≈ ${est['usd']:.4f} "
          f"(+ ~$0.00 checkpoint storage; local eval free)")
    n = validate_subset(args.train)
    print(f"subset OK: {n} examples in {args.train}")
    # Preflight BEFORE anything billable, per environment:
    # - converter sources + fix marker (always required);
    # - TRAIN interpreter: tinker SDK only (torch must NOT be needed here);
    # - CONVERT interpreter: torch + cookbook data/weights APIs + converter deps;
    # - batch helper script present.
    # Any failure stops here (exit 4), never after paid training has happened.
    from src.finetune.converter import (  # noqa: E402
        CONVERT_IMPORTS,
        HELPER_SCRIPT,
        TRAIN_IMPORTS,
        check_files,
        check_python_modules,
    )

    conv_dir = Path(CONVERT_LORA_SCRIPT).parent
    convert_python = args.convert_python or os.environ.get("TRAILY_CONVERT_PYTHON") \
        or sys.executable
    helper = REPO_ROOT / HELPER_SCRIPT
    checks: list[tuple[bool, str]] = []
    ok_files, msg_files = check_files(conv_dir)
    checks.append((ok_files, f"sources: {msg_files}"))
    ok_train, msg_train = check_python_modules(
        sys.executable, TRAIN_IMPORTS, hint="pip install tinker (isolated train venv)")
    checks.append((ok_train, f"train env [{sys.executable}]: {msg_train}"))
    ok_conv, msg_conv = check_python_modules(convert_python, CONVERT_IMPORTS)
    checks.append((ok_conv, f"convert env [{convert_python}]: {msg_conv}"))
    checks.append((helper.is_file(), f"helper: {helper} "
                   f"{'present' if helper.is_file() else 'MISSING'}"))
    for _, msg in checks:
        print(f"converter preflight: {msg}")
    failed = [msg for ok, msg in checks if not ok]
    if failed and not args.dry_print_only:
        print(f"STOP (exit {CONVERTER_ABSENT_EXIT}): preflight failed — "
              "fix the items above (tools/fetch-converter.sh, "
              "tools/requirements-convert.txt, --convert-python); nothing billed.")
        return CONVERTER_ABSENT_EXIT
    if args.dry_print_only:
        print("dry-print-only: stopping BEFORE any billable call. Planned:")
        print("  ServiceClient() [auth: TINKER_API_KEY env] -> "
              f"create_lora_training_client(base_model={MODEL_ID!r}, "
              f"rank={args.rank}, seed={args.seed})")
        print("  helper dump-data (convert env) -> rebuild Datum locally -> "
              "resolved forward_backward + optim_step(AdamParams) loop -> "
              "save_weights_for_sampler (NOT save_state: export needs the sampler path)")
        print("  helper export (convert env): weights.download -> weights.build_lora_adapter")
        print(f"  convert: {CONVERT_LORA_SCRIPT} <peft> --outfile adapter.gguf")
        print("  smoke: llama-cli -m <base Q4_K_M> --lora adapter.gguf --reasoning off ...")
        return 0
    if not os.environ.get("TINKER_API_KEY"):
        print("REFUSING: TINKER_API_KEY is not set (key stays in env, never in code).")
        return 2
    # --- Billable section: requires explicit approval (see module docstring).
    # Train interpreter imports tinker ONLY: cookbook/torch live exclusively
    # in the convert interpreter (importing them here was the torch failure).
    import tinker  # noqa: E402 (imported here so --dry-print-only needs no SDK)
    from tinker.types import Datum, ModelInput, TensorData  # noqa: E402

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    def run_helper(*helper_argv: str, capture: bool = False) -> subprocess.CompletedProcess:
        return subprocess.run(
            [convert_python, str(REPO_ROOT / "scripts" / "tinker_batch_helper.py"),
             *helper_argv],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
            text=True, check=False, cwd=str(REPO_ROOT),
        )

    def rebuild_batch(payload_examples: list[dict]) -> list:
        rebuilt = []
        for ex in payload_examples:
            loss = {k: TensorData(data=v["data"], dtype=v["dtype"])
                    for k, v in ex["loss"].items()}
            rebuilt.append(Datum(model_input=ModelInput.from_ints(ex["input_tokens"]),
                                 loss_fn_inputs=loss))
        return rebuilt

    # Phase 0: render batches once in the convert interpreter (torch lives
    # there); training consumes plain-JSON Datum payloads rebuilt below.
    dump = run_helper("dump-data", "--train", args.train, "--batch-size",
                      str(args.batch_size), "--max-examples", str(n), capture=True)
    if dump.returncode != 0:
        print(f"batch rendering FAILED (rc={dump.returncode}): {(dump.stderr or '')[-2000:]}")
        return 2
    try:
        rendered = json.loads(dump.stdout)["examples"]
    except (json.JSONDecodeError, KeyError) as exc:
        print(f"batch payload unreadable: {exc}")
        return 2
    if len(rendered) < n:
        print(f"batch rendering returned {len(rendered)} examples, need {n}")
        return 2
    print(f"batches rendered in convert env: {len(rendered)} examples")

    # Phase 1: train (minimum chain: resolved updates + sampler checkpoint).
    service = tinker.ServiceClient()
    train_client = service.create_lora_training_client(
        base_model=MODEL_ID, rank=args.rank, seed=args.seed)
    examples = rebuild_batch(rendered)
    try:
        run_training_steps(train_client, examples, args.batch_size,
                           args.steps, args.learning_rate)
        ckpt_path = save_sampler_checkpoint(train_client)
    except TinkerRunError as exc:
        print(f"Training failed: {exc}")
        return TRAIN_FAILED_EXIT
    print(f"sampler checkpoint saved: {ckpt_path}")

    # Phase 2: export + download via the helper (cookbook weights APIs need
    # torch, so they run under --convert-python; key passes via inherited env).
    ckpt_dir = workdir / "ckpt"
    peft_dir = workdir / "peft"
    exp = run_helper("export", "--tinker-path", ckpt_path,
                     "--ckpt-dir", str(ckpt_dir), "--peft-dir", str(peft_dir))
    if exp.returncode != 0:
        print(f"export FAILED (rc={exp.returncode}); checkpoint retained at {ckpt_path}")
        return 2
    print(f"PEFT adapter exported: {peft_dir}")

    # Phase 3: convert PEFT -> GGUF-LoRA (prepared checkout; preflighted above).
    # Safety net: sources could vanish mid-run; re-check before invoking.
    from src.finetune.converter import check_files as _check_files  # noqa: E402

    converter = Path(CONVERT_LORA_SCRIPT)
    lora_gguf = workdir / "adapter.gguf"
    ok_files, _ = _check_files(converter.parent)
    if not ok_files:
        print(f"STOP (exit {CONVERTER_ABSENT_EXIT}): converter sources vanished mid-run; "
              "rebuild with tools/fetch-converter.sh.")
        return CONVERTER_ABSENT_EXIT
    # repo gguf-py shadows any pip gguf (PyPI 0.19.0 predates MODEL_ARCH.DFLASH).
    env = dict(os.environ)
    gguf_py = str(converter.parent / "gguf-py")
    env["PYTHONPATH"] = gguf_py + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    proc = subprocess.run(
        [convert_python, str(converter), str(peft_dir), "--outfile", str(lora_gguf)],
        capture_output=True, text=True, check=False, cwd=str(converter.parent), env=env,
    )
    if proc.returncode != 0 or not lora_gguf.is_file():
        print(f"conversion FAILED (rc={proc.returncode}): {proc.stderr[-2000:]}")
        return 4
    print(f"converted: {lora_gguf}")

    # Phase 4: smoke-test with installed llama.cpp (base stays Q4_K_M).
    from src import config as _config  # noqa: E402 (local-only; after billing section)
    from src.llm.answer import build_llm_argv, parse_llm_output  # noqa: E402

    prompt = first_eval_user_prompt(args.eval_prompt_src)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
        tmp.write(prompt)
        prompt_file = tmp.name
    try:
        argv_llama = build_llm_argv(prompt_file) + ["--lora", str(lora_gguf)]
        proc = subprocess.run(
            argv_llama, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, check=False, timeout=600,
        )
    finally:
        Path(prompt_file).unlink(missing_ok=True)
    if proc.returncode != 0:
        print(f"llama-cli --lora FAILED (rc={proc.returncode}); inspect stderr log, not the key.")
        return 3
    try:
        answer = parse_llm_output(proc.stdout, prompt)
    except Exception as exc:  # fail-closed parser: report, don't speak logs
        print(f"smoke answer unparseable: {exc}")
        return 3
    print(f"SMOKE ANSWER (adapter active): {answer!r}")
    print("Chain complete: train -> checkpoint -> PEFT -> GGUF-LoRA -> llama.cpp answer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
