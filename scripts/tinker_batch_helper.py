"""Weight-processing helper: runs in the CONVERSION interpreter (torch present).

Subcommands (all local; only `export` touches the network via the Tinker API):
- dump-data: render a messages-JSONL file to plain-JSON Datum payloads
  (token-id lists) on stdout for the training process to reconstruct.
- export: download a checkpoint + build the PEFT adapter (needs torch).

Never imported by scripts/tinker_dryrun.py (it would drag torch into the
training interpreter). Invoked as a subprocess with --convert-python.
Reads TINKER_API_KEY from the environment for `export`; never prints it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_dump_data(args: argparse.Namespace) -> int:
    from tinker_cookbook.supervised.data import FromConversationFileBuilder
    from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

    builder = FromConversationFileBuilder(
        file_path=args.train,
        common_config=ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer=args.model,
            renderer_name=args.renderer,
            max_length=args.max_length,
            batch_size=args.batch_size,
        ),
    )
    train_ds, _ = builder()
    examples = []
    idx = 0
    while True:
        try:
            batch = train_ds.get_batch(idx)
        except Exception:
            break
        if not batch:
            break
        for datum in batch:
            if datum.model_input_spans is not None or datum.loss_fn_input_spans is not None:
                raise ValueError("provenance spans present: extend serialization first, "
                                 "refusing to silently drop training signal")
            loss = {}
            for key, tensor in datum.loss_fn_inputs.items():
                loss[key] = {"data": tensor.data, "dtype": str(tensor.dtype)}
            examples.append({"input_tokens": datum.model_input.to_ints(), "loss": loss})
            if len(examples) >= args.max_examples:
                break
        if len(examples) >= args.max_examples:
            break
        idx += 1
    payload = {"examples": examples, "model": args.model, "renderer": args.renderer}
    if args.out:
        Path(args.out).write_text(json.dumps(payload) + "\n", encoding="utf-8")
    else:
        print(json.dumps(payload))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    import os

    if not os.environ.get("TINKER_API_KEY"):
        print("REFUSING: TINKER_API_KEY is not set (env only, never logged).")
        return 2
    from tinker_cookbook import weights

    weights.download(tinker_path=args.tinker_path, output_dir=args.ckpt_dir)
    weights.build_lora_adapter(base_model=args.model, adapter_path=args.ckpt_dir,
                               output_path=args.peft_dir)
    print(f"exported PEFT adapter: {args.peft_dir}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tinker weight-processing helper.")
    sub = parser.add_subparsers(dest="command", required=True)
    dump = sub.add_parser("dump-data", help="render JSONL to Datum JSON")
    dump.add_argument("--train", required=True)
    dump.add_argument("--model", default="Qwen/Qwen3.5-4B")
    dump.add_argument("--renderer", default="qwen3")
    dump.add_argument("--max-length", type=int, default=2048)
    dump.add_argument("--batch-size", type=int, default=2)
    dump.add_argument("--max-examples", type=int, default=2)
    dump.add_argument("--out", default=None)
    exp = sub.add_parser("export", help="download checkpoint + build PEFT adapter")
    exp.add_argument("--tinker-path", required=True)
    exp.add_argument("--ckpt-dir", required=True)
    exp.add_argument("--peft-dir", required=True)
    exp.add_argument("--model", default="Qwen/Qwen3.5-4B")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "dump-data":
        return cmd_dump_data(args)
    return cmd_export(args)


if __name__ == "__main__":
    raise SystemExit(main())
