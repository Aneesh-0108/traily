"""Pinned upstream LoRA converter metadata + preflight checks (tested).

The installed llama-cli (0.4.0-dev, commit 5266f24da7, 2026-09-04) predates
the upstream Qwen3.5 LoRA V-head fix, so conversion must use upstream
sources at exactly this commit — never the installed package, never an
assumed master:

- PINNED_COMMIT = b016f461be57 (2026-09-30):
  "convert : fix LoRA conversion crash for Qwen3.5 V-head reorder (#28324)".
  Without it, converting adapters that target out_proj crashes with
  NotImplementedError (LoraTorchTensor cannot reshape its row dim).
- Layout under tools/lora-converter/ (git-ignored, built by
  tools/fetch-converter.sh): convert_lora_to_gguf.py, conversion/*.py
  (94 files), gguf-py/gguf/*.py (repo version — PyPI gguf 0.19.0 predates
  MODEL_ARCH.DFLASH and FAILS to import the pinned converter).
- Runtime python needs: torch, safetensors, numpy, transformers,
  huggingface_hub (gguf comes from the checkout via PYTHONPATH).

Verified evidence (2026-10-11, synthetic PEFT adapter, real Qwen3.5-4B
dims): out_proj/in_proj_qkv V-head paths + classic o_proj + MLP convert
cleanly; output arch tag `qwen35` matches the base GGUF; the installed
llama-cli loads it via --lora (rc=0) and generates. Random weights prove
FORMAT/loadability, not numerical correctness — that needs the real
trained adapter (i.e. the dry run itself).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PINNED_COMMIT = "b016f461be57"
PINNED_COMMIT_DATE = "2026-09-30"
PINNED_SUBJECT = "convert : fix LoRA conversion crash for Qwen3.5 V-head reorder (#28324)"
CONVERTER_SCRIPT_NAME = "convert_lora_to_gguf.py"
CONVERTER_FILES = ("convert_lora_to_gguf.py", "conversion/__init__.py",
                   "conversion/qwen.py", "gguf-py/gguf/constants.py")
# Marker proving the V-head fix is present in the prepared sources.
VHEAD_FIX_MARKER = "get_lora_A_B"
CONVERTER_DEPS = ("torch", "safetensors", "numpy", "transformers", "huggingface_hub")
# Import contract per interpreter (verified 2026-10-11):
# - TRAIN interpreter needs ONLY the tinker SDK: cookbook/torch imports
#   there caused the ModuleNotFoundError that killed the first attempt.
# - CONVERT interpreter needs torch + transformers + cookbook rendering and
#   weight APIs on top of the converter deps.
TRAIN_IMPORTS = ("tinker",)
CONVERT_IMPORTS = CONVERTER_DEPS + ("tinker_cookbook.supervised.data",
                                   "tinker_cookbook.weights")
HELPER_SCRIPT = "scripts/tinker_batch_helper.py"

# Distinct exit: converter unavailable (pre- AND post-billing checkpoints).
CONVERTER_ABSENT_EXIT = 4


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent  # src/finetune/ -> root


def default_converter_dir() -> Path:
    return _repo_root() / "tools" / "lora-converter"


def converter_script(repo_root: Path | None = None) -> Path:
    base = Path(repo_root) if repo_root else _repo_root()
    return base / "tools" / "lora-converter" / CONVERTER_SCRIPT_NAME


def check_files(converter_dir: str | Path) -> tuple[bool, str]:
    """Required sources present AND carrying the V-head fix marker."""
    d = Path(converter_dir)
    missing = [f for f in CONVERTER_FILES if not (d / f).is_file()]
    if missing:
        return False, f"missing converter files: {missing} (run tools/fetch-converter.sh)"
    qwen = (d / "conversion" / "qwen.py").read_text(encoding="utf-8")
    if VHEAD_FIX_MARKER not in qwen:
        return False, "conversion/qwen.py lacks the Qwen3.5 V-head fix; re-fetch at the pinned commit"
    return True, "converter sources OK"


def check_python_modules(python: str, modules: tuple[str, ...] = CONVERTER_DEPS,
                         hint: str = "install per tools/requirements-convert.txt") -> tuple[bool, str]:
    """The given interpreter can import every listed module."""
    missing: list[str] = []
    for mod in modules:
        proc = subprocess.run(
            [python, "-c", f"import {mod}"], capture_output=True, check=False)
        if proc.returncode != 0:
            missing.append(mod)
    if missing:
        return False, f"{python} lacks: {missing} ({hint})"
    return True, "converter dependencies OK"


def preflight(python: str = sys.executable,
              converter_dir: str | Path | None = None) -> tuple[bool, list[str]]:
    """Full preflight: files + fix marker + interpreter deps. No billing."""
    d = Path(converter_dir) if converter_dir else default_converter_dir()
    ok_files, msg_files = check_files(d)
    ok_mods, msg_mods = check_python_modules(python)
    return ok_files and ok_mods, [msg_files, msg_mods]
