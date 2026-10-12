#!/usr/bin/env bash
# Export a Tinker LoRA checkpoint and convert it for local llama.cpp.
# PREPARED PROCEDURE — DO NOT RUN without explicit approval.
# Mirrors scripts/tinker_dryrun.py phases 2-4 for manual use.
#
# Usage (after approval + successful dry-run training):
#   REQUIRE_EXPLICIT_APPROVAL=1 bash scripts/tinker_export_local.sh \
#       tinker://<run_id>/sampler_weights/final ./local/traily-smoke
#
# Chain (CONFIRMED vs UNVERIFIED marked inline):
#   1. download checkpoint            [CONFIRMED API: weights.download]
#   2. PEFT adapter export            [CONFIRMED API: weights.build_lora_adapter]
#   3. PEFT -> GGUF-LoRA conversion   [PREPARED: tools/lora-converter at
#      pinned commit b016f461be57 (Qwen3.5 V-head fix); synthetic-adapter
#      conversion + llama-cli --lora load VERIFIED 2026-10-11. Real-adapter
#      numerical correctness still needs the trained adapter itself.]
#   4. run: llama-cli --lora         [CONFIRMED: flag EXISTS in installed llama-cli;
#      synthetic adapter loads (rc=0) and generates]
#   OR (alternative, fully merged model):
#   5. merge to full HF model        [CONFIRMED API: weights.build_hf_model]
#   6. HF -> GGUF conversion         [UNVERIFIED: convert_hf_to_gguf.py absent;
#      needs ~8GB HF base weights Qwen/Qwen3.5-4B]
#   7. quantize Q4_K_M               [CONFIRMED: llama-quantize INSTALLED]
#   8. eval: TRAILY_LLM_MODEL=...    [CONFIRMED: env override + harness exist]
set -u
if [ "${REQUIRE_EXPLICIT_APPROVAL:-0}" != "1" ]; then
  echo "REFUSING: set REQUIRE_EXPLICIT_APPROVAL=1 after explicit approval."
  echo "This procedure downloads checkpoints and may bill storage."
  exit 2
fi
if [ $# -ne 2 ]; then
  echo "usage: $0 tinker://<run>/sampler_weights/final <out-dir>"
  exit 2
fi
TINKER_PATH="$1"; OUT_DIR="$2"
if [ -z "${TINKER_API_KEY:-}" ]; then
  echo "REFUSING: TINKER_API_KEY is not set (env only, never committed)."
  exit 2
fi
VENV="${TRAILY_TRAIN_VENV:-/tmp/traily-train-venv}/bin/python"
echo "== 1. download checkpoint [CONFIRMED API] =="
echo "$VENV -c \"from tinker_cookbook import weights; weights.download(tinker_path='$TINKER_PATH', output_dir='$OUT_DIR/ckpt')\""
echo "== 2. PEFT export [CONFIRMED API] =="
echo "$VENV -c \"from tinker_cookbook import weights; weights.build_lora_adapter(base_model='Qwen/Qwen3.5-4B', adapter_path='$OUT_DIR/ckpt', output_path='$OUT_DIR/peft')\""
echo "== 3. PEFT->GGUF-LoRA [PREPARED checkout; synthetic-verified] =="
echo "CONV=tools/lora-converter  # pinned b016f461be57, V-head fix present"
echo "PYTHONPATH=\$CONV/gguf-py <convert-venv>/bin/python \$CONV/convert_lora_to_gguf.py $OUT_DIR/peft --outfile $OUT_DIR/adapter.gguf --base <dir-with-config.json>"
echo "== 4. smoke test [flag CONFIRMED, load UNTESTED] =="
echo "llama-cli -m models/llm/Qwen3.5-4B-Q4_K_M.gguf --lora $OUT_DIR/adapter.gguf -ngl 0 --reasoning off --jinja -st -c 2048 -b 128 -ub 64 --simple-io --no-display-prompt --no-warmup -n 96 -f <prompt> < /dev/null"
echo "== 5-8. merge alternative: build_hf_model [CONFIRMED] -> convert_hf_to_gguf.py [UNVERIFIED, absent] -> llama-quantize Q4_K_M [INSTALLED] -> TRAILY_LLM_MODEL=<out> python3 scripts/evaluate_baseline.py [CONFIRMED harness] =="
echo "Procedure printed only. No download, bill, or conversion was executed."
