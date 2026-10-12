#!/usr/bin/env bash
# Fetch the pinned upstream LoRA converter (reproducible, offline-safe afterwards).
# Pinned commit b016f461be57 (2026-09-30): Qwen3.5 LoRA V-head reorder fix.
# Newer master is NOT used: later conversion/qwen.py changes are unrelated
# features, and pinning keeps conversion bit-reproducible.
# Output goes to tools/lora-converter/ (git-ignored). Runtime .venv untouched.
set -u
COMMIT="${TRAILY_CONVERTER_COMMIT:-b016f461be57}"
DEST="${1:-tools/lora-converter}"
RAW="https://raw.githubusercontent.com/ggml-org/llama.cpp/$COMMIT"
echo "Fetching converter at $COMMIT -> $DEST"
mkdir -p "$DEST/conversion" "$DEST/gguf-py/gguf"
curl -fsSL "$RAW/convert_lora_to_gguf.py" -o "$DEST/convert_lora_to_gguf.py" || exit 1
curl -fsSL "https://api.github.com/repos/ggml-org/llama.cpp/contents/conversion?ref=$COMMIT" \
  | python3 -c "import json,sys; print('\n'.join(f['name'] for f in json.load(sys.stdin) if f['name'].endswith('.py')))" > "$DEST/.files.txt"
while read -r f; do
  [ -n "$f" ] || continue
  curl -fsSL "$RAW/conversion/$f" -o "$DEST/conversion/$f" || exit 1
done < "$DEST/.files.txt"
curl -fsSL "https://api.github.com/repos/ggml-org/llama.cpp/contents/gguf-py/gguf?ref=$COMMIT" \
  | python3 -c "import json,sys; print('\n'.join(f['name'] for f in json.load(sys.stdin) if f['name'].endswith('.py')))" > "$DEST/.gguf-files.txt"
while read -r f; do
  [ -n "$f" ] || continue
  curl -fsSL "$RAW/gguf-py/gguf/$f" -o "$DEST/gguf-py/gguf/$f" || exit 1
done < "$DEST/.gguf-files.txt"
grep -q "get_lora_A_B" "$DEST/conversion/qwen.py" \
  || { echo "V-head fix marker MISSING — aborting"; exit 1; }
echo "$COMMIT" > "$DEST/.pinned-commit"
echo "OK: $(wc -l < "$DEST/.files.txt") conversion files + gguf-py at $COMMIT (V-head fix present)"
