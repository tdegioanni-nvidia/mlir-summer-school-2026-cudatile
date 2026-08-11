#!/usr/bin/env bash
set -euo pipefail

example_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
tile_repo="${CUDA_TILE_REPO:-$example_dir/../../cuda-tile}"
runtime_python="${CUDA_TILE_RUNTIME_PYTHON:-$example_dir/../venv/bin/python}"

"$tile_repo/build/bin/cuda-tile-translate" \
  --mlir-to-cudatilebc \
  --no-implicit-module \
  --bytecode-version=13.3 \
  "$example_dir/token-example.mlir" \
  -o "$example_dir/token-example.tileirbc"

"$runtime_python" "$example_dir/run.py"
