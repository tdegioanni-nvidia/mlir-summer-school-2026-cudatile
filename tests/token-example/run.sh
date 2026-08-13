#!/usr/bin/env bash
set -euo pipefail

example_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
runtime_python="python"

"$example_dir/../../cuda-tile-translate" \
  --mlir-to-cudatilebc \
  --no-implicit-module \
  --bytecode-version=13.3 \
  "$example_dir/token-example.mlir" \
  -o "$example_dir/token-example.tileirbc"

"$runtime_python" "$example_dir/run.py"
