#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/pytorch/bin/python}"
exec "$PYTHON_BIN" -m vestibular_fusion reproduce \
  --config configs/paths.local.json \
  --datasets monifeixing vrq city \
  --device "${DEVICE:-cuda}"
