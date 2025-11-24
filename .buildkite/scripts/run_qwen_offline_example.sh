#!/usr/bin/env bash

set -euo pipefail

# Move to repo root (script lives in .buildkite/scripts/)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install uv

uv pip install vllm==0.11.0 --torch-backend=auto
uv pip install -e .

EXAMPLE_DIR="examples/offline_inference/qwen2_5_omni"
cd "${EXAMPLE_DIR}"

if [[ ! -f top100.txt ]]; then
  echo "Hello from vLLM-omni Buildkite smoke test." > top100.txt
fi

bash run_multiple_prompts.sh
