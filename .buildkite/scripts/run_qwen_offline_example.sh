#!/usr/bin/env bash

set -euo pipefail

# Move to repo root (script lives in .buildkite/scripts/)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

python3 -m pip install --upgrade pip uv
UV_BIN="uv"
command -v "${UV_BIN}" >/dev/null 2>&1 || UV_BIN="python3 -m uv"

${UV_BIN} python install 3.12
${UV_BIN} venv --python 3.12 .venv
source .venv/bin/activate

${UV_BIN} pip install --python .venv/bin/python vllm==0.11.0 --torch-backend=auto
${UV_BIN} pip install --python .venv/bin/python -e .

EXAMPLE_DIR="examples/offline_inference/qwen2_5_omni"
cd "${EXAMPLE_DIR}"

if [[ ! -f top100.txt ]]; then
  echo "Hello from vLLM-omni Buildkite smoke test." > top100.txt
fi

bash run_multiple_prompts.sh
