#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
UV_BIN="uv"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv-simple-test}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

cd "${REPO_ROOT}"

# Ensure pip-installed scripts (including uv) are on PATH
PYTHON_SCRIPTS="$(${PYTHON_BIN} -c 'import sysconfig; print(sysconfig.get_path("scripts"))')"
USER_BASE="$(${PYTHON_BIN} -m site --user-base 2>/dev/null || true)"
PATH="${PYTHON_SCRIPTS}:${PATH}"
if [[ -n "${USER_BASE}" ]]; then
  PATH="${USER_BASE}/bin:${PATH}"
fi

if ! command -v "${UV_BIN}" >/dev/null 2>&1; then
  "${PYTHON_BIN}" -m pip install --upgrade pip
  "${PYTHON_BIN}" -m pip install uv
  hash -r
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  "${UV_BIN}" venv --python "${PYTHON_BIN}" "${VENV_DIR}"
fi

VENV_PYTHON="${VENV_DIR}/bin/python"
[[ -x "${VENV_PYTHON}" ]] || { echo "Python not found in ${VENV_DIR}"; exit 1; }

"${UV_BIN}" pip install --python "${VENV_PYTHON}" vllm==0.11.0
"${UV_BIN}" pip install --python "${VENV_PYTHON}" -e ".[dev]"
"${VENV_PYTHON}" -m pytest tests/test_omni_llm.py
