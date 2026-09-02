# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_local_minicpmo_perf_uses_omni_runner(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "tools" / "nightly" / "run_nightly_jobs.sh"
    env = os.environ.copy()
    env.update({"REPO_ROOT": str(repo_root), "LOG_DIR": str(tmp_path / "logs")})

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--test-type",
            "local",
            "--model-type",
            "omni",
            "--label-substr",
            "minicpmo_4_5_omniinteract",
            "--dry-run",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    output = result.stdout + result.stderr

    assert result.returncode == 0, output
    assert "omni and local_model" in output
    assert "tests/dfx/perf/scripts/run_benchmark.py" in output
    assert "test_minicpmo_4_5_omniinteract.json" in output
    assert "run_diffusion_benchmark.py" not in output
