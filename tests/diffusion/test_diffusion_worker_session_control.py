# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _worker_with_runner(runner):
    worker = object.__new__(DiffusionWorker)
    worker.model_runner = runner
    return worker


def test_session_control_calls_supported_model_runner_hook() -> None:
    reset_calls: list[str] = []
    runner = SimpleNamespace(reset_session=reset_calls.append)
    worker = _worker_with_runner(runner)

    result = worker.handle_session_control("reset", "session-1")

    assert result == {
        "supported": True,
        "action": "reset",
        "session_id": "session-1",
    }
    assert reset_calls == ["session-1"]


def test_session_control_reports_unsupported_runner_and_action() -> None:
    worker = _worker_with_runner(SimpleNamespace())

    assert worker.handle_session_control("close", "session-1") == {
        "supported": False,
        "action": "close",
        "session_id": "session-1",
    }
    assert worker.handle_session_control("advance", "session-1") == {
        "supported": False,
        "action": "advance",
        "session_id": "session-1",
        "error": "unsupported session action: advance",
    }
