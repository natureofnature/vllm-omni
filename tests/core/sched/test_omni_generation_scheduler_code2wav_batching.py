# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Tests for adaptive Code2Wav request gathering."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import vllm_omni  # noqa: F401 - apply vLLM patches before scheduler import
from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _scheduler(
    ready_count: int,
    *,
    chunk_seq: int = 2,
    batch_enabled: bool = True,
) -> OmniGenerationScheduler:
    scheduler = OmniGenerationScheduler.__new__(OmniGenerationScheduler)
    scheduler._code2wav_batch_enabled = batch_enabled
    scheduler.chunk_transfer_adapter = SimpleNamespace(receives_chunks=True)
    scheduler._code2wav_batch_target_size = 8
    scheduler._code2wav_initial_batch_size = 4
    scheduler._code2wav_batch_wait_s = 0.1
    scheduler._code2wav_batch_quiet_s = 0.025
    scheduler._code2wav_batch_wait_started = None
    scheduler._code2wav_batch_last_ready_count = 0
    scheduler._code2wav_batch_last_progress = None
    scheduler.running = [
        SimpleNamespace(
            request_id=f"request-{index}",
            prompt_token_ids=[],
            num_computed_tokens=0,
            additional_information={"meta": {"chunk_seq": chunk_seq}},
        )
        for index in range(8)
    ]
    scheduler.waiting = []
    scheduler.requests = {request.request_id: request for request in scheduler.running}
    _set_ready(scheduler, ready_count)
    return scheduler


def _set_ready(scheduler: OmniGenerationScheduler, ready_count: int) -> None:
    for index, request in enumerate(scheduler.running):
        request.prompt_token_ids = [1] if index < ready_count else []


def _set_chunk_seq(scheduler: OmniGenerationScheduler, chunk_seq: int) -> None:
    for request in scheduler.running:
        request.additional_information["meta"]["chunk_seq"] = chunk_seq


def test_code2wav_batch_wait_releases_when_target_is_ready() -> None:
    scheduler = _scheduler(2)

    assert scheduler._should_wait_for_code2wav_batch(1.0)
    _set_ready(scheduler, 4)
    assert scheduler._should_wait_for_code2wav_batch(1.02)
    _set_ready(scheduler, 7)
    assert scheduler._should_wait_for_code2wav_batch(1.04)
    _set_ready(scheduler, 8)
    assert not scheduler._should_wait_for_code2wav_batch(1.05)
    assert scheduler._code2wav_batch_wait_started is None


def test_code2wav_batch_wait_releases_after_quiet_window() -> None:
    scheduler = _scheduler(3)

    assert scheduler._should_wait_for_code2wav_batch(2.0)
    assert scheduler._should_wait_for_code2wav_batch(2.024)
    assert not scheduler._should_wait_for_code2wav_batch(2.026)


def test_code2wav_batch_wait_honors_total_window_during_progress() -> None:
    scheduler = _scheduler(1)

    assert scheduler._should_wait_for_code2wav_batch(3.0)
    for ready_count, now in ((2, 3.02), (3, 3.04), (4, 3.06), (5, 3.08)):
        _set_ready(scheduler, ready_count)
        assert scheduler._should_wait_for_code2wav_batch(now)
    _set_ready(scheduler, 6)
    assert not scheduler._should_wait_for_code2wav_batch(3.101)


def test_code2wav_batch_wait_resets_without_ready_work() -> None:
    scheduler = _scheduler(2)

    assert scheduler._should_wait_for_code2wav_batch(4.0)
    _set_ready(scheduler, 0)
    assert not scheduler._should_wait_for_code2wav_batch(4.01)
    assert scheduler._code2wav_batch_wait_started is None
    assert scheduler._code2wav_batch_last_ready_count == 0
    assert scheduler._code2wav_batch_last_progress is None


def test_code2wav_batch_wait_uses_smaller_initial_target() -> None:
    scheduler = _scheduler(3, chunk_seq=1)

    assert scheduler._should_wait_for_code2wav_batch(5.0)
    _set_ready(scheduler, 4)
    assert not scheduler._should_wait_for_code2wav_batch(5.01)
    assert scheduler._code2wav_batch_wait_started is None

    _set_chunk_seq(scheduler, 2)
    assert scheduler._should_wait_for_code2wav_batch(5.02)

    _set_chunk_seq(scheduler, 0)
    assert not scheduler._should_wait_for_code2wav_batch(5.03)


def test_code2wav_initial_batch_waits_for_late_request_admission() -> None:
    scheduler = _scheduler(1, chunk_seq=0)
    scheduler.running = scheduler.running[:1]
    scheduler.requests = {request.request_id: request for request in scheduler.running}

    assert scheduler._should_wait_for_code2wav_batch(6.0)
    # Initial admission ignores the steady-state quiet window.
    assert scheduler._should_wait_for_code2wav_batch(6.03)
    assert not scheduler._should_wait_for_code2wav_batch(6.101)


def test_code2wav_steady_batch_does_not_wait_for_unregistered_requests() -> None:
    scheduler = _scheduler(1, chunk_seq=2)
    scheduler.running = scheduler.running[:1]
    scheduler.requests = {request.request_id: request for request in scheduler.running}

    assert not scheduler._should_wait_for_code2wav_batch(7.0)
    assert scheduler._code2wav_batch_wait_started is None


def test_code2wav_mixed_wave_uses_steady_quiet_window() -> None:
    scheduler = _scheduler(4, chunk_seq=2)
    scheduler.running[0].additional_information["meta"]["chunk_seq"] = 0

    assert scheduler._should_wait_for_code2wav_batch(8.0)
    assert scheduler._should_wait_for_code2wav_batch(8.024)
    assert not scheduler._should_wait_for_code2wav_batch(8.026)


def test_code2wav_batch_wait_is_disabled_outside_code2wav_stage() -> None:
    scheduler = _scheduler(3, batch_enabled=False)

    assert not scheduler._should_wait_for_code2wav_batch(6.0)


@pytest.mark.parametrize("queue_name", ["running", "waiting"])
def test_code2wav_batch_wait_ignores_untracked_queue_entries(queue_name: str) -> None:
    scheduler = _scheduler(0, chunk_seq=0)
    zombie = scheduler.running.pop()
    zombie.prompt_token_ids = [1]
    getattr(scheduler, queue_name).append(zombie)
    scheduler.requests.pop(zombie.request_id)

    assert not scheduler._should_wait_for_code2wav_batch(9.0)
