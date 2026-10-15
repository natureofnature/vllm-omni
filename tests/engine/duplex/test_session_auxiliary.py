# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Auxiliary execution contracts independent of a model's memory policy."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from tests.engine.duplex.test_session_runner import close_harness, open_harness
from vllm_omni.engine.duplex.commands import CreateItem, DeleteItem, Heartbeat
from vllm_omni.engine.duplex.config import DuplexSessionState
from vllm_omni.engine.duplex.contracts import DuplexAppendPlan, duplex_is_auxiliary_request
from vllm_omni.engine.duplex.plugin import (
    DuplexAuxiliaryPolicy,
    DuplexAuxiliaryRequest,
    DuplexInputCapacityError,
    duplex_history_digest,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


async def wait_until(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.001)

    await asyncio.wait_for(wait(), timeout=2)


def install_work(h, mocker, *, allow_during_response=True, retry_delays_s=()):
    """Supply a small model policy while exercising the actual Session runner."""
    work = SimpleNamespace(ready=True, revision=0, applied=[], histories=[])
    plugin = h.runner.plugin
    mocker.patch.object(plugin, "supports_auxiliary_requests", True)

    def policy(*, state, history_revision, history):
        work.histories.append(history)
        if work.ready:
            return DuplexAuxiliaryPolicy(work.revision, allow_during_response=allow_during_response)
        return None

    async def prepare(*, history, runtime_config, state):
        return DuplexAuxiliaryRequest(
            stage_id=0,
            prompt={"prompt": "private work"},
            sampling_params=h.manager.sampling_params_for(h.session)[0],
            history_length=0,
            history_digest=duplex_history_digest(()),
            model_source=work.revision,
            retry_delays_s=retry_delays_s,
        )

    def apply(*, request, output, state):
        if not output.accepted:
            return False
        work.applied.append(request.model_source)
        work.ready = False
        return True

    mocker.patch.object(plugin, "auxiliary_policy", side_effect=policy)
    mocker.patch.object(plugin, "prepare_auxiliary_request", side_effect=prepare)
    mocker.patch.object(plugin, "apply_auxiliary_result", side_effect=apply)
    mocker.patch.object(
        plugin,
        "auxiliary_source_matches",
        side_effect=lambda *, request, state, history: request.model_source == work.revision,
    )
    mocker.patch.object(
        plugin,
        "auxiliary_preempted_by_input",
        side_effect=lambda *, payload, state, has_new_user_input: has_new_user_input,
    )
    return work


def auxiliary_submissions(h):
    return [
        submission for submission in h.port.submissions if duplex_is_auxiliary_request(submission.context.request_id)
    ]


def finish_work(h, *, accepted=True):
    submission = auxiliary_submissions(h)[-1]
    h.deliver(
        SimpleNamespace(request_id=submission.context.request_id, accepted=accepted, finished=True),
        stage_id=0,
        segment_finished=True,
    )
    return submission.context.request_id


@pytest.mark.asyncio
async def test_memory_tasks_remain_session_local_and_runner_owned(mocker):
    first = await open_harness()
    second = await open_harness()
    try:
        first_work = install_work(first, mocker)
        second_work = install_work(second, mocker)
        first.submit(Heartbeat())
        second.submit(Heartbeat())
        await wait_until(lambda: len(auxiliary_submissions(first)) == len(auxiliary_submissions(second)) == 1)
        first_task = first.runner.memory._task
        second_task = second.runner.memory._task
        assert first.runner.memory is not second.runner.memory
        assert first_task in first.runner._background_tasks
        assert second_task in second.runner._background_tasks

        await close_harness(first)
        assert first_task.done()
        assert first_task not in first.runner._background_tasks
        assert not any(duplex_is_auxiliary_request(request_id) for request_id in first.session.resource_request_ids())
        first_request_id = auxiliary_submissions(first)[0].context.request_id
        assert ([first_request_id], True) in first.port.cleanups
        assert not second_task.done()
        finish_work(second)
        await wait_until(lambda: second_work.applied == [0])
        assert first_work.applied == []
        await wait_until(lambda: second_task not in second.runner._background_tasks)
    finally:
        await close_harness(first)
        await close_harness(second)


@pytest.mark.asyncio
async def test_accepted_append_wakes_concurrent_work_without_idle_gap(mocker):
    h = await open_harness(auto_response=False)
    try:
        work = install_work(h, mocker)
        work.ready = False
        mocker.patch.object(
            h.runner.plugin, "prepare_append_plan", return_value=DuplexAppendPlan(prompt={"prompt_token_ids": [1]})
        )
        mocker.patch.object(
            h.runner.plugin, "commit_append_plan", side_effect=lambda request_id: setattr(work, "ready", True)
        )
        task = await h.runner._start_append({}, final=True, precreate_response=True)
        assert await task
        await wait_until(lambda: len(auxiliary_submissions(h)) == 1)
        assert h.session.active_response_id is not None
        assert len(h.port.submissions) == 2
        assert not duplex_is_auxiliary_request(h.port.submissions[0].context.request_id)
        finish_work(h)
        await wait_until(lambda: work.applied == [0])
        assert h.session.active_response_id is not None
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_new_user_item_preempts_and_blocks_immediate_restart(mocker):
    h = await open_harness()
    try:
        install_work(h, mocker)
        h.session.begin_response()
        h.submit(Heartbeat())
        await wait_until(lambda: len(auxiliary_submissions(h)) == 1)
        request_id = auxiliary_submissions(h)[0].context.request_id
        h.submit(
            CreateItem(
                item={
                    "id": "question",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Answer this now."}],
                }
            )
        )
        await wait_until(lambda: ([request_id], True) in h.port.cleanups)
        for _ in range(3):
            await h.run(Heartbeat())
        assert h.runner.memory._interaction_pending
        assert len(auxiliary_submissions(h)) == 1
        assert h.runner.memory._failures == 0
        assert request_id not in h.session.resource_request_ids()
    finally:
        await close_harness(h)


@pytest.mark.asyncio
@pytest.mark.parametrize("item_count", [1, 2])
async def test_deleting_pending_user_items_restores_observation_work(mocker, item_count):
    h = await open_harness()
    try:
        work = install_work(h, mocker)
        h.submit(Heartbeat())
        await wait_until(lambda: len(auxiliary_submissions(h)) == 1)
        request_id = auxiliary_submissions(h)[0].context.request_id
        for index in range(item_count):
            h.submit(
                CreateItem(
                    item={
                        "id": f"question-{index}",
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Temporary question"}],
                    }
                )
            )
        await wait_until(lambda: len(h.runner.memory._pending_user_items) == item_count)
        await wait_until(lambda: ([request_id], True) in h.port.cleanups)
        for index in range(item_count):
            h.submit(DeleteItem(item_id=f"question-{index}"))
            h.submit(Heartbeat())
            await wait_until(lambda: len(h.runner.memory._pending_user_items) == item_count - index - 1)
            if index + 1 < item_count:
                assert h.runner.memory._interaction_pending
                assert len(auxiliary_submissions(h)) == 1
        await wait_until(lambda: len(auxiliary_submissions(h)) == 2)
        assert not h.runner.memory._interaction_pending
        finish_work(h)
        await wait_until(lambda: work.applied == [0])
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_consumed_user_intent_does_not_starve_following_observations(mocker):
    h = await open_harness(auto_response=False)
    try:
        work = install_work(h, mocker)
        mocker.patch.object(
            h.runner.plugin, "prepare_append_plan", return_value=DuplexAppendPlan(prompt={"prompt_token_ids": [1]})
        )
        await h.run(
            CreateItem(
                item={
                    "id": "question",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "What is visible?"}],
                }
            )
        )
        question = await h.runner._start_append({"video_frames": [b"frame"]}, final=True, precreate_response=True)
        assert await question
        await h.run(Heartbeat())
        assert not auxiliary_submissions(h)
        assert h.runner.memory._interaction_active
        assert not h.runner.memory._interaction_pending
        # Complete the interaction then immediately start the next observation:
        # no mailbox idle gap, and the historical unanswered counter remains.
        question_turn = h.session.active_response_turn_id
        assert question_turn is not None
        h.session.end_response()
        h.session.complete_model_turn(question_turn)
        assert h.session.unanswered_user_items()
        observation = await h.runner._start_append(
            {"video_frames": [b"next frame"]}, final=True, precreate_response=True
        )
        assert await observation
        await wait_until(lambda: len(auxiliary_submissions(h)) == 1)
        assert h.session.active_response_id is not None
        assert not h.runner.memory._interaction_active
        finish_work(h)
        await wait_until(lambda: work.applied == [0])
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_retries_are_bounded_by_source_despite_new_history(mocker):
    h = await open_harness()
    try:
        install_work(h, mocker, retry_delays_s=(0.005, 0.01))
        h.submit(Heartbeat())
        for attempt in range(1, 4):
            await wait_until(lambda: len(auxiliary_submissions(h)) == attempt)
            h.session.append_history_message({"role": "user", "content": ""})
            finish_work(h, accepted=False)
        await wait_until(lambda: h.runner.memory._failures == 3)
        events = await h.run(Heartbeat())
        assert len(auxiliary_submissions(h)) == 3
        assert h.runner.memory._retry_timer is None
        assert any(getattr(event, "code", None) == "auxiliary_work_paused" for event in events)
        assert not any(duplex_is_auxiliary_request(request_id) for request_id in h.session.resource_request_ids())
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_invalid_result_exception_uses_bounded_retry(mocker):
    h = await open_harness()
    try:
        install_work(h, mocker, retry_delays_s=(0,))
        apply = mocker.patch.object(h.runner.plugin, "apply_auxiliary_result", side_effect=ValueError("invalid output"))
        h.submit(Heartbeat())
        await wait_until(lambda: len(auxiliary_submissions(h)) == 1)
        finish_work(h)
        await wait_until(lambda: len(auxiliary_submissions(h)) == 2)
        assert apply.call_count == 1
        assert h.runner.memory._failures == 1
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_prepare_failure_does_not_consume_submitted_retry_budget(mocker):
    h = await open_harness()
    try:
        install_work(h, mocker, retry_delays_s=(0,))
        prepare = mocker.patch.object(
            h.runner.plugin, "prepare_auxiliary_request", side_effect=ValueError("bad source")
        )
        await h.run(Heartbeat())
        await h.run(Heartbeat())
        assert prepare.call_count == 1
        assert not h.port.submissions
        assert h.runner.memory._failures == 0
        assert h.runner.memory._retry_timer is None
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_history_edit_reconciles_while_auxiliary_is_running(mocker):
    h = await open_harness()
    try:
        work = install_work(h, mocker)
        message = {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "original"}}]}
        h.session.register_history_item("image", message)
        h.submit(Heartbeat())
        await wait_until(lambda: len(auxiliary_submissions(h)) == 1)
        snapshot = work.histories[-1]
        h.submit(Heartbeat())
        await wait_until(lambda: len(work.histories) >= 2)
        assert work.histories[-1] is snapshot
        h.session.register_history_item(
            "image", {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "edited"}}]}
        )

        def reconcile(*, state, history_revision, history):
            assert history[-1]["content"][0]["image_url"]["url"] == "edited"
            work.revision += 1
            work.ready = False
            return None

        policy = mocker.patch.object(h.runner.plugin, "auxiliary_policy", side_effect=reconcile)
        h.submit(Heartbeat())
        await wait_until(lambda: policy.call_count > 0)
        finish_work(h)
        await h.settle()
        assert not work.applied
        assert not any(duplex_is_auxiliary_request(request_id) for request_id in h.session.resource_request_ids())
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_close_cancels_pending_retry_timer(mocker):
    h = await open_harness()
    try:
        install_work(h, mocker, retry_delays_s=(60,))
        h.submit(Heartbeat())
        await wait_until(lambda: len(auxiliary_submissions(h)) == 1)
        finish_work(h, accepted=False)
        await wait_until(lambda: h.runner.memory._retry_timer is not None)
        timer = h.runner.memory._retry_timer
        await close_harness(h)
        assert timer.cancelled()
        assert not any(duplex_is_auxiliary_request(request_id) for request_id in h.session.resource_request_ids())
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_capacity_rejection_releases_request_and_allows_new_append(mocker):
    h = await open_harness(auto_response=False)
    try:
        original = h.runner.plugin.prepare_prompt_config
        reject = mocker.patch.object(
            h.runner.plugin,
            "prepare_prompt_config",
            side_effect=DuplexInputCapacityError("Memory is full", code="test_capacity"),
        )
        first = await h.runner._start_append({}, final=True, precreate_response=True)
        assert not await first
        events = await h.settle()
        assert h.session.state is DuplexSessionState.OPEN
        assert not h.port.submissions
        assert not h.session.resource_request_ids()
        assert not h.session.lease.active_operations
        assert h.runner.tasks.append_tail is None
        assert h.session.active_response_id is None
        assert any(getattr(event, "code", None) == "test_capacity" for event in events)
        assert any(event.type == "response.done" and event.status == "failed" for event in events)
        reject.side_effect = original
        mocker.patch.object(
            h.runner.plugin, "prepare_append_plan", return_value=DuplexAppendPlan(prompt={"prompt_token_ids": [1]})
        )
        second = await h.runner._start_append({}, final=True, precreate_response=True)
        assert await second
        assert len(h.port.submissions) == 1
        assert h.session.input_seq == 1
        assert h.session.state is DuplexSessionState.OPEN
    finally:
        await close_harness(h)
