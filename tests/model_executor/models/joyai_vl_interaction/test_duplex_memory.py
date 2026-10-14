# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tests.model_executor.models.joyai_vl_interaction.test_duplex_plugin import open_joyai
from vllm_omni.engine.duplex.commands import Heartbeat
from vllm_omni.engine.duplex.plugin import duplex_history_digest, duplex_text_history

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def committed_turns(session, count=6):
    for index in range(count):
        for role, text in (("user", f"Remember cabinet {index}: K731"), ("assistant", f"Cabinet {index} recorded.")):
            message = {"role": role, "content": text}
            session.append_history_message(message)
            session.register_history_item(f"{role}-{index}", message)


async def start_summary(h):
    h.submit(Heartbeat())

    async def wait_for_submission():
        while not h.port.submissions:
            await asyncio.sleep(0.005)

    await asyncio.wait_for(wait_for_submission(), timeout=2)
    return h.port.submissions[-1]


def summary_output(request_id, text="Cabinet 0 has code K731.", finish_reason="stop"):
    return SimpleNamespace(
        request_id=request_id,
        finished=True,
        outputs=[SimpleNamespace(text=text, finish_reason=finish_reason)],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stage_id,stage_count", [(0, 3), (1, 4)])
async def test_summary_uses_private_same_stage_request_and_preserves_new_tail(stage_id, stage_count):
    h = await open_joyai(modalities=("text",))
    try:
        h.port._stage_count = stage_count
        h.runner.plugin.joyai_stage_id = stage_id
        committed_turns(h.session)
        initial_seq = h.session.input_seq
        submission = await start_summary(h)
        context = submission.context
        assert context.stage_id == context.final_stage_id == stage_id
        assert not submission.resumable and not submission.already_submitted
        assert "auxiliary-" in context.request_id
        assert context.request_id not in h.runner.plugin.data_plane.requests
        # New observations must not starve consolidation of an unchanged prefix.
        h.session.append_history_message({"role": "user", "content": "Look at the window now."})
        h.deliver(summary_output(context.request_id), stage_id=stage_id, segment_finished=True)
        events = await h.settle()
        assert h.runner.model_state.memory.text == "Cabinet 0 has code K731."
        assert h.runner.model_state.memory.covered_messages == 4
        assert h.session.history[-1]["content"] == "Look at the window now."
        assert h.session.active_response_id is None
        assert h.session.input_seq == initial_seq
        assert not any(event.to_realtime()["type"].startswith("response.") for event in events)
        assert h.runner.model_state.last_response_text == ""
        assert ([context.request_id], True) in h.port.cleanups
        assert context.request_id not in h.session.resource_request_ids()
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["edit", "delete", "epoch", "instructions"])
async def test_summary_rejects_changed_source_or_context(mutation):
    h = await open_joyai(modalities=("text",))
    try:
        committed_turns(h.session)
        submission = await start_summary(h)
        if mutation == "edit":
            h.session.register_history_item("assistant-0", {"role": "assistant", "content": "Only the played prefix."})
        elif mutation == "delete":
            h.session.delete_history_item("user-0")
        elif mutation == "instructions":
            h.session.replace_runtime_config({**h.session.runtime_config, "instructions": "New instructions"})
        else:
            h.session.epoch += 1
        h.deliver(summary_output(submission.context.request_id), stage_id=0, segment_finished=True)
        await h.settle()
        assert h.runner.model_state.memory.text == ""
        assert not any(event.to_realtime()["type"].startswith("response.") for event in h.events)
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["error", "timeout", "empty", "truncated"])
async def test_failed_summary_keeps_memory_and_does_not_retry_same_prefix(failure):
    h = await open_joyai(modalities=("text",))
    try:
        committed_turns(h.session)
        memory = h.runner.model_state.memory
        memory.text = "Earlier cabinet fact."
        memory.covered_messages = 2
        memory.covered_digest = duplex_history_digest(duplex_text_history(h.session.history)[:2])
        if failure == "timeout":
            original = h.runner.plugin.prepare_auxiliary_request

            async def short_deadline(**kwargs):
                request = await original(**kwargs)
                return replace(request, timeout_s=0.02) if request is not None else None

            h.runner.plugin.prepare_auxiliary_request = short_deadline
        submission = await start_summary(h)
        request_id = submission.context.request_id
        if failure == "error":
            h.runner.on_stage_failure(0, RuntimeError("synthetic failure"), request_id=request_id)
        elif failure != "timeout":
            h.deliver(
                summary_output(request_id, text="" if failure == "empty" else "partial", finish_reason="length" if failure == "truncated" else "stop"),
                stage_id=0,
                segment_finished=True,
            )
        await h.settle()
        assert memory.text == "Earlier cabinet fact."
        await h.run(Heartbeat())
        assert len(h.port.submissions) == 1
        assert not any(event.to_realtime()["type"].startswith("response.") for event in h.events)
        assert ([request_id], True) in h.port.cleanups
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_close_cancels_auxiliary_and_late_result_cannot_write():
    h = await open_joyai(modalities=("text",))
    committed_turns(h.session)
    submission = await start_summary(h)
    task = h.runner._auxiliary_task
    await h.manager.shutdown()
    assert task.done()
    assert ([submission.context.request_id], True) in h.port.cleanups
    assert h.deliver(summary_output(submission.context.request_id), stage_id=0, segment_finished=True)
    assert h.runner.model_state.memory.text == ""


@pytest.mark.asyncio
async def test_memory_is_bounded_and_instructions_are_outside_summary():
    h = await open_joyai(modalities=("text",))
    try:
        committed_turns(h.session, count=20)
        plugin, state = h.runner.plugin, h.runner.model_state
        request = await plugin.prepare_auxiliary_request(
            history=duplex_text_history(h.session.history), runtime_config=h.session.runtime_config, state=state
        )
        assert request.history_length == 8
        assert request.sampling_params.max_tokens == 512
        assert "Watch the pot." not in request.prompt["prompt"]
        plugin.apply_auxiliary_result(request=request, output=summary_output("private"), state=state)
        prepared = plugin.prepare_prompt_config({"conversation": list(h.session.history)}, state=state, payload={})
        assert prepared["joyai_summary"] == "Cabinet 0 has code K731."
        h.session.delete_history_item("user-0")
        prepared = plugin.prepare_prompt_config({"conversation": list(h.session.history)}, state=state, payload={})
        assert prepared["joyai_summary"] == ""
    finally:
        await h.manager.shutdown()
