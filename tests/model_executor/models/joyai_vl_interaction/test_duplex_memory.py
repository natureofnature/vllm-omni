# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
from dataclasses import replace

import pytest
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import SamplingParams

from tests.engine.duplex.test_session_runner import tts_output
from tests.model_executor.models.joyai_vl_interaction.test_duplex_plugin import observe, open_joyai
from vllm_omni.engine.duplex.commands import AckPlayback, CancelInput, Heartbeat
from vllm_omni.engine.duplex.contracts import DuplexFence
from vllm_omni.engine.duplex.messages import CloseDuplexSessionMessage
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
    return RequestOutput(
        request_id=request_id,
        prompt="",
        prompt_token_ids=[],
        prompt_logprobs=None,
        finished=True,
        outputs=[
            CompletionOutput(
                index=0,
                text=text,
                token_ids=[],
                cumulative_logprob=0.0,
                logprobs=None,
                finish_reason=finish_reason,
                stop_reason=None,
            )
        ],
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
                summary_output(
                    request_id,
                    text="" if failure == "empty" else "partial",
                    finish_reason="length" if failure == "truncated" else "stop",
                ),
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
async def test_auxiliary_failure_does_not_finish_concurrent_response():
    h = await open_joyai(modalities=("text",))
    try:
        committed_turns(h.session)
        submission = await start_summary(h)
        response_id = h.session.begin_response()
        h.runner.on_stage_failure(0, RuntimeError("synthetic failure"), request_id=submission.context.request_id)
        events = await h.settle()
        assert h.session.active_response_id == response_id
        assert not any(event.to_realtime()["type"] in {"response.done", "error"} for event in events)
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_summary_excludes_unplayed_speech_and_rejects_late_full_ack():
    h = await open_joyai()
    try:
        h.session.append_history_message({"role": "user", "content": "Observe the cabinet."})
        request_id = await observe(h)
        response_id = h.session.active_response_id
        await h.deliver_and_settle(
            tts_output(request_id, text="</response> Unheard secret is Z984.", finished=True),
            stage_id=0,
            segment_finished=True,
        )
        await h.deliver_and_settle(tts_output(request_id, finished=True), stage_id=2, segment_finished=True)
        committed_turns(h.session)
        # The normal observation is already recorded in the stage-port harness.
        h.port.submissions.clear()
        submission = await start_summary(h)
        assert "Unheard secret" not in submission.prompt["prompt"]
        h.submit(AckPlayback(response_id=response_id, played_ms=1000))
        await h.settle(timeout_s=0.1)
        assert any("Unheard secret" in str(message.get("content")) for message in h.session.history)
        h.deliver(summary_output(submission.context.request_id), stage_id=0, segment_finished=True)
        await h.settle()
        assert h.runner.model_state.memory.text == ""
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_failed_auxiliary_cleanup_retains_one_resource_and_blocks_more_jobs():
    h = await open_joyai(modalities=("text",))
    original_cleanup = h.port.cleanup
    try:
        committed_turns(h.session)
        submission = await start_summary(h)

        async def fail_cleanup(*args, **kwargs):
            raise RuntimeError("synthetic cleanup failure")

        h.port.cleanup = fail_cleanup
        h.deliver(summary_output(submission.context.request_id), stage_id=0, segment_finished=True)
        await h.settle()
        assert submission.context.request_id in h.session.resource_request_ids(submitted=True)
        committed_turns(h.session, count=2)
        await h.run(Heartbeat())
        assert len(h.port.submissions) == 1
    finally:
        h.port.cleanup = original_cleanup
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_close", [False, True])
async def test_input_cancel_retains_auxiliary_after_failed_aborts_until_close_retry(fail_close):
    h = await open_joyai(modalities=("text",))
    original_cleanup = h.port.cleanup
    failures_remaining = 2

    async def fail_cleanup(request_ids: list[str], *, abort: bool = False) -> None:
        nonlocal failures_remaining
        await original_cleanup(request_ids, abort=abort)
        if failures_remaining:
            failures_remaining -= 1
            raise RuntimeError("synthetic abort failure")

    try:
        committed_turns(h.session)
        submission = await start_summary(h)
        request_id = submission.context.request_id
        cancelled_request_ids = h.session.resource_request_ids()
        assert h.session.resource_request_ids(submitted=True) == [request_id]
        old_epoch = h.session.epoch
        task = h.runner._auxiliary_task
        h.port.cleanup = fail_cleanup

        events = await h.run(CancelInput())
        assert any(event.to_realtime()["type"] == "input_audio_buffer.cleared" for event in events)
        assert h.session.epoch == old_epoch + 1
        assert task is not None and task.done()
        # Open reserves the normal stage-0 request without submitting it. The
        # cancellation batch includes it; the auxiliary's retry contains only
        # the submitted summary request.
        assert h.port.cleanups == [(cancelled_request_ids, True), ([request_id], True)]
        assert failures_remaining == 0
        assert h.session.resource_request_ids(submitted=True) == [request_id]
        await h.run(Heartbeat())
        assert len(h.port.submissions) == 1

        failures_remaining = int(fail_close)
        await h.manager.close(CloseDuplexSessionMessage(control_id="close-memory", session_id=h.session.session_id))
        assert [abort for ids, abort in h.port.cleanups if request_id in ids] == [True] * 3
        if fail_close:
            assert h.manager._closing[h.session.session_id].submitted_request_ids == (request_id,)
            assert await h.manager.reap_expired() == 1
            assert [abort for ids, abort in h.port.cleanups if request_id in ids] == [True] * 4
        assert h.session.session_id not in h.manager._closing
    finally:
        h.port.cleanup = original_cleanup
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_input_cancel_cleans_auxiliary_only_once_when_abort_succeeds():
    h = await open_joyai(modalities=("text",))
    try:
        committed_turns(h.session)
        submission = await start_summary(h)
        request_id = submission.context.request_id
        cancelled_request_ids = h.session.resource_request_ids()
        await h.run(CancelInput())
        assert h.port.cleanups == [(cancelled_request_ids, True)]
        assert request_id not in h.session.resource_request_ids()
        assert h.runner.model_state.memory.text == ""
    finally:
        await h.manager.shutdown()


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


@pytest.mark.asyncio
@pytest.mark.parametrize("part_transcript", [True, False])
async def test_spoken_instruction_is_summarized_once_and_correction_invalidates(part_transcript):
    h = await open_joyai(modalities=("text",))
    try:
        committed_turns(h.session)
        part = {"type": "audio_url", "audio_url": {"url": "native-duplex:input-audio"}}
        if part_transcript:
            part["transcript"] = "Watch the red cabinet."
        spoken = {"role": "user", "content": [part], "transcript": "Watch the red cabinet."}
        h.session.register_history_item("user-0", spoken)
        submission = await start_summary(h)
        assert submission.prompt["prompt"].count("Watch the red cabinet.") == 1
        assert "native-duplex:input-audio" not in submission.prompt["prompt"]
        h.deliver(summary_output(submission.context.request_id), stage_id=0, segment_finished=True)
        await h.settle()
        assert h.runner.model_state.memory.text
        corrected = {"role": "user", "content": [{**part, "transcript": "Watch the blue cabinet."}]}
        h.session.register_history_item("user-0", corrected)
        config = h.runner.plugin.prepare_prompt_config(
            {"conversation": list(h.session.history)}, state=h.runner.model_state, payload={}
        )
        assert config["joyai_summary"] == ""
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_imported_history_catches_up_without_omitting_unsummarized_middle():
    h = await open_joyai(modalities=("text",))
    try:
        committed_turns(h.session, count=12)
        first = await start_summary(h)
        assert "Cabinet 3 recorded." in first.prompt["prompt"]
        h.deliver(summary_output(first.context.request_id), stage_id=0, segment_finished=True)
        await h.settle()
        assert len(h.port.submissions) == 2
        second = h.port.submissions[-1]
        assert "Cabinet 0 has code K731." in second.prompt["prompt"]
        assert "Cabinet 7 recorded." in second.prompt["prompt"]
        config = h.runner.plugin.prepare_prompt_config(
            {"conversation": list(h.session.history)}, state=h.runner.model_state, payload={}
        )
        prompt = h.runner.plugin.plan_append(
            request_id="normal",
            fence=DuplexFence("s"),
            session_config=config,
            runtime_config=h.session.runtime_config,
            seq=1,
            turn_seq=1,
            payload={},
            final=True,
            sampling_params=SamplingParams(max_tokens=64),
        ).prompt["prompt"]
        assert "Cabinet 0 has code K731." in prompt
        assert "Cabinet 4 recorded." in prompt
        assert "Cabinet 11 recorded." in prompt
        h.deliver(
            summary_output(second.context.request_id, "Cabinets 0 through 7 recorded."),
            stage_id=0,
            segment_finished=True,
        )
        await h.settle()
        assert h.runner.model_state.memory.covered_messages == 16
        assert len(h.port.submissions) == 2
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_unchanged_short_history_does_not_schedule_repeated_preparation(mocker):
    h = await open_joyai(modalities=("text",))
    try:
        committed_turns(h.session, count=2)
        prepare = mocker.spy(h.runner.plugin, "prepare_auxiliary_request")
        await h.run(Heartbeat())
        first_count = prepare.call_count
        assert first_count == 1
        await h.run(Heartbeat())
        assert prepare.call_count == first_count
        assert h.port.submissions == []
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_queued_completion_keeps_identity_when_earlier_command_appends_tail(monkeypatch):
    h = await open_joyai(modalities=("text",))
    release = asyncio.Event()
    try:
        committed_turns(h.session)
        submission = await start_summary(h)
        task = h.runner._auxiliary_task
        entered = asyncio.Event()
        original = h.runner._on_command

        async def hold_command(command):
            entered.set()
            await release.wait()
            await original(command)

        monkeypatch.setattr(h.runner, "_on_command", hold_command)
        h.deliver(summary_output(submission.context.request_id), stage_id=0, segment_finished=True)
        h.submit(Heartbeat())
        await asyncio.wait_for(entered.wait(), timeout=2)
        await asyncio.wait_for(task, timeout=2)
        h.session.append_history_message({"role": "user", "content": "New observation."})
        release.set()
        await h.settle()
        assert h.runner.model_state.memory.text == "Cabinet 0 has code K731."
    finally:
        release.set()
        await h.manager.shutdown()
