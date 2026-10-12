# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import base64
import io
import threading

import pytest
from PIL import Image
from vllm.sampling_params import SamplingParams

from tests.engine.duplex.test_session_runner import SESSION_ID, Harness, RecordingStagePort, tts_output
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig, load_deploy_config, merge_pipeline_deploy
from vllm_omni.engine.duplex.commands import (
    AckPlayback,
    AppendAudio,
    CancelResponse,
    CloseSession,
    Commit,
    CreateItem,
    CreateResponse,
)
from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import DuplexFence
from vllm_omni.engine.duplex.delivery import DuplexOutputBuffer
from vllm_omni.engine.duplex.messages import DuplexControlResultMessage, OpenDuplexSessionMessage
from vllm_omni.engine.duplex.plugin import DuplexRuntimeConfigError
from vllm_omni.engine.duplex.realtime_commands import translate_realtime_command
from vllm_omni.engine.duplex.session.manager import DuplexSessionManager
from vllm_omni.model_executor.models.joyai_vl_interaction.duplex.input import JoyAIFrameBuffer
from vllm_omni.model_executor.models.joyai_vl_interaction.duplex.plugin import JoyAIDuplexPlugin
from vllm_omni.model_executor.models.joyai_vl_interaction.pipeline import JOYAI_VL_INTERACTION_PIPELINE

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class Processor:
    def apply_chat_template(self, messages, **kwargs):
        return repr(messages)

    def __call__(self, *, text, images):
        return {"input_ids": [[0] * (len(text[0]) + 256 * len(images or []))]}


def frame(color="red"):
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


async def open_joyai(*, modalities=("text", "audio")):
    plugin = JoyAIDuplexPlugin(lambda audio, *args: "AAAA")
    plugin.processor = Processor()
    plugin.max_model_len = 16384
    port = RecordingStagePort(stage_count=3)
    output: asyncio.Queue[object] = asyncio.Queue()
    results: asyncio.Queue[DuplexControlResultMessage] = asyncio.Queue()
    output_buffer = DuplexOutputBuffer(max_bytes=2 * 1024 * 1024, max_events=512)
    manager = DuplexSessionManager(
        plugin=plugin,
        stage_port=port,
        output_sink=output,
        result_sink=results,
        runtime_config=DuplexSessionRuntimeConfig(),
        model_config=None,
    )
    await manager.handle(
        OpenDuplexSessionMessage(
            control_id="open",
            session_id=SESSION_ID,
            session_config=DuplexSessionConfig(
                model="joyai", instructions="Watch the pot.", modalities=list(modalities)
            ),
            output_buffer=output_buffer,
        )
    )
    result = await results.get()
    assert result.ok, result
    harness = Harness(
        manager=manager,
        port=port,
        output=output,
        output_buffer=output_buffer,
        results=results,
        runner=manager.runners[SESSION_ID],
    )
    await harness.settle()
    return harness


async def observe(h):
    count = len(h.port.submissions)
    await h.run(AppendAudio(video_frames=(frame(),)))
    await h.run(Commit(final=True, create_response=True))
    assert len(h.port.submissions) == count + 1, [event.to_realtime() for event in h.events]
    return h.port.submissions[-1].context.request_id


def test_frame_buffer_latest_snapshot_rollback_and_clear():
    buffer = JoyAIFrameBuffer()
    buffer.prepare_append({"video_frames": ["old", "latest"]}, operation_id="a", chunk_period_ms=1000, allow_emit=True)
    assert buffer.pending_byte_count == len("latest")
    reservation = buffer.prepare_commit(operation_id="b", chunk_period_ms=1000)
    assert reservation.payload["video_frames"] == ["latest"]
    assert reservation.byte_count == len("latest")
    assert not buffer.has_pending() and buffer.has_reserved()
    reservation.rollback()
    assert buffer.has_pending() and not buffer.has_reserved()
    reservation = buffer.prepare_commit(operation_id="c", chunk_period_ms=1000)
    buffer.prepare_append({"video_frames": ["newer"]}, operation_id="d", chunk_period_ms=1000, allow_emit=False)
    reservation.rollback()
    assert buffer.frames == ["newer"]
    reservation = buffer.prepare_commit(operation_id="e", chunk_period_ms=1000)
    buffer.clear()
    reservation.rollback()
    assert buffer.pending_byte_count == 0 and not buffer.has_reserved()


def test_opt_in_profile_preserves_turn_pipeline():
    normal = load_deploy_config(get_deploy_config_path("joyai_vl_interaction.yaml"))
    duplex = load_deploy_config(get_deploy_config_path("joyai_vl_interaction_duplex.yaml"))
    assert normal.session_mode == "turn"
    assert duplex.session_mode == "duplex"
    assert normal.stages == duplex.stages
    assert normal.connectors == duplex.connectors
    assert not duplex.async_chunk
    stages = merge_pipeline_deploy(JOYAI_VL_INTERACTION_PIPELINE, duplex)
    assert len(stages) == 3
    assert all(stage.session_mode == "duplex" for stage in stages)


def test_capabilities_and_configuration_refuse_audio_and_tools():
    plugin = JoyAIDuplexPlugin(lambda *args: "")
    caps = plugin.capabilities(max_sessions=2)
    assert caps.allows_video_without_audio()
    assert caps.validate_append_modalities(has_audio=False, has_video=True) is None
    assert caps.validate_append_modalities(has_audio=True, has_video=False)
    assert caps.supports_text_only_turn and caps.supports_multi_session
    for extra in ({"realtime_tools": [{"name": "search"}]}, {"auto_response": True}):
        with pytest.raises(DuplexRuntimeConfigError):
            plugin.runtime_config_for_update(DuplexSessionConfig(extra_body=extra), {})


@pytest.mark.asyncio
async def test_silent_observation_finishes_request_and_keeps_session_open():
    h = await open_joyai()
    try:
        request_id = await observe(h)
        prompt = h.port.submissions[-1].prompt
        assert "Watch the pot." in prompt["prompt"]
        assert "audio" not in prompt["multi_modal_data"]
        assert len(prompt["multi_modal_data"]["image"]) == 1
        consumed = h.deliver(
            tts_output(request_id, text="</silence>", finished=True), stage_id=0, segment_finished=True
        )
        assert consumed, "Silent observations must stop before TTS"
        await h.settle()
        assert h.session.active_response_id is None
        assert h.port.cleanups[-1][0] == [request_id]
        next_id = await observe(h)
        assert next_id != request_id
        await h.run(CloseSession())
        assert not h.runner.plugin.data_plane.requests
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_spoken_action_forwards_once_and_audio_completes_response():
    h = await open_joyai()
    try:
        request_id = await observe(h)
        response_id = h.session.active_response_id
        consumed = h.deliver(
            tts_output(request_id, text="</response> The pot is smoking.", finished=True),
            stage_id=0,
            segment_finished=True,
        )
        assert not consumed, "Spoken actions must enter the existing TTS stage"
        await h.settle()
        assert h.session.active_response_id == response_id
        await h.deliver_and_settle(tts_output(request_id, finished=True), stage_id=2, segment_finished=True)
        await h.run(AckPlayback(response_id=response_id, played_ms=1000))
        assert h.session.active_response_id is None
        events = [event.to_realtime() for event in h.events]
        assert any(e.get("delta") == "The pot is smoking." for e in events)
        assert any(e["type"] == "response.output_audio.delta" for e in events)
        assert h.port.cleanups[-1][0] == [request_id]
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_repeated_response_is_silent_and_decision_is_pure():
    h = await open_joyai()
    try:
        h.runner.model_state.last_response_text = "The pot is smoking."
        request_id = await observe(h)
        output = tts_output(request_id, text="</response> The pot is smoking.", finished=True)
        plugin = h.runner.plugin
        decision = plugin.decide_output(
            stage_id=0,
            final_stage_id=2,
            segment_finished=True,
            segment_token_ids=(),
            segment_output_metadata={},
            output=output,
        )
        assert decision is not None
        assert h.runner.model_state.last_response_text == "The pot is smoking."
        assert h.deliver(output, stage_id=0, segment_finished=True)
        await h.settle()
        assert h.runner.model_state.last_response_text == "The pot is smoking."
        assert h.session.active_response_id is None
        request_id = await observe(h)
        assert h.deliver(
            tts_output(request_id, text="</response> The pot is smoking.", finished=True),
            stage_id=0,
            segment_finished=True,
        )
        await h.settle()
        assert h.runner.model_state.last_response_text == "The pot is smoking."
        request_id = await observe(h)
        await h.deliver_and_settle(
            tts_output(request_id, text="</silence>", finished=True), stage_id=0, segment_finished=True
        )
        assert h.runner.model_state.last_response_text == ""
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_cancel_releases_request_and_late_output_cannot_speak():
    h = await open_joyai()
    try:
        request_id = await observe(h)
        await h.deliver_and_settle(
            tts_output(request_id, text="</response> Smoke!", finished=True), stage_id=0, segment_finished=True
        )
        await h.run(CancelResponse())
        assert h.runner.plugin.data_plane.is_terminal(request_id)
        assert not any(message.get("content") == "Smoke!" for message in h.session.history)
        before = len(h.events)
        await h.deliver_and_settle(tts_output(request_id, finished=True), stage_id=2, epoch=0)
        assert not any(event.to_realtime()["type"] == "response.output_audio.delta" for event in h.events[before:])
        assert h.runner.model_state.last_response_text == ""
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["delegation", "missing_audio", "empty_audio"])
async def test_failed_output_releases_resources_and_next_observation_runs(failure):
    h = await open_joyai()
    try:
        request_id = await observe(h)
        action = "</delegation> Ask the agent" if failure == "delegation" else "</response> Smoke!"
        await h.deliver_and_settle(
            tts_output(request_id, text=action, finished=True), stage_id=0, segment_finished=True
        )
        if failure != "delegation":
            output = tts_output(request_id, finished=True)
            output.multimodal_output = {} if failure == "missing_audio" else {"audio": []}
            await h.deliver_and_settle(output, stage_id=2, segment_finished=True)
        assert h.session.active_response_id is None
        assert h.runner.plugin.data_plane.is_terminal(request_id)
        assert h.port.cleanups[-1][0] == [request_id]
        events = [event.to_realtime() for event in h.events]
        failures = [event for event in events if event["type"] == "response.done"]
        assert len(failures) == 1 and failures[0]["response"]["status"] == "failed"
        assert await observe(h) != request_id
        await h.run(CloseSession())
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_text_item_uses_same_session_without_audio():
    h = await open_joyai()
    try:
        await h.run(
            translate_realtime_command(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "id": "instruction",
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Tell me when smoke appears."}],
                    },
                }
            )
        )
        await h.run(CreateResponse())
        assert len(h.port.submissions) == 1
        prompt = h.port.submissions[0].prompt
        assert "Tell me when smoke appears." in prompt["prompt"]
        assert prompt["multi_modal_data"] == {}
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["</silence>", "</response> Repeat", "</delegation> Ask the agent"])
async def test_intermediate_terminal_allows_text_followup(action):
    h = await open_joyai()
    try:
        h.runner.model_state.last_response_text = "Repeat"
        request_id = await observe(h)
        await h.deliver_and_settle(
            tts_output(request_id, text=action, finished=True), stage_id=0, segment_finished=True
        )
        assert h.session.active_response_id is None
        assert h.runner.run.stream_request_id is None
        assert h.port.cleanups[-1][0] == [request_id]
        await h.run(
            translate_realtime_command(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "id": "followup",
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "What do you see now?"}],
                    },
                }
            )
        )
        await h.run(CreateResponse())
        assert len(h.port.submissions) == 2
        assert h.port.submissions[-1].context.request_id != request_id
        assert "What do you see now?" in h.port.submissions[-1].prompt["prompt"]
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("modalities", [("text",), ("text", "audio")])
async def test_response_history_requires_playback_only_for_audio(modalities):
    h = await open_joyai(modalities=modalities)
    try:
        await h.run(
            CreateItem(
                item={
                    "id": "question",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "What do you see?"}],
                }
            )
        )
        request_id = await observe(h)
        response_id = h.session.active_response_id
        await h.deliver_and_settle(
            tts_output(request_id, text="</response> Smoke!", finished=True), stage_id=0, segment_finished=True
        )
        await h.deliver_and_settle(tts_output(request_id, finished=True), stage_id=2, segment_finished=True)
        assert h.session.active_response_id is None
        spoken = "audio" in modalities
        assert any(message.get("content") == "Smoke!" for message in h.session.history) is not spoken
        assert (h.session.playback.sent_ms > 0) is spoken
        assert h.session.playback.text_requires_complete_audio is spoken
        if spoken:
            await h.run(AckPlayback(response_id=response_id, played_ms=1000))
            assert any(message.get("content") == "Smoke!" for message in h.session.history)
        else:
            assert not any(event.to_realtime()["type"] == "response.output_audio.delta" for event in h.events)
        next_request = await observe(h)
        assert next_request != request_id
        assert "</response> Smoke!" in h.port.submissions[-1].prompt["prompt"]
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_camera_input_during_output_waits_for_previous_turn():
    h = await open_joyai()
    try:
        request_id = await observe(h)
        await h.run(AppendAudio(video_frames=(frame("blue"),)))
        await h.run(Commit(final=True, create_response=True))
        assert len(h.port.submissions) == 1
        await h.deliver_and_settle(
            tts_output(request_id, text="</silence>", finished=True), stage_id=0, segment_finished=True
        )
        assert len(h.port.submissions) == 2
        latest = h.port.submissions[-1]
        assert latest.context.request_id != request_id
        assert latest.prompt["multi_modal_data"]["image"][-1].getpixel((0, 0)) == (0, 0, 255)
        await h.run(CloseSession())
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_cancel_during_prompt_preparation_cannot_submit(monkeypatch):
    h = await open_joyai()
    entered, released = threading.Event(), threading.Event()
    original = h.runner.plugin.plan_append

    def blocked_plan(**kwargs):
        entered.set()
        assert released.wait(timeout=5)
        return original(**kwargs)

    monkeypatch.setattr(h.runner.plugin, "plan_append", blocked_plan)
    try:
        h.submit(AppendAudio(video_frames=(frame(),)))
        h.submit(Commit(final=True, create_response=True))
        assert await asyncio.to_thread(entered.wait, 3)
        await h.run(CancelResponse())
        released.set()
        await h.settle()
        assert not h.port.submissions
        assert not h.runner.plugin.data_plane.requests
    finally:
        released.set()
        await h.run(CloseSession())
        await h.manager.shutdown()


def test_conversation_images_and_camera_frames_share_input_order():
    plugin = JoyAIDuplexPlugin(lambda *args: "")
    plugin.processor = Processor()
    plugin.max_model_len = 16384
    state = plugin.create_session_state()
    conversation: list[dict[str, object]] = []
    config = {"conversation": conversation}
    runtime = plugin.runtime_config_for_update(DuplexSessionConfig(), {})

    def prepare(frames=()):
        payload = {"video_frames": list(frames)}
        prompt_config = plugin.prepare_prompt_config(config, state=state, payload=payload)
        plan = plugin.plan_append(
            request_id="r",
            fence=DuplexFence("s"),
            session_config=prompt_config,
            runtime_config=runtime,
            seq=1,
            turn_seq=1,
            payload=payload,
            final=True,
            sampling_params=SamplingParams(max_tokens=64),
        )
        return [image.getpixel((0, 0)) for image in plan.prompt["multi_modal_data"]["image"]]

    red, blue, green = (255, 0, 0), (0, 0, 255), (0, 128, 0)
    assert prepare([frame()] * 4) == [red] * 4
    image_message: dict[str, object] = {
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": frame("blue")}}],
    }
    config["conversation"].append(image_message)
    assert prepare() == [red, red, red, blue]
    assert prepare([frame("green")]) == [red, red, blue, green]
    config["conversation"].append({"role": "user", "content": "What changed?"})
    assert prepare() == [red, red, blue, green]
    config["conversation"].append(image_message.copy())
    assert prepare() == [red, blue, green, blue]
    config["conversation"] = [{"role": "user", "content": "New scene"}]
    assert prepare([frame("green")]) == [green]


def test_prompt_budget_counts_visual_tokens_and_keeps_instruction():
    plugin = JoyAIDuplexPlugin(lambda *args: "")
    plugin.processor = Processor()
    plugin.max_model_len = 1500
    config = DuplexSessionConfig(instructions="Watch the pot.")
    runtime = plugin.runtime_config_for_update(config, {})
    kwargs = dict(
        request_id="r",
        fence=DuplexFence("s"),
        runtime_config=runtime,
        seq=1,
        turn_seq=1,
        payload={"video_frames": [frame()]},
        final=True,
        sampling_params=SamplingParams(max_tokens=64),
    )
    session_config = {
        "joyai_frames": [frame()] * 3 + [frame("blue")],
        "conversation": [
            {"role": "user", "content": "old" * 1000},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "Any smoke?"},
        ],
    }
    plan = plugin.plan_append(session_config=session_config, **kwargs)
    prompt = plan.prompt
    assert "Watch the pot." in prompt["prompt"] and "Any smoke?" in prompt["prompt"]
    assert "old answer" not in prompt["prompt"]
    assert prompt["multi_modal_data"]["image"][-1].getpixel((0, 0)) == (0, 0, 255)
    assert (
        len(plugin.processor(text=[prompt["prompt"]], images=prompt["multi_modal_data"]["image"])["input_ids"][0])
        <= 1436
    )
    plugin.max_model_len = 10
    with pytest.raises(DuplexRuntimeConfigError, match="context budget"):
        plugin.plan_append(session_config=session_config, **kwargs)
