# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Visual payload and lifecycle regressions; mocked text is not a quality eval."""

import asyncio

import pytest
from PIL import Image

from tests.engine.duplex.test_session_runner import tts_output
from tests.model_executor.models.joyai_vl_interaction.test_duplex_memory import summary_output
from tests.model_executor.models.joyai_vl_interaction.test_duplex_plugin import frame, open_joyai
from vllm_omni.engine.duplex.commands import AppendAudio, Commit, CreateResponse, DeleteItem, Heartbeat
from vllm_omni.engine.duplex.plugin import duplex_text_history
from vllm_omni.engine.duplex.realtime_commands import translate_realtime_command

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def submissions(h, *, auxiliary):
    return [item for item in h.port.submissions if ("auxiliary-" in item.context.request_id) == auxiliary]


async def settle(h):
    # Auxiliary requests deliberately wait for test-controlled completions.
    return await h.settle(idle_s=0.01, timeout_s=0.15)


async def command(h, value):
    h.submit(value)
    return await settle(h)


async def wait_auxiliary(h, count=1):
    async def wait():
        while len(submissions(h, auxiliary=True)) < count:
            await asyncio.sleep(0.005)

    await asyncio.wait_for(wait(), timeout=2)
    return submissions(h, auxiliary=True)[count - 1]


async def silence(h, submission):
    assert h.deliver(
        tts_output(submission.context.request_id, text="</silence>", finished=True),
        stage_id=submission.context.stage_id,
        segment_finished=True,
    )
    await settle(h)


async def camera(h, color="red"):
    before = len(submissions(h, auxiliary=False))
    await command(h, AppendAudio(video_frames=(frame(color),)))
    await command(h, Commit(final=True, create_response=True))
    observations = submissions(h, auxiliary=False)
    assert len(observations) == before + 1
    submission = observations[-1]
    await silence(h, submission)
    return submission


async def finish_summary(h, submission, text):
    assert h.deliver(
        summary_output(submission.context.request_id, text),
        stage_id=submission.context.stage_id,
        segment_finished=True,
    )
    return await settle(h)


def user_item(item_id, content):
    return translate_realtime_command(
        {
            "type": "conversation.item.create",
            "item": {"id": item_id, "type": "message", "role": "user", "content": content},
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stage_id,stage_count", [(0, 3), (1, 4)])
async def test_silent_frames_submit_timestamped_images_to_selected_private_stage(stage_id, stage_count):
    h = await open_joyai(modalities=("text",))
    try:
        # Exercise the auxiliary stage-selection contract. Observations still
        # use the base JoyAI stage 0; this is not an ASR-profile integration test.
        h.port._stage_count = stage_count
        h.runner.plugin.joyai_stage_id = stage_id
        memory = h.runner.model_state.visual_memory
        memory.chunk_frames = 4
        for color in ("red", "blue", "green", "yellow"):
            await camera(h, color)
        auxiliary = await wait_auxiliary(h)
        context = auxiliary.context
        assert context.stage_id == context.final_stage_id == stage_id
        assert not auxiliary.resumable and not auxiliary.already_submitted
        assert context.request_id not in h.runner.plugin.data_plane.requests
        images = auxiliary.prompt["multi_modal_data"]["image"]
        assert len(images) == 4 and all(isinstance(image, Image.Image) for image in images)
        assert [image.getpixel((0, 0)) for image in images] == [(255, 0, 0), (0, 0, 255), (0, 128, 0), (255, 255, 0)]
        assert all(image is entry.image for image, entry in zip(images, memory.pending))
        for entry in memory.pending:
            assert f"<{entry.time_s:.3f}s>" in auxiliary.prompt["prompt"]
        events = await finish_summary(h, auxiliary, "A red object preceded blue, green, and yellow objects.")
        assert not memory.pending and len(memory.mid_term) == 1
        assert not any(event.to_realtime()["type"].startswith("response.") for event in events)
        assert len(h.port.submissions) == 5, "Private summaries must not create TTS-stage requests"
        assert ([context.request_id], True) in h.port.cleanups
        assert context.request_id not in h.session.resource_request_ids()
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_new_camera_tail_survives_mid_summary_and_long_term_rollover():
    h = await open_joyai(modalities=("text",))
    try:
        memory = h.runner.model_state.visual_memory
        memory.chunk_frames = 2
        memory.mid_group = 2
        await camera(h, "red")
        await camera(h, "blue")
        first = await wait_auxiliary(h)
        first_ids = [entry.id for entry in memory.pending]
        await camera(h, "green")
        tail = memory.pending[-1]
        assert ([first.context.request_id], True) not in h.port.cleanups
        await finish_summary(h, first, "A red cup was followed by a blue cup.")
        assert memory.pending == [tail]
        assert (memory.mid_term[0].first_id, memory.mid_term[0].last_id) == tuple(first_ids)
        await camera(h, "yellow")
        second = await wait_auxiliary(h, 2)
        await finish_summary(h, second, "A green cup was followed by a yellow cup.")
        long_request = await wait_auxiliary(h, 3)
        assert long_request.prompt["multi_modal_data"] == {}
        assert "red cup" in long_request.prompt["prompt"] and "green cup" in long_request.prompt["prompt"]
        assert not memory.pending and len(memory.mid_term) == 2
        await finish_summary(h, long_request, "Cups appeared in red, blue, green, and yellow order.")
        assert not memory.mid_term and len(memory.long_term) == 1
        assert (memory.long_term[0].first_id, memory.long_term[0].last_id) == (0, 3)
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["inflight", "mid", "long"])
async def test_deleting_conversation_image_revokes_visual_memory_without_new_append(phase):
    h = await open_joyai(modalities=("text",))
    try:
        memory = h.runner.model_state.visual_memory
        memory.chunk_frames = 1
        memory.mid_group = 2
        await command(
            h, user_item("picture", [{"type": "input_image", "image_url": "data:image/png;base64," + frame()}])
        )
        await command(h, CreateResponse())
        await silence(h, submissions(h, auxiliary=False)[-1])
        first = await wait_auxiliary(h)
        if phase != "inflight":
            await finish_summary(h, first, "A red cup is on the table.")
            assert memory.mid_term
        if phase == "long":
            await camera(h, "blue")
            await finish_summary(h, await wait_auxiliary(h, 2), "A blue cup replaced it.")
            await finish_summary(h, await wait_auxiliary(h, 3), "The red cup was replaced by a blue cup.")
            assert memory.long_term
        generation = memory.generation
        observation_count = len(submissions(h, auxiliary=False))
        await command(h, DeleteItem(item_id="picture"))
        await command(h, Heartbeat())
        if phase == "inflight":
            h.deliver(
                summary_output(first.context.request_id, "Stale red cup memory."),
                stage_id=first.context.stage_id,
                segment_finished=True,
            )
            await settle(h)
        assert memory.generation > generation
        assert not memory.pending and not memory.mid_term and not memory.long_term
        assert len(submissions(h, auxiliary=False)) == observation_count
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_delayed_question_keeps_early_unsummarized_image_beyond_four_frames():
    h = await open_joyai(modalities=("text",))
    try:
        h.runner.model_state.visual_memory.chunk_frames = 100
        for color in ("red", "blue", "blue", "blue", "blue"):
            await camera(h, color)
        assert not submissions(h, auxiliary=True)
        await command(h, user_item("question", [{"type": "input_text", "text": "What was the first color?"}]))
        await command(h, CreateResponse())
        prompt = submissions(h, auxiliary=False)[-1].prompt
        assert "What was the first color?" in prompt["prompt"]
        images = prompt["multi_modal_data"]["image"]
        assert len(images) == 5
        assert images[0].getpixel((0, 0)) == (255, 0, 0)
        assert all(image.getpixel((0, 0)) == (0, 0, 255) for image in images[1:])
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", ["count", "bytes"])
async def test_full_visual_memory_rejects_frame_but_next_text_request_still_runs(limit):
    h = await open_joyai(modalities=("text",))
    try:
        memory = h.runner.model_state.visual_memory
        memory.chunk_frames = 100
        if limit == "count":
            memory.max_pending = 2
        else:
            memory.max_pixel_bytes = 2 * 32 * 32 * 3
        await camera(h, "red")
        await camera(h, "blue")
        before = tuple(memory.pending)
        count = len(submissions(h, auxiliary=False))
        await command(h, AppendAudio(video_frames=(frame("green"),)))
        events = await command(h, Commit(final=True, create_response=True))
        assert any(event.to_realtime()["type"] == "error" for event in events)
        assert tuple(memory.pending) == before
        assert len(submissions(h, auxiliary=False)) == count
        assert not h.runner.closing
        await command(h, user_item("question", [{"type": "input_text", "text": "Describe the earlier objects."}]))
        await command(h, CreateResponse())
        assert len(submissions(h, auxiliary=False)) == count + 1
        assert "Describe the earlier objects." in submissions(h, auxiliary=False)[-1].prompt["prompt"]
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_context_capacity_closes_short_prefix_without_discarding_tail():
    h = await open_joyai(modalities=("text",))
    try:
        plugin = h.runner.plugin
        state = h.runner.model_state
        memory = state.visual_memory
        for color in ("red", "blue", "green", "yellow"):
            await camera(h, color)
        memory.chunk_frames = 4
        source = memory.frame_source()
        full = plugin._prepare_visual_summary(source, tuple(memory.pending), memory.mid_max_tokens)
        assert full is not None
        text_tokens = len(plugin.processor.tokenizer(text=[full.prompt["prompt"]])["input_ids"][0])
        visual_tokens = sum(entry.token_count for entry in memory.pending) - 4
        plugin.max_model_len = text_tokens + visual_tokens + memory.mid_max_tokens - 1
        request = await plugin.prepare_auxiliary_request(
            history=duplex_text_history(h.session.history), runtime_config=h.session.runtime_config, state=state
        )
        assert request is not None
        covered = len(request.model_source.covered_ids)
        assert 0 < covered < 4
        assert request.model_source.closure_reason == "context"
        assert len(request.prompt["multi_modal_data"]["image"]) == covered
        tail = memory.pending[covered:]
        assert plugin.apply_auxiliary_result(
            request=request, output=summary_output("test", "The early colors."), state=state
        )
        assert memory.pending == tail
        assert memory.mid_term[0].last_id == covered - 1
        assert memory.mid_term[0].closure_reason == "context"
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_image_count_limit_closes_short_prefix_with_image_limit_reason():
    h = await open_joyai(modalities=("text",))
    try:
        plugin = h.runner.plugin
        state = h.runner.model_state
        memory = state.visual_memory
        for color in ("red", "blue", "green", "yellow"):
            await camera(h, color)
        memory.chunk_frames = 4
        plugin.max_images = 2
        request = await plugin.prepare_auxiliary_request(
            history=duplex_text_history(h.session.history), runtime_config=h.session.runtime_config, state=state
        )
        assert request is not None
        source = request.model_source
        assert source.covered_ids == (0, 1)
        assert source.closure_reason == "image_limit"
        images = request.prompt["multi_modal_data"]["image"]
        assert [image.getpixel((0, 0)) for image in images] == [(255, 0, 0), (0, 0, 255)]
        tail = memory.pending[2:]
        assert plugin.apply_auxiliary_result(
            request=request, output=summary_output("test", "Red then blue."), state=state
        )
        assert memory.pending == tail
        assert memory.mid_term[0].closure_reason == "image_limit"
    finally:
        await h.manager.shutdown()
