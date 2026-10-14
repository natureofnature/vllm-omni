# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Received audio must win over silence before PCM buffering, too."""

import asyncio
from types import SimpleNamespace

import pytest

from tests.engine.duplex.test_session_runner import append_audio, types
from tests.engine.duplex.test_session_runner_nemotron_voicechat import open_nemotron_harness
from vllm_omni.engine.duplex import commands
from vllm_omni.engine.duplex.session import model_channel as model_channel_module
from vllm_omni.engine.duplex.session import runner as runner_module

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def activate_response(h, monkeypatch):
    # Only replace these modules' clock reference; the event loop keeps its
    # real clock. The silence deadline is already due, so no timed sleep runs.
    clock = SimpleNamespace(monotonic=lambda: 100.081)
    monkeypatch.setattr(runner_module, "time", clock)
    monkeypatch.setattr(model_channel_module, "time", clock)
    h.runner.model_state.last_native_submit_monotonic = 100.0
    h.session.begin_response(turn_id=h.session.turn_id)


async def schedule_silence(h):
    response_id = h.session.active_response_id
    assert response_id is not None
    return await h.runner._schedule_silence_continuation(
        h.runner.model.silence_unit_payload(),
        request_id=h.session.active_request_id,
        owner_id=f"response:{response_id}",
        response_id=response_id,
        response_owned=True,
        expected_epoch=h.session.epoch,
        expected_model_turn_id=None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("waiting_at", ["mailbox", "decode"])
async def test_received_audio_prevents_silence_before_pcm_buffering(monkeypatch, waiting_at):
    async with open_nemotron_harness(monkeypatch) as h:
        await h.run(append_audio(samples=1280))
        activate_response(h, monkeypatch)
        entered, release = asyncio.Event(), asyncio.Event()
        if waiting_at == "decode":
            original = h.runner.offload

            async def gated_decode(fn, *args, **kwargs):
                if fn is runner_module.convert_input_audio_with_rate:
                    entered.set()
                    await release.wait()
                return await original(fn, *args, **kwargs)

            monkeypatch.setattr(h.runner, "offload", gated_decode)
            h.submit(append_audio(samples=1280))
        else:
            original = h.runner._handle_item
            blocker = commands.UpdateSession(patch={})

            async def gated_mailbox(item):
                if item is blocker:
                    entered.set()
                    await release.wait()
                return await original(item)

            monkeypatch.setattr(h.runner, "_handle_item", gated_mailbox)
            h.submit(blocker)
            h.submit(append_audio(samples=1280))
        await asyncio.wait_for(entered.wait(), timeout=2)
        try:
            # The old predicate sees both as empty and accepts a full synthetic
            # acoustic frame ahead of the client's already-received frame.
            assert not h.runner.model_state.audio_buffer.has_pending()
            assert not h.runner.model_state.audio_buffer.has_reserved()
            assert await schedule_silence(h) is False
            assert len(h.port.submissions) == 1
        finally:
            release.set()
            await h.settle()
        assert len(h.port.submissions) == 2
        assert [s.prompt["model_intermediate_buffer"]["duplex"]["source_input_seq"] for s in h.port.submissions] == [
            1,
            2,
        ]
        # Successful dispatch releases input priority; reply continuation can
        # resume after the real frame has been submitted.
        h.runner.model_state.last_native_submit_monotonic = 100.0
        assert await schedule_silence(h) is True
        assert await h.runner.tasks.append_tail
        assert len(h.port.submissions) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [ValueError("invalid audio"), RuntimeError("decoder failed")])
async def test_rejected_audio_releases_silence_priority(monkeypatch, failure):
    async with open_nemotron_harness(monkeypatch) as h:
        await h.run(append_audio(samples=1280))
        activate_response(h, monkeypatch)
        original = h.runner.offload

        async def reject_decode(fn, *args, **kwargs):
            if fn is runner_module.convert_input_audio_with_rate:
                raise failure
            return await original(fn, *args, **kwargs)

        monkeypatch.setattr(h.runner, "offload", reject_decode)
        events = await h.run(append_audio(samples=1280))
        assert "error" in types(events)
        assert len(h.port.submissions) == 1
        # Neither the validation-error return nor the worker's exception path
        # may leave a permanent real-input flag that suppresses continuation.
        assert await schedule_silence(h) is True
        assert await h.runner.tasks.append_tail
        assert len(h.port.submissions) == 2
