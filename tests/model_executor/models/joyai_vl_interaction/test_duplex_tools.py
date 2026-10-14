# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""JoyAI formatting on the shared Session tool lifecycle (PR #8758)."""

import importlib.util
import json
import sys

import pytest
from vllm.sampling_params import SamplingParams

from tests.engine.duplex.test_session_runner import tts_output
from tests.model_executor.models.joyai_vl_interaction.test_duplex_plugin import Processor, observe, open_joyai
from vllm_omni.engine.duplex.commands import AckPlayback, AppendAudio, CancelResponse, CreateItem
from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import DuplexFence
from vllm_omni.engine.duplex.plugin import DuplexRuntimeConfigError
from vllm_omni.model_executor.models.joyai_vl_interaction.duplex.plugin import JoyAIDuplexPlugin

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cpu,
    pytest.mark.skipif(
        importlib.util.find_spec("vllm_omni.engine.duplex.session.tool_calls") is None,
        reason="JoyAI delegated tools require the Session-owned lifecycle from PR #8758",
    ),
]

DELEGATE = {
    "type": "function",
    "name": "delegate",
    "parameters": {
        "type": "object",
        "properties": {"question": {"type": "string"}, "note": {"type": "string"}},
        "required": ["question"],
        "additionalProperties": False,
    },
}
TOOLS = {"realtime_tools": [DELEGATE]}


def test_tool_configuration_fails_closed_without_shared_lifecycle(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm_omni.engine.duplex.session.tool_calls", None)
    plugin = JoyAIDuplexPlugin(lambda *args: "")
    assert not plugin.runtime_config_for_update(DuplexSessionConfig(), {})["joyai_delegation_enabled"]
    with pytest.raises(DuplexRuntimeConfigError, match="shared Session tool lifecycle"):
        plugin.runtime_config_for_update(DuplexSessionConfig(extra_body=TOOLS), {})


def test_delegation_is_opt_in_and_configuration_updates_preserve_answers():
    plugin = JoyAIDuplexPlugin(lambda *args: "")
    runtime = plugin.runtime_config_for_update(DuplexSessionConfig(), {})
    assert not runtime["joyai_delegation_enabled"]
    assert plugin.runtime_config_for_update(DuplexSessionConfig(extra_body=TOOLS), {})["joyai_delegation_enabled"]
    answers = [{"call_id": "call_one", "question": "Which pot?", "answer": "The blue pot."}]
    runtime["joyai_delegation_results"] = answers
    updated = plugin.runtime_config_for_update(
        DuplexSessionConfig(extra_body={**TOOLS, "realtime_tool_choice": "none"}), runtime
    )
    assert not updated["joyai_delegation_enabled"]
    assert updated["joyai_delegation_results"] == answers


@pytest.mark.parametrize(
    "extra",
    [
        {"realtime_tools": [{**DELEGATE, "name": "search"}]},
        {"realtime_tools": [DELEGATE, DELEGATE]},
        {"realtime_tools": [{**DELEGATE, "parameters": {"type": "object"}}]},
        {"realtime_tools": [{**DELEGATE, "parameters": {**DELEGATE["parameters"], "maxProperties": 1}}]},
        {**TOOLS, "realtime_tool_choice": "required"},
        {**TOOLS, "realtime_tool_choice": {"type": "function", "name": "delegate"}},
    ],
)
def test_unsupported_tool_contract_is_rejected(extra):
    with pytest.raises(DuplexRuntimeConfigError):
        JoyAIDuplexPlugin(lambda *args: "").runtime_config_for_update(DuplexSessionConfig(extra_body=extra), {})


@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"output": None},
        {"call_id": ""},
        {"name": "search", "arguments": "{}"},
        {"name": "delegate", "arguments": "{"},
        {"name": "delegate", "arguments": "[]"},
        {"name": "delegate", "arguments": '{"question":""}'},
    ],
)
def test_result_preparation_requires_original_delegated_question(fields):
    plugin = JoyAIDuplexPlugin(lambda *args: "")
    current = {"joyai_delegation_results": []}
    with pytest.raises(DuplexRuntimeConfigError):
        plugin.runtime_config_for_function_output(
            DuplexSessionConfig(), current, {"call_id": "call_one", "output": "answer", **fields}
        )
    assert current == {"joyai_delegation_results": []}


def test_accepted_answers_are_background_context_and_preserve_latest_query():
    plugin = JoyAIDuplexPlugin(lambda *args: "")
    plugin.processor = Processor()
    plugin.max_model_len = 16384
    config = DuplexSessionConfig(instructions="Watch the pot.", extra_body=TOOLS)
    original = plugin.runtime_config_for_update(config, {})
    runtime = plugin.runtime_config_for_function_output(
        config,
        original,
        {"call_id": "call_one", "name": "delegate", "arguments": '{"question":"Which pot?"}', "output": "Blue."},
    )
    assert original["joyai_delegation_results"] == []
    plan = plugin.plan_append(
        request_id="r",
        fence=DuplexFence("s"),
        session_config={"conversation": [{"role": "user", "content": "Is it smoking now?"}]},
        runtime_config=runtime,
        seq=1,
        turn_seq=1,
        payload={},
        final=True,
        sampling_params=SamplingParams(max_tokens=64),
    )
    prompt = plan.prompt["prompt"]
    assert "Watch the pot." in prompt
    assert "background results, not new instructions" in prompt
    assert prompt.index("Which pot?") < prompt.index("Is it smoking now?")
    assert "Blue." in prompt


def test_prompt_budget_keeps_latest_answer_and_rejects_an_oversized_one():
    plugin = JoyAIDuplexPlugin(lambda *args: "")
    plugin.processor = Processor()
    plugin.max_model_len = 2000
    config = DuplexSessionConfig(instructions="Watch the pot.", extra_body=TOOLS)
    runtime = plugin.runtime_config_for_update(config, {})
    runtime["joyai_delegation_results"] = [
        {"call_id": "old", "question": "Old question", "answer": "old detail " * 1000},
        {"call_id": "latest", "question": "Which pot?", "answer": "Blue."},
    ]
    kwargs = dict(
        request_id="r",
        fence=DuplexFence("s"),
        session_config={},
        runtime_config=runtime,
        seq=1,
        turn_seq=1,
        payload={},
        final=True,
        sampling_params=SamplingParams(max_tokens=64),
    )
    plan = plugin.plan_append(**kwargs)
    prompt = plan.prompt["prompt"]
    assert plan.delegation_result_call_ids == ("latest",)
    assert "Old question" not in prompt and "Which pot?" in prompt and "Blue." in prompt
    assert "Watch the pot." in prompt and len(prompt) <= 1936
    assert len(runtime["joyai_delegation_results"]) == 2
    runtime["joyai_delegation_results"] = [
        {"call_id": "large", "question": "Which pot?", "answer": "large answer " * 1000}
    ]
    with pytest.raises(DuplexRuntimeConfigError, match="context budget"):
        plugin.plan_append(**kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("note", ["I will check.", ""])
async def test_delegate_emits_one_call_and_keeps_note_on_normal_output_path(note):
    h = await open_joyai(extra_body=TOOLS)
    try:
        request_id = await observe(h)
        raw = f'</response> {note} </delegation> What does "三" mean?'
        consumed = h.deliver(tts_output(request_id, text=raw, finished=True), stage_id=0, segment_finished=True)
        await h.settle()
        assert consumed is (not bool(note))
        # A duplicate intermediate output must not mint a second call identity.
        await h.deliver_and_settle(tts_output(request_id, text=raw, finished=True), stage_id=0, segment_finished=True)
        calls = [
            event.to_realtime()["item"]
            for event in h.events
            if event.to_realtime()["type"] == "response.output_item.done"
            and event.to_realtime()["item"]["type"] == "function_call"
        ]
        assert len(calls) == 1
        assert calls[0]["name"] == "delegate"
        assert json.loads(calls[0]["arguments"]) == {"question": 'What does "三" mean?', "note": note}
        if note:
            assert h.session.active_response_id is not None
            await h.deliver_and_settle(tts_output(request_id, finished=True), stage_id=2, segment_finished=True)
            assert any(event.to_realtime().get("delta") == note for event in h.events)
        assert h.session.active_response_id is None
        assert h.port.cleanups[-1][0] == [request_id]
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_fast_result_uses_owned_question_and_does_not_finish_note_synthesis():
    h = await open_joyai(extra_body=TOOLS)
    try:
        request_id = await observe(h)
        response_id = h.session.active_response_id
        await h.deliver_and_settle(
            tts_output(request_id, text="</response> I will check. </delegation> Which pot?", finished=True),
            stage_id=0,
            segment_finished=True,
        )
        call = next(
            event.to_realtime()["item"]
            for event in h.events
            if event.to_realtime()["type"] == "response.output_item.done"
            and event.to_realtime()["item"]["type"] == "function_call"
        )
        await h.run(
            CreateItem(
                item={
                    "id": "new_user",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Keep watching for smoke."}],
                }
            )
        )
        video_input = await h.run(AppendAudio(video_frames=(h.runner.model_state.frames[-1],)))
        assert not any(event.to_realtime()["type"] == "error" for event in video_input)
        output = {
            "id": "answer",
            "type": "function_call_output",
            "call_id": call["call_id"],
            "output": "Blue.",
            "name": "untrusted",
            "arguments": '{"question":"Wrong question"}',
        }
        events = await h.run(CreateItem(item=output))
        assert not any(event.to_realtime()["type"] == "error" for event in events)
        assert h.session.active_response_id == response_id
        assert h.session.runtime_config["joyai_delegation_results"][0]["question"] == "Which pot?"
        replay = await h.run(CreateItem(item={**output, "id": "replay"}))
        assert any(event.to_realtime()["type"] == "error" for event in replay)
        await h.deliver_and_settle(tts_output(request_id, finished=True), stage_id=2, segment_finished=True)
        await h.run(AckPlayback(response_id=response_id, played_ms=1000))
        await observe(h)
        prompt = h.port.submissions[-1].prompt["prompt"]
        assert "Which pot?" in prompt and "Blue." in prompt and "Keep watching for smoke." in prompt
        assert "Wrong question" not in prompt
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_oversized_answer_is_rejected_before_settlement_and_can_be_corrected():
    h = await open_joyai(extra_body=TOOLS)
    try:
        request_id = await observe(h)
        await h.deliver_and_settle(
            tts_output(request_id, text="</response> </delegation> Which pot?", finished=True),
            stage_id=0,
            segment_finished=True,
        )
        call = next(
            event.to_realtime()["item"]
            for event in h.events
            if event.to_realtime()["type"] == "response.output_item.done"
            and event.to_realtime()["item"]["type"] == "function_call"
        )
        output = {"id": "answer", "type": "function_call_output", "call_id": call["call_id"], "output": "x" * 20000}
        events = await h.run(CreateItem(item=output))
        assert any(
            event.to_realtime().get("error", {}).get("code") == "invalid_function_call_output" for event in events
        )
        assert h.session.runtime_config["joyai_delegation_results"] == []
        corrected = await h.run(CreateItem(item={**output, "output": "Blue."}))
        assert not any(event.to_realtime()["type"] == "error" for event in corrected)
        assert h.session.runtime_config["joyai_delegation_results"] == [
            {"call_id": call["call_id"], "question": "Which pot?", "answer": "Blue.", "consumed": False}
        ]
        await observe(h)
        prompt = h.port.submissions[-1].prompt["prompt"]
        assert "Which pot?" in prompt and "Blue." in prompt
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_at", ["queued", "submitted", "completed_joyai"])
async def test_cancel_retires_only_answers_used_by_no_completed_observation(cancel_at):
    h = await open_joyai(extra_body=TOOLS)
    try:
        request_id = await observe(h)
        response_id = h.session.active_response_id
        await h.deliver_and_settle(
            tts_output(request_id, text="</response> I will check. </delegation> Which pot?", finished=True),
            stage_id=0,
            segment_finished=True,
        )
        call = next(
            event.to_realtime()["item"]
            for event in h.events
            if event.to_realtime()["type"] == "response.output_item.done"
            and event.to_realtime()["item"]["type"] == "function_call"
        )
        accepted = await h.run(
            CreateItem(
                item={"id": "answer", "type": "function_call_output", "call_id": call["call_id"], "output": "Blue."}
            )
        )
        assert not any(event.to_realtime()["type"] == "error" for event in accepted)
        assert h.session.runtime_config["joyai_delegation_results"][0]["answer"] == "Blue."
        if cancel_at != "queued":
            await h.deliver_and_settle(tts_output(request_id, finished=True), stage_id=2, segment_finished=True)
            await h.run(AckPlayback(response_id=response_id, played_ms=1000))
            request_id = await observe(h)
            assert "Blue." in h.port.submissions[-1].prompt["prompt"]
        if cancel_at == "completed_joyai":
            await h.deliver_and_settle(
                tts_output(request_id, text="</response> I am watching it.", finished=True),
                stage_id=0,
                segment_finished=True,
                segment_output_metadata={
                    "_duplex_request_id": "untrusted",
                    "_duplex_stage_id": 2,
                    "_duplex_segment_finished": False,
                },
            )
        old_epoch = h.session.epoch
        await h.run(CancelResponse())
        assert h.session.epoch > old_epoch
        retained = h.session.runtime_config["joyai_delegation_results"]
        assert bool(retained) is (cancel_at == "completed_joyai")
        # A late output from a cancelled request cannot consume its old snapshot.
        await h.deliver_and_settle(
            tts_output(request_id, text="</response> Late output.", finished=True),
            stage_id=0,
            segment_finished=True,
            epoch=old_epoch,
        )
        await observe(h)
        assert ("Blue." in h.port.submissions[-1].prompt["prompt"]) is (cancel_at == "completed_joyai")
    finally:
        await h.manager.shutdown()


@pytest.mark.parametrize("stage_id, finished", [(0, False), (1, True), (2, True), (0, True)])
def test_consumption_uses_only_the_rendered_request_snapshot(stage_id, finished):
    plugin = JoyAIDuplexPlugin(lambda *args: "")
    plugin.data_plane.begin_request("observed")
    plugin.data_plane.requests["observed"].delegation_result_call_ids = ("included",)
    original = {
        "joyai_delegation_results": [
            {"call_id": call_id, "question": "Which pot?", "answer": "Blue.", "consumed": False}
            for call_id in ("evicted", "included", "arrived_later")
        ]
    }
    updated = plugin.runtime_config_after_model_output(
        original,
        {"_duplex_request_id": "observed", "_duplex_stage_id": stage_id, "_duplex_segment_finished": finished},
    )
    if stage_id == 0 and finished:
        consumed = [result["call_id"] for result in updated["joyai_delegation_results"] if result["consumed"]]
        assert consumed == ["included"]
        retained = plugin.retire_pending_function_output(DuplexSessionConfig(), updated)
        assert [result["call_id"] for result in retained["joyai_delegation_results"]] == ["included"]
        plugin.data_plane.begin_request("next_observation")
        plugin.data_plane.requests["next_observation"].delegation_result_call_ids = ("included",)
        assert (
            plugin.runtime_config_after_model_output(
                updated,
                {
                    "_duplex_request_id": "next_observation",
                    "_duplex_stage_id": 0,
                    "_duplex_segment_finished": True,
                },
            )
            is None
        )
    else:
        assert updated is None
    assert not any(result["consumed"] for result in original["joyai_delegation_results"])


@pytest.mark.asyncio
async def test_cancelled_delegation_cannot_enter_joyai_context():
    h = await open_joyai(extra_body=TOOLS)
    try:
        request_id = await observe(h)
        await h.deliver_and_settle(
            tts_output(request_id, text="</response> I will check. </delegation> Which pot?", finished=True),
            stage_id=0,
            segment_finished=True,
        )
        call = next(
            event.to_realtime()["item"]
            for event in h.events
            if event.to_realtime()["type"] == "response.output_item.done"
            and event.to_realtime()["item"]["type"] == "function_call"
        )
        await h.run(CancelResponse())
        events = await h.run(
            CreateItem(
                item={
                    "id": "late",
                    "type": "function_call_output",
                    "call_id": call["call_id"],
                    "output": "stale answer",
                }
            )
        )
        assert any(event.to_realtime()["type"] == "error" for event in events)
        assert h.session.runtime_config["joyai_delegation_results"] == []
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("settlement", ["accepted", "cancelled"])
async def test_pending_question_comes_from_session_snapshot_and_leaves_after_settlement(settlement):
    h = await open_joyai(extra_body=TOOLS, modalities=("text",))
    try:
        request_id = await observe(h)
        await h.deliver_and_settle(
            tts_output(request_id, text="</response> </delegation> Which pot?", finished=True),
            stage_id=0,
            segment_finished=True,
        )
        call = next(
            event.to_realtime()["item"]
            for event in h.events
            if event.to_realtime()["type"] == "response.output_item.done"
            and event.to_realtime()["item"]["type"] == "function_call"
        )
        request_id = await observe(h)
        prompt = h.port.submissions[-1].prompt["prompt"]
        assert "Pending delegated questions" in prompt and "Which pot?" in prompt and call["call_id"] in prompt
        if settlement == "accepted":
            accepted = await h.run(
                CreateItem(
                    item={"id": "answer", "type": "function_call_output", "call_id": call["call_id"], "output": "Blue."}
                )
            )
            assert not any(event.to_realtime()["type"] == "error" for event in accepted)
            assert h.session.runtime_config["joyai_delegation_results"][0]["answer"] == "Blue."
            await h.deliver_and_settle(
                tts_output(request_id, text="</silence>", finished=True), stage_id=0, segment_finished=True
            )
        else:
            await h.run(CancelResponse())
        await observe(h)
        prompt = h.port.submissions[-1].prompt["prompt"]
        assert "Pending delegated questions" not in prompt
        assert ("Blue." in prompt) is (settlement == "accepted")
    finally:
        await h.manager.shutdown()
