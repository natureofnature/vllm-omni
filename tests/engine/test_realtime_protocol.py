# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Wire-contract coverage for the shared engine Realtime protocol layer."""

import base64
import struct

import pytest

from vllm_omni.engine.duplex.commands import (
    AppendAudio,
    CancelResponse,
    ClearInput,
    CloseSession,
    Commit,
    CreateItem,
    CreateResponse,
    DuplexCommand,
    DuplexCommandError,
    command_from_realtime,
)
from vllm_omni.engine.realtime.commands import (
    RealtimeInputDefaults,
    translate_realtime_command,
)
from vllm_omni.engine.realtime.projection import (
    RealtimeProjectionState,
    clear_input_buffer,
    note_input_append,
    project_internal_event,
    resolve_cancel_response,
    resolve_commit,
    resolve_create_item,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"type": "input_audio_buffer.commit"}, Commit()),
        ({"type": "input_audio_buffer.commit", "create_response": True}, Commit(create_response=True)),
        ({"type": "input.commit"}, Commit(create_response=True)),
        ({"type": "input_audio_buffer.clear"}, ClearInput()),
        (
            {"type": "response.create", "response": {"modalities": ["text"]}},
            CreateResponse(options={"modalities": ["text"]}),
        ),
        ({"type": "response.cancel", "response_id": "response-a"}, CancelResponse(response_id="response-a")),
        ({"type": "session.close"}, CloseSession()),
    ],
)
def test_framework_command_entrypoint_uses_shared_protocol(payload: dict[str, object], expected: DuplexCommand) -> None:
    assert translate_realtime_command(payload) == expected
    assert command_from_realtime(payload) == expected


def test_audio_append_preserves_pcm_and_native_duplex_hints() -> None:
    pcm = struct.pack("<hh", 0, 16384)
    command = translate_realtime_command(
        {
            "type": "input_audio_buffer.append",
            "event_id": "client-a",
            "audio": base64.b64encode(pcm).decode(),
            "is_speech": True,
            "force_listen": True,
            "audio_end_ms": 200,
        }
    )
    assert isinstance(command, AppendAudio)
    assert command.audio == struct.pack("<ff", 0.0, 0.5)
    assert command.format == "pcm_f32le"
    assert command.sample_rate_hz == 16000
    assert command.event_id == "client-a"
    assert command.payload()["force_listen"] is True
    assert command.payload()["audio_end_ms"] == 200


def test_wire_defaults_do_not_leak_between_sessions() -> None:
    original = RealtimeInputDefaults()
    updated = original.with_session_payload(
        {"audio": {"input": {"format": {"type": "audio/pcm_f32le", "rate": 24000}}}}
    )
    payload = {"type": "input_audio_buffer.append", "audio": "AAAAAA==", "is_speech": False}
    command = translate_realtime_command(payload, defaults=updated)
    assert isinstance(command, AppendAudio)
    assert (command.format, command.sample_rate_hz) == ("pcm_f32le", 24000)
    assert (original.input_audio_format, original.input_sample_rate_hz) == ("pcm16", 16000)


def test_session_created_preserves_initial_update_and_resume_fields() -> None:
    state = RealtimeProjectionState(session_id="session-a", model="model-a", initial_session_update=True)
    wire = [
        event.to_realtime()
        for event in project_internal_event(
            state,
            {"type": "session.created", "attachment_generation": 2, "resume_token": "test-token"},
        )
    ]
    assert [event["type"] for event in wire] == ["session.created", "session.updated"]
    assert wire[0]["session"] == wire[1]["session"]
    assert wire[0]["session"]["id"] == "session-a"
    assert wire[0]["session"]["model"] == "model-a"
    assert wire[0]["attachment_generation"] == 2
    assert wire[0]["resume_token"] == "test-token"
    assert not state.initial_session_update


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        ({"type": "unknown.event"}, "unknown_event"),
        ({"type": "conversation.item.delete"}, "missing_item_id"),
        ({"type": "input_audio_buffer.append", "format": "mp3", "audio": "AA=="}, "unsupported_audio_format"),
        ({"type": "input_audio_buffer.append", "format": "pcm_f32le", "audio": "%%%"}, "bad_audio"),
    ],
)
def test_protocol_errors_keep_client_event_identity(payload: dict[str, object], code: str) -> None:
    with pytest.raises(DuplexCommandError) as exc:
        translate_realtime_command({**payload, "event_id": "bad-client-event"})
    assert exc.value.code == code
    assert exc.value.event_id == "bad-client-event"


def test_minicpm_video_extension_keeps_existing_validation() -> None:
    # The wire validator checks an image header; decoding belongs to model input.
    frame = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()
    payload = {
        "type": "input_audio_buffer.append",
        "audio": "AAAAAA==",
        "video_frames": [frame, None, frame],
        "max_slice_nums": 1,
        "force_listen": True,
    }
    command = translate_realtime_command(payload)
    assert isinstance(command, AppendAudio)
    assert command.video_frames == (frame, frame)
    assert command.payload()["force_listen"] is True
    with pytest.raises(DuplexCommandError) as exc:
        translate_realtime_command({**payload, "video_frames": [frame] * 3})
    assert exc.value.code == "invalid_video_frames"


def test_empty_commit_is_resolved_by_engine_state_not_parser() -> None:
    command = command_from_realtime({"type": "input_audio_buffer.commit", "event_id": "commit-a"})
    assert isinstance(command, Commit)
    result = resolve_commit(RealtimeProjectionState(session_id="session-a"), command)
    assert result.payload is None
    assert result.events[0].to_realtime()["error"]["code"] == "input_audio_buffer_empty"
    assert result.events[0].to_realtime()["error"]["event_id"] == "commit-a"


def test_append_commit_and_clear_keep_session_state_isolated() -> None:
    state = RealtimeProjectionState(session_id="session-a")
    other = RealtimeProjectionState(session_id="session-b")
    append = command_from_realtime({"type": "input_audio_buffer.append", "audio": "AAAAAA==", "is_speech": True})
    assert isinstance(append, AppendAudio)
    note_input_append(state, append.payload())
    item_id = state.active_input_item_id
    result = resolve_commit(state, Commit(event_id="commit-a"))
    assert result.payload is not None
    assert result.payload["realtime_item_id"] == item_id
    assert result.payload["response_create"] is False
    assert result.payload["realtime_event_id"] == "commit-a"
    assert not state.input_audio_buffer_has_audio
    assert state.pending_commit_item_ids == [item_id]
    assert not other.input_audio_buffer_has_audio
    assert other.pending_commit_item_ids == []
    note_input_append(state, append.payload())
    clear_input_buffer(state)
    assert state.active_input_item_id is None
    assert not state.input_audio_buffer_has_audio
    # Clearing the next buffer must not erase already submitted commit identities.
    assert state.pending_commit_item_ids == [item_id]


def test_text_item_is_resolved_without_native_listen_speak() -> None:
    state = RealtimeProjectionState(session_id="session-a")
    command = command_from_realtime(
        {
            "type": "conversation.item.create",
            "item": {"id": "user-a", "role": "user", "content": [{"type": "input_text", "text": "Hello"}]},
        }
    )
    assert isinstance(command, CreateItem)
    resolved = resolve_create_item(state, command)
    assert resolved.payloads
    assert command.item["id"] == "user-a"
    assert command.item["type"] == "message"
    assert all(event.type not in {"response.listen", "response.speak"} for event in resolved.events)


def test_text_response_completes_once_without_native_decision_events() -> None:
    state = RealtimeProjectionState(session_id="session-a", model="text-model")
    wire = [
        projected.to_realtime()
        for event in (
            {"type": "response.created", "response_id": "response-a", "modalities": ["text"]},
            {"type": "response.text.delta", "response_id": "response-a", "delta": "Hello"},
            {"type": "response.done", "response_id": "response-a"},
        )
        for projected in project_internal_event(state, event)
    ]
    types = [event["type"] for event in wire]
    assert types.count("response.created") == 1
    assert types.count("response.output_text.done") == 1
    assert types.count("response.done") == 1
    assert not {"response.listen", "response.speak", "response.audio.done"}.intersection(types)
    done = next(event for event in wire if event["type"] == "response.done")
    assert done["response"]["id"] == "response-a"
    assert done["response"]["status"] == "completed"
    assert done["response"]["output"][0]["content"] == [{"type": "text", "text": "Hello"}]
    assert state.active_response_id is None
    assert project_internal_event(state, {"type": "response.done", "response_id": "response-a"}) == []
    cancelled = resolve_cancel_response(state, CancelResponse(event_id="cancel-a", response_id="response-a"))
    assert cancelled.payloads == []
    assert cancelled.events[0].to_realtime()["error"]["code"] == "response_not_active"


def test_minicpm_terminal_audio_keeps_wire_sequence_and_identity() -> None:
    state = RealtimeProjectionState(session_id="session-a")
    project_internal_event(state, {"type": "response.created", "response_id": "response-a"})
    audio = base64.b64encode(struct.pack("<hh", 0, 16384)).decode()
    wire = [
        projected.to_realtime()
        for projected in project_internal_event(
            state,
            {
                "type": "response.output_audio.delta",
                "response_id": "response-a",
                "audio": audio,
                "format": "pcm",
                "sample_rate_hz": 24000,
                "text": "Hello",
                "model_speak": True,
                "end_of_turn": True,
            },
        )
    ]
    types = [event["type"] for event in wire]
    assert types[:5] == [
        "response.speak",
        "response.audio.delta",
        "response.audio_transcript.delta",
        "response.audio.done",
        "response.audio_transcript.done",
    ]
    assert types.count("response.done") == 1
    delta = wire[1]
    assert delta["response_id"] == "response-a"
    assert delta["item_id"] == "item_response-a"
    assert delta["delta"] == audio
    assert delta["sample_rate_hz"] == 24000
    assert wire[4]["transcript"] == "Hello"
    assert project_internal_event(state, {"type": "response.done", "response_id": "response-a"}) == []


def test_listen_does_not_synthesize_response_completion() -> None:
    state = RealtimeProjectionState(session_id="session-a")
    events = project_internal_event(state, {"type": "response.listen", "input_seq": 7})
    assert [event.type for event in events] == ["response.listen"]
    assert events[0].to_realtime()["response"]["metadata"]["input_seq"] == 7
    assert state.response_states == {}
