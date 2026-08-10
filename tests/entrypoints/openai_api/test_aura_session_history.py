# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for AURA SessionHistory."""

from __future__ import annotations

import numpy as np
import pytest

from vllm_omni.model_executor.stage_input_processors.aura_session_history import (
    AURA_IM_END_TOKEN_ID,
    AURA_SILENT_TOKEN_ID,
    SessionHistory,
    aura_silent_stop_token_ids,
    is_effectively_silent,
    normalize_assistant_text,
    should_stop_aura_silent_generation,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _video_tuple(num_frames: int = 2) -> tuple[np.ndarray, dict]:
    frames = np.zeros((num_frames, 8, 8, 3), dtype=np.uint8)
    metadata = {
        "fps": 2.0,
        "duration": num_frames / 2.0,
        "total_num_frames": num_frames,
        "frames_indices": list(range(num_frames)),
        "video_backend": "opencv",
        "do_sample_frames": False,
    }
    return frames, metadata


def test_get_vllm_inputs_includes_video_and_text():
    history = SessionHistory()
    history.add_user_message("What is happening?", video_tuple=_video_tuple())

    vllm_inputs = history.get_vllm_inputs()

    assert "<|video_pad|>" in vllm_inputs["prompt"]
    assert "What is happening?" in vllm_inputs["prompt"]
    assert vllm_inputs["prompt"].endswith("<|im_start|>assistant")
    assert "video" in vllm_inputs["multi_modal_data"]
    assert len(vllm_inputs["multi_modal_data"]["video"]) == 1


def test_preview_vllm_inputs_matches_committed_turn():
    history = SessionHistory()
    history.add_user_message("first", video_tuple=_video_tuple())
    history.add_assistant_message("reply")

    preview = history.preview_vllm_inputs("second", video_tuple=_video_tuple(3))
    history.add_user_message("second", video_tuple=_video_tuple(3), mm_uuid=history._pending_mm_uuid)
    committed = history.get_vllm_inputs()

    assert preview["prompt"] == committed["prompt"]
    assert len(preview["multi_modal_data"]["video"]) == len(committed["multi_modal_data"]["video"])


def test_preview_attaches_stable_uuids_and_marks_warm_history():
    history = SessionHistory()
    uid0 = history.new_mm_uuid()
    history.add_user_message("first", video_tuple=_video_tuple(), mm_uuid=uid0)
    history.add_assistant_message("reply")
    history.mark_mm_uuids_warm([uid0])

    preview = history.preview_vllm_inputs("second", video_tuple=_video_tuple(3))
    videos = preview["multi_modal_data"]["video"]
    uuids = preview["multi_modal_uuids"]["video"]

    # Pixels always present (HF path cannot take None); warm count is diagnostic.
    assert videos[0] is not None
    assert videos[1] is not None
    assert uuids[0] == uid0
    assert uuids[1] == history._pending_mm_uuid
    assert preview["mm_uuid_only_videos"] == 1
    assert preview["mm_pixel_videos"] == 1


def test_to_dict_roundtrip_preserves_history():
    history = SessionHistory(max_rounds=4, num_rounds_keep=2, pruning_enabled=False)
    history.add_user_message("", video_tuple=_video_tuple())
    history.add_assistant_message("I see movement.")
    history.add_user_message("Tell me more.", video_tuple=_video_tuple(3))
    history.add_assistant_message("<|silent|>")

    restored = SessionHistory.from_dict(history.to_dict())

    assert restored.system_prompt == history.system_prompt
    assert restored.current_rounds == history.current_rounds
    assert len(restored.history) == len(history.history)
    restored_inputs = restored.get_vllm_inputs()
    original_inputs = history.get_vllm_inputs()
    assert restored_inputs["prompt"] == original_inputs["prompt"]
    assert len(restored_inputs["multi_modal_data"].get("video", [])) == len(
        original_inputs["multi_modal_data"].get("video", [])
    )


def test_is_effectively_silent_treats_punctuation_filler_as_silent():
    assert is_effectively_silent("")
    assert is_effectively_silent("  ")
    assert is_effectively_silent("<|silent|>")
    assert is_effectively_silent(" ﹑")
    assert is_effectively_silent("，。")
    assert not is_effectively_silent("好的")
    assert not is_effectively_silent(" 好的，")


def test_add_assistant_message_normalizes_punctuation_filler_to_silent():
    history = SessionHistory()
    history.add_assistant_message(" ﹑")
    assert history.history[-1]["content"] == "<|silent|>"
    assert normalize_assistant_text(" ﹑") == "<|silent|>"


def test_pruning_moves_old_rounds_to_context_history():
    history = SessionHistory(
        max_rounds=2,
        num_rounds_keep=1,
        pruning_enabled=True,
        max_context_qas=5,
        max_1qna_rounds=4,
    )

    for round_idx in range(3):
        history.add_user_message(f"question {round_idx}", video_tuple=_video_tuple())
        history.add_assistant_message(f"answer {round_idx}")

    assert history._sw_round_count() <= history.max_rounds
    assert len(history._context_history) >= 1
    assert any("question 0" in msg.get("content", "") for qa in history._context_history for msg in qa)
    for qa in history._context_history:
        for msg in qa:
            content = msg.get("content", "")
            assert isinstance(content, str), "context_history stores text-only user/assistant content"
            assert "<|video_pad|>" not in content
            assert content != "<|silent|>"


def test_aura_session_state_commit_turn_clears_frames_without_api_history():
    from vllm_omni.model_executor.stage_input_processors.aura_session_history import (
        clear_all_sessions,
        create_streaming_session,
        get_or_create_session_history,
        get_session_history,
        record_pending_turn,
    )

    clear_all_sessions()
    state = create_streaming_session(pruning_enabled=False)
    state.turn_frame_arrays = [np.zeros((4, 4, 3), dtype=np.uint8), np.ones((4, 4, 3), dtype=np.uint8)]
    state.freeze_turn_video(
        {
            "video": [
                (
                    np.zeros((2, 4, 4, 3), dtype=np.uint8),
                    {"fps": 2.0, "total_num_frames": 2},
                )
            ]
        }
    )
    get_or_create_session_history(state.session_id, system_prompt="sys")
    record_pending_turn(
        state.session_id,
        request_id="req-1",
        transcript="hello",
        video_tuple=(np.zeros((2, 4, 4, 3), dtype=np.uint8), {"fps": 2.0, "total_num_frames": 2}),
    )

    state.commit_turn(response_text="reply", request_id="req-1")

    assert state.turn_frame_arrays == []
    assert state.pending_turn_video is None
    assert state.history.current_rounds == 0
    stage_hist = get_session_history(state.session_id)
    assert stage_hist is not None
    prompt = stage_hist.get_vllm_inputs()["prompt"]
    assert "hello" in prompt
    assert "reply" in prompt
    clear_all_sessions()


def test_should_stop_aura_silent_generation_on_first_token():
    assert aura_silent_stop_token_ids() == (AURA_SILENT_TOKEN_ID, AURA_IM_END_TOKEN_ID)
    assert should_stop_aura_silent_generation(token_ids=[AURA_SILENT_TOKEN_ID])
    assert should_stop_aura_silent_generation(token_ids=[AURA_IM_END_TOKEN_ID])
    assert not should_stop_aura_silent_generation(token_ids=[42])
    assert should_stop_aura_silent_generation(text="<|silent|>")
    assert should_stop_aura_silent_generation(text=" ﹑")
    assert not should_stop_aura_silent_generation(text="好的")
