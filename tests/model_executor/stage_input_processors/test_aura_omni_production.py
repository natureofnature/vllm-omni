# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import (
    PRECOMPUTED_TEXT_IDS_KEY,
)
from vllm_omni.model_executor.stage_input_processors.aura_omni import (
    SILENT_TEXT,
    _estimate_assistant_prompt_tokens,
    _estimate_instruct_prompt_tokens,
    _estimate_tts_max_new_tokens,
    _estimate_tts_prompt_len_from_token_ids,
    _pop_emit_ready_tts_text,
    _pop_native_tts_sentence,
    asr2aura,
    asr2aura_async_chunk,
    aura2tts,
    aura2tts_async_chunk,
    build_aura_input,
    build_aura_streaming_turn_additional_information,
    pack_aura_video_ndarray,
    resolve_aura_async_chunk_stage_payload,
    unpack_aura_video_ndarray,
    video_tuple_from_deferred_multi_modal,
)
from vllm_omni.model_executor.stage_input_processors.aura_session_history import (
    SessionHistory,
    clear_all_sessions,
    get_session_history,
    register_session,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _source_output(text: str, request_id: str = "req-1", token_ids: list[int] | None = None):
    output = SimpleNamespace(text=text, cumulative_token_ids=token_ids or [1, 2, 3], multimodal_output={})
    return SimpleNamespace(request_id=request_id, outputs=[output])


def _partial_source_output(text: str, request_id: str = "req-1", token_ids: list[int] | None = None):
    output = SimpleNamespace(text=text, cumulative_token_ids=token_ids or [1, 2, 3], multimodal_output={})
    return SimpleNamespace(request_id=request_id, outputs=[output], finished=False)


def _source_delta_final_output(cumulative_text: str, request_id: str = "req-1", token_ids: list[int] | None = None):
    output = SimpleNamespace(
        text="partial",
        cumulative_text=cumulative_text,
        cumulative_token_ids=token_ids or [1, 2, 3],
        multimodal_output={},
    )
    return SimpleNamespace(request_id=request_id, outputs=[output], finished=True)


def _transfer_manager():
    return SimpleNamespace(config=SimpleNamespace(), request_payload={})


def test_asr2aura_carries_video_payload_and_transcript():
    prompt = {
        "multi_modal_data": {"video": ["frame-0", "frame-1"]},
        "additional_information": {"aura_system_prompt": ["system"]},
    }

    [next_input] = asr2aura([_source_output("What is happening now?")], prompt=[prompt])

    assert next_input["multi_modal_data"] == {"video": ["frame-0", "frame-1"]}
    assert "<|video_pad|>" in next_input["prompt"]
    assert "What is happening now?" in next_input["prompt"]
    assert next_input["prompt"].startswith("<|im_start|>system\nsystem")


def test_asr2aura_forwards_tts_options_to_aura_worker():
    prompt = {
        "multi_modal_data": {"video": ["frame-0"]},
        "additional_information": {
            "tts_task_type": ["Base"],
            "tts_ref_audio": ["voice.wav"],
            "tts_ref_text": ["hello"],
            "aura_system_prompt": ["system"],
        },
    }

    [next_input] = asr2aura([_source_output("看看视频")], prompt=[prompt])

    assert next_input["additional_information"] == {
        "tts_task_type": ["Base"],
        "tts_ref_audio": ["voice.wav"],
        "tts_ref_text": ["hello"],
        "aura_system_prompt": ["system"],
    }


def test_asr2aura_drops_audio_before_qwen3_vl_stage():
    prompt = {
        "multi_modal_data": {
            "audio": ("wave", 16000),
            "video": ["frame-0", "frame-1"],
        },
    }

    [next_input] = asr2aura([_source_output("Check the video")], prompt=[prompt])

    assert next_input["multi_modal_data"] == {"video": ["frame-0", "frame-1"]}
    assert "<|video_pad|>" in next_input["prompt"]


def test_asr2aura_reads_video_stashed_for_downstream_stage():
    prompt = {
        "multi_modal_data": {"audio": ("wave", 16000)},
        "additional_information": {
            "deferred_multi_modal_data": {"video": ["frame-0", "frame-1"]},
        },
    }

    [next_input] = asr2aura([_source_output("Check the video")], prompt=[prompt])

    assert next_input["multi_modal_data"] == {"video": ["frame-0", "frame-1"]}
    assert "<|video_pad|>" in next_input["prompt"]


def test_asr2aura_supports_video_only_observation():
    prompt = {"multi_modal_data": {"video": ["frame-0", "frame-1"]}}

    [next_input] = asr2aura([_source_output("")], prompt=[prompt])

    assert "<|video_pad|>" in next_input["prompt"]
    assert "<|im_start|>assistant" in next_input["prompt"]


def test_asr2aura_async_chunk_waits_until_asr_finished(monkeypatch):
    class FakeTokenizer:
        def encode(self, text):
            return [ord(ch) for ch in text]

    monkeypatch.setattr(
        "vllm_omni.model_executor.stage_input_processors.aura_omni.cached_tokenizer_from_config",
        lambda _config: FakeTokenizer(),
    )
    transfer_manager = _transfer_manager()
    request = SimpleNamespace(
        request_id="req-1",
        external_req_id="req-1",
        output_text="看看视频",
        additional_information={
            "aura_system_prompt": ["system"],
            "deferred_multi_modal_data": {"video": ["frame-0"]},
            "tts_ref_audio": ["voice.wav"],
        },
        is_finished=lambda: False,
    )

    assert asr2aura_async_chunk(transfer_manager, None, request, is_finished=False) is None
    request.output_text = "看看视频里有什么"

    payload = asr2aura_async_chunk(transfer_manager, None, request, is_finished=True)

    assert payload["aura_asr_transcript"] == "看看视频里有什么"
    assert payload["additional_information"]["tts_ref_audio"] == ["voice.wav"]
    assert "aura_turn_video" not in payload


def test_aura2tts_builds_qwen3_tts_prompt_information():
    prompt = {
        "additional_information": {
            "tts_language": ["Chinese"],
            "tts_instruct": ["Calm voice."],
            "tts_ref_audio": ["ref.wav"],
            "tts_ref_text": ["Reference transcript sample."],
        }
    }

    [tts_input] = aura2tts([_source_output("Hello.")], prompt=[prompt])

    assert tts_input["prompt_token_ids"]
    assert tts_input["additional_information"]["text"] == ["Hello."]
    assert PRECOMPUTED_TEXT_IDS_KEY not in tts_input["additional_information"]
    assert tts_input["additional_information"]["task_type"] == ["Base"]
    assert tts_input["additional_information"]["language"] == ["Chinese"]
    assert tts_input["additional_information"]["ref_audio"] == ["ref.wav"]
    assert tts_input["additional_information"]["ref_text"] == ["Reference transcript sample."]
    assert tts_input["additional_information"]["x_vector_only_mode"] == [False]
    assert tts_input["additional_information"]["instruct"] == ["Calm voice."]
    assert tts_input["additional_information"]["max_new_tokens"][0] < 2048


def test_aura2tts_preserves_explicit_tts_max_new_tokens():
    prompt = {
        "additional_information": {
            "tts_task_type": ["CustomVoice"],
            "tts_speaker": ["Vivian"],
            "tts_max_new_tokens": [96],
        }
    }

    [tts_input] = aura2tts([_source_output("Hello.")], prompt=[prompt])

    assert tts_input["additional_information"]["max_new_tokens"] == [96]


def test_aura2tts_prefers_streaming_cumulative_text():
    prompt = {
        "additional_information": {
            "tts_ref_audio": ["ref.wav"],
            "tts_ref_text": ["Reference transcript sample."],
        }
    }

    [tts_input] = aura2tts(
        [_source_delta_final_output("The complete AURA reply.")],
        prompt=[prompt],
    )

    assert tts_input["additional_information"]["text"] == ["The complete AURA reply."]


def test_aura2tts_supports_x_vector_only_mode_for_base():
    prompt = {
        "additional_information": {
            "tts_task_type": ["Base"],
            "tts_x_vector_only_mode": [True],
            "tts_ref_audio": ["ref.wav"],
            "tts_ref_text": ["Reference transcript sample."],
        }
    }

    [tts_input] = aura2tts([_source_output("Hello.")], prompt=[prompt])

    assert tts_input["additional_information"]["x_vector_only_mode"] == [True]


def test_aura2tts_supports_custom_voice_mode():
    prompt = {
        "additional_information": {
            "tts_task_type": ["CustomVoice"],
            "tts_speaker": ["vivian"],
        }
    }

    [tts_input] = aura2tts([_source_output("Hello.")], prompt=[prompt])

    assert tts_input["additional_information"]["task_type"] == ["CustomVoice"]
    assert tts_input["additional_information"]["speaker"] == ["Vivian"]
    assert "ref_audio" not in tts_input["additional_information"]
    # Placeholder length ≈ real CustomVoice prefill for short English (was hardcoded
    # 14 under AURA content_ids; text-mode estimate is intentionally nearby).
    assert 10 <= len(tts_input["prompt_token_ids"]) <= 24


def test_aura2tts_passes_token_ids_to_qwen3_tts_when_enabled():
    prompt = {
        "additional_information": {
            "tts_ref_audio": ["ref.wav"],
            "tts_ref_text": ["Reference transcript sample."],
            "tts_pass_token_ids": [True],
        }
    }

    [tts_input] = aura2tts(
        [
            _source_output(
                "Hello.",
                token_ids=[151644, 77091, 198, 108386, 1773, 151645, 198],
            )
        ],
        prompt=[prompt],
    )

    text_ids = tts_input["additional_information"][PRECOMPUTED_TEXT_IDS_KEY][0]
    assert 108386 in text_ids
    assert 1773 in text_ids
    assert "text" not in tts_input["additional_information"]


def test_aura2tts_async_chunk_reads_nested_request_additional_information():
    transfer_manager = _transfer_manager()
    request = SimpleNamespace(
        request_id="req-1",
        external_req_id="req-1",
        output_token_ids=[108386, 1773],
        output_text="你好",
        additional_information={
            "additional_information": {
                "tts_task_type": ["CustomVoice"],
                "tts_speaker": ["vivian"],
                "tts_language": ["Chinese"],
            }
        },
        is_finished=lambda: False,
    )

    assert aura2tts_async_chunk(transfer_manager, None, request) is None
    payload = aura2tts_async_chunk(transfer_manager, None, request, is_finished=True)

    assert payload["task_type"] == ["CustomVoice"]
    assert payload["speaker"] == ["Vivian"]
    assert payload["language"] == ["Chinese"]
    assert payload["text"] == ["你好"]
    assert payload["prompt_token_ids"]
    assert payload["meta"]["next_stage_prompt_len"] == len(payload["prompt_token_ids"])
    assert payload["meta"]["finished"].item() is True
    assert "ref_audio" not in payload
    assert "ref_text" not in payload


def test_aura2tts_async_chunk_keeps_tts_metadata_when_request_info_is_cleared():
    transfer_manager = _transfer_manager()
    request = SimpleNamespace(
        request_id="req-1",
        external_req_id="req-1",
        output_token_ids=[108386, 1773],
        output_text="你好",
        additional_information={
            "additional_information": {
                "tts_task_type": ["CustomVoice"],
                "tts_speaker": ["vivian"],
                "tts_language": ["Chinese"],
            }
        },
        is_finished=lambda: False,
    )

    assert aura2tts_async_chunk(transfer_manager, None, request, is_finished=False) is None
    request.additional_information = {}
    payload = aura2tts_async_chunk(transfer_manager, None, request, is_finished=True)

    assert payload["task_type"] == ["CustomVoice"]
    assert payload["speaker"] == ["Vivian"]
    assert payload["language"] == ["Chinese"]
    assert payload["text"] == ["你好"]
    assert "ref_audio" not in payload


def test_aura2tts_async_chunk_reads_tts_metadata_from_stage_payload_when_request_info_is_cleared():
    transfer_manager = _transfer_manager()
    request = SimpleNamespace(
        request_id="req-1",
        external_req_id="req-1",
        output_token_ids=[108386, 1773],
        output_text="你好",
        additional_information={},
        omni_stage_payload={
            "prompt": "aura prompt",
            "additional_information": {
                "tts_task_type": ["CustomVoice"],
                "tts_speaker": ["vivian"],
                "tts_language": ["Chinese"],
            },
        },
        is_finished=lambda: False,
    )

    payload = aura2tts_async_chunk(transfer_manager, None, request, is_finished=True)

    assert payload["task_type"] == ["CustomVoice"]
    assert payload["speaker"] == ["Vivian"]
    assert payload["language"] == ["Chinese"]
    assert payload["text"] == ["你好"]
    assert "ref_audio" not in payload


def test_aura2tts_async_chunk_accumulates_and_sends_full_text_once_finished():
    transfer_manager = _transfer_manager()
    request = SimpleNamespace(
        request_id="req-1",
        external_req_id="req-1",
        output_token_ids=[108386, 1773],
        output_text="你好",
        additional_information={
            "tts_ref_audio": ["custom.wav"],
            "tts_ref_text": ["custom transcript"],
        },
        is_finished=lambda: False,
    )

    assert aura2tts_async_chunk(transfer_manager, None, request) is None
    request.output_token_ids = [108386, 1773, 104139]
    request.output_text = "你好，世界"
    payload = aura2tts_async_chunk(transfer_manager, None, request, is_finished=True)

    assert payload["text"] == ["你好，世界"]
    assert payload["task_type"] == ["Base"]
    assert payload["ref_audio"] == ["custom.wav"]
    assert payload["prompt_token_ids"]


def test_aura2tts_async_chunk_passes_token_ids_only_when_enabled():
    transfer_manager = _transfer_manager()
    request = SimpleNamespace(
        request_id="req-1",
        external_req_id="req-1",
        output_token_ids=[108386, 1773],
        output_text="你好",
        additional_information={
            "tts_ref_audio": ["custom.wav"],
            "tts_ref_text": ["custom transcript"],
            "tts_pass_token_ids": [True],
        },
        is_finished=lambda: False,
    )

    assert aura2tts_async_chunk(transfer_manager, None, request) is None
    request.output_token_ids = [108386, 1773, 104139]
    payload = aura2tts_async_chunk(transfer_manager, None, request, is_finished=True)

    assert PRECOMPUTED_TEXT_IDS_KEY in payload
    assert "text" not in payload
    assert payload["task_type"] == ["Base"]
    assert 104139 in payload[PRECOMPUTED_TEXT_IDS_KEY][0]


def test_aura2tts_async_chunk_decodes_text_instead_of_passing_source_token_ids(monkeypatch):
    class FakeTokenizer:
        def decode(self, token_ids):
            assert token_ids == [101, 102, 103, 104]
            return "第一句\n\n第二句"

    monkeypatch.setattr(
        "vllm_omni.model_executor.stage_input_processors.aura_omni.cached_tokenizer_from_config",
        lambda config: FakeTokenizer(),
    )
    transfer_manager = _transfer_manager()
    transfer_manager.config = SimpleNamespace()
    request = SimpleNamespace(
        request_id="req-1",
        external_req_id="req-1",
        output_token_ids=[101, 102, 103, 104],
        additional_information={
            "tts_task_type": ["CustomVoice"],
            "tts_speaker": ["Vivian"],
            "tts_language": ["Chinese"],
        },
        is_finished=lambda: False,
    )

    payload = aura2tts_async_chunk(transfer_manager, None, request, is_finished=True)

    assert payload["text"] == ["第一句 第二句"]
    assert payload["task_type"] == ["CustomVoice"]
    assert PRECOMPUTED_TEXT_IDS_KEY not in payload
    assert payload["prompt_token_ids"]


def test_aura2tts_async_chunk_holds_silent_token_prefix_until_finished():
    request = SimpleNamespace(
        request_id="req-1",
        output_token_ids=[151669],
        additional_information={},
        is_finished=lambda: False,
    )

    assert aura2tts_async_chunk(None, None, request) is None


def test_aura2tts_async_chunk_emits_finish_payload_on_silent_turn():
    request = SimpleNamespace(
        request_id="req-1",
        output_token_ids=[151669],
        additional_information={},
        is_finished=lambda: True,
    )

    payload = aura2tts_async_chunk(None, None, request, is_finished=True)

    assert payload is not None
    assert payload["prompt_token_ids"] == []
    assert payload["meta"]["finished"].item() is True


def test_aura2tts_async_chunk_emits_finish_payload_when_tts_input_is_silent():
    transfer_manager = SimpleNamespace(request_payload={})
    request = SimpleNamespace(
        request_id="req-2",
        external_req_id="req-2",
        output_token_ids=[100, 101, 102],
        output_text=SILENT_TEXT,
        additional_information={},
        is_finished=lambda: True,
    )

    payload = aura2tts_async_chunk(transfer_manager, None, request, is_finished=True)

    assert payload is not None
    assert payload["prompt_token_ids"] == []
    assert payload["meta"]["finished"].item() is True


def test_pop_native_tts_sentence_matches_native_boundaries():
    s, rest = _pop_native_tts_sentence("你好世界。下一句")
    assert s == "你好世界。"
    assert rest == "下一句"
    s, rest = _pop_native_tts_sentence("短，")
    assert s is None and rest == "短，"


def test_pop_emit_ready_tts_text_holds_short_sentence_until_merge():
    held, rest = _pop_emit_ready_tts_text("有。")
    assert held is None
    assert rest == "有。"

    merged, rest = _pop_emit_ready_tts_text("有。画面正中央坐着一位戴着眼镜、黑色耳机的年轻男子，")
    assert merged == "有。画面正中央坐着一位戴着眼镜、黑色耳机的年轻男子，"
    assert rest == ""


def test_estimate_tts_max_new_tokens_scales_without_flat_48_floor():
    # Solo "有。" must not keep a multi-second filler budget.
    assert _estimate_tts_max_new_tokens("有。", []) < 48
    assert _estimate_tts_max_new_tokens("有。", []) >= 16
    longer = _estimate_tts_max_new_tokens("画面正中央坐着一位戴着眼镜、黑色耳机的年轻男子，", [])
    assert longer > _estimate_tts_max_new_tokens("有。", [])
    assert _estimate_tts_max_new_tokens("x", [], explicit=96) == 96


def test_estimate_tts_max_new_tokens_tight_for_short_chinese():
    """Failed-EOS babble must not get a ~5s budget for an 8-char reply."""
    text = "好，我在这儿呢。"
    cap = _estimate_tts_max_new_tokens(text, [])
    assert cap <= 40  # ~3.3s @ 12Hz; old chars*6+8 was ~56
    # Text length wins over longer AURA content_ids in text-mode.
    assert _estimate_tts_max_new_tokens(text, [0] * 20) == cap


def test_estimate_tts_prompt_len_does_not_use_instruct_char_count():
    """Long English instruct must not inflate prompt_len by ~100 zero pads."""
    long_instruct = (
        "Speak clearly and briskly. Start with the first word immediately. "
        "Do not cough, clear your throat, sigh, moan, hum, laugh, or add filler sounds."
    )
    assert len(long_instruct) > 100
    # Token estimate should be far below naive char length (~144) and near real ~40.
    assert 30 <= _estimate_instruct_prompt_tokens(long_instruct) <= 50

    text = "当然可以，我正看着你呢。"
    length_ids = [0] * _estimate_assistant_prompt_tokens(text)
    with_instruct = _estimate_tts_prompt_len_from_token_ids(
        length_ids,
        task_type="CustomVoice",
        language="Chinese",
        instruct=long_instruct,
    )
    without = _estimate_tts_prompt_len_from_token_ids(
        length_ids,
        task_type="CustomVoice",
        language="Chinese",
        instruct="",
    )
    # Old bug: instruct added +144; real TTS tokenizer adds ~40.
    assert with_instruct - without < 55
    # Absolute size near real tokenizer (~60 with instruct, ~20 without).
    assert 45 <= with_instruct <= 80
    assert 12 <= without <= 35


def test_aura2tts_customvoice_prompt_len_stays_near_text_size():
    long_instruct = (
        "Speak clearly and briskly. Start with the first word immediately. "
        "Do not cough, clear your throat, sigh, moan, hum, laugh, or add filler sounds."
    )
    prompt = {
        "additional_information": {
            "tts_task_type": ["CustomVoice"],
            "tts_speaker": ["Serena"],
            "tts_language": ["Chinese"],
            "tts_instruct": [long_instruct],
        }
    }
    [tts_input] = aura2tts([_source_output("当然可以，我正看着你呢。")], prompt=[prompt])
    # Pre-fix this was ~164 (char-length instruct). Real tokenizer ≈ 60.
    assert len(tts_input["prompt_token_ids"]) < 100
    assert len(tts_input["prompt_token_ids"]) > 30


def test_aura2tts_async_chunk_emits_mid_generation_sentence(monkeypatch):
    monkeypatch.setenv("VLLM_AURA_SENTENCE_TTS", "1")
    transfer_manager = _transfer_manager()
    request = SimpleNamespace(
        request_id="req-sent",
        external_req_id="req-sent",
        output_token_ids=[1, 2, 3],
        output_text="今天天气很好。后面还有",
        additional_information={
            "tts_task_type": ["CustomVoice"],
            "tts_speaker": ["Vivian"],
            "tts_language": ["Chinese"],
        },
        is_finished=lambda: False,
    )
    mid = aura2tts_async_chunk(transfer_manager, None, request, is_finished=False)
    assert mid is not None
    assert mid["text"] == ["今天天气很好。"]
    # Finish with more text: remnant after prior sentence emit.
    request.output_text = "今天天气很好。后面还有内容"
    request.is_finished = lambda: True
    fin = aura2tts_async_chunk(transfer_manager, None, request, is_finished=True)
    assert fin is not None
    assert "后面还有内容" in (fin.get("text") or [""])[0] or fin.get("prompt_token_ids") == []


def test_aura2tts_async_chunk_merges_short_leading_sentence(monkeypatch):
    monkeypatch.setenv("VLLM_AURA_SENTENCE_TTS", "1")
    transfer_manager = _transfer_manager()
    request = SimpleNamespace(
        request_id="req-short",
        external_req_id="req-short",
        output_token_ids=[1, 2, 3],
        output_text="有。",
        additional_information={
            "tts_task_type": ["CustomVoice"],
            "tts_speaker": ["Serena"],
            "tts_language": ["Chinese"],
        },
        is_finished=lambda: False,
    )
    assert aura2tts_async_chunk(transfer_manager, None, request, is_finished=False) is None

    request.output_text = "有。画面正中央坐着一位戴着眼镜、黑色耳机的年轻男子，他正面对着"
    mid = aura2tts_async_chunk(transfer_manager, None, request, is_finished=False)
    assert mid is not None
    assert mid["text"] == ["有。画面正中央坐着一位戴着眼镜、黑色耳机的年轻男子，"]
    # Merged emit uses proportional budget; solo "有。" would have been < 48.
    assert mid["max_new_tokens"][0] == _estimate_tts_max_new_tokens(mid["text"][0], [])
    assert _estimate_tts_max_new_tokens("有。", []) < 48


def test_aura2tts_drops_silent_response():
    assert aura2tts([_source_output(SILENT_TEXT)]) == []


def test_aura2tts_streaming_partial_content_enters_tts():
    prompt = {
        "additional_information": {
            "tts_ref_audio": ["ref.wav"],
            "tts_ref_text": ["Reference transcript sample."],
        }
    }

    [tts_input] = aura2tts(
        [_partial_source_output("你好", token_ids=[151644, 77091, 198, 108386])],
        prompt=[prompt],
    )

    assert tts_input["additional_information"]["text"] == ["你好"]
    assert PRECOMPUTED_TEXT_IDS_KEY not in tts_input["additional_information"]


def test_build_aura_input_uses_stage_session_history_when_server_registry_misses():
    clear_all_sessions()
    history = SessionHistory(system_prompt="system")
    history.add_user_message("第一輪問題")
    history.add_assistant_message("第一輪回答")

    from vllm_omni.model_executor.stage_input_processors import aura_session_history as ash

    ash.get_or_create_session_history("aura-stage-test", system_prompt="system")
    ash._STAGE_WORKER_SESSIONS["aura-stage-test"] = history

    video = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    metadata = {
        "fps": 2.0,
        "duration": 1.0,
        "total_num_frames": 2,
        "frames_indices": [0, 1],
        "video_backend": "opencv",
        "do_sample_frames": False,
    }
    additional_info = {
        "aura_session_id": "aura-stage-test",
        "deferred_multi_modal_data": {"video": [(video, metadata)]},
        "aura_system_prompt": ["system"],
    }

    next_input = build_aura_input(
        transcript="第二輪問題",
        additional_info=additional_info,
        multi_modal_data={},
        request_id="req-2",
    )

    assert "第一輪問題" in next_input["prompt"]
    assert "第一輪回答" in next_input["prompt"]
    assert "第二輪問題" in next_input["prompt"]
    assert len(next_input["multi_modal_data"]["video"]) == 1
    assert next_input["additional_information"]["aura_session_id"] == "aura-stage-test"


def test_build_aura_input_ignores_api_session_registry():
    """API ``AuraSessionState.history`` / ``register_session`` must not drive prompts."""
    clear_all_sessions()

    api_history = SessionHistory(system_prompt="system")
    api_history.add_user_message("API-only turn")
    api_history.add_assistant_message("should-not-appear")
    register_session("aura-ignore-api", api_history)

    video = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    metadata = {"fps": 2.0, "duration": 1.0, "total_num_frames": 2}
    next_input = build_aura_input(
        transcript="stage-first",
        additional_info={
            "aura_session_id": "aura-ignore-api",
            "deferred_multi_modal_data": {"video": [(video, metadata)]},
            "aura_system_prompt": ["system"],
        },
        multi_modal_data={},
        request_id="req-1",
    )
    assert "API-only turn" not in next_input["prompt"]
    assert "should-not-appear" not in next_input["prompt"]
    assert "stage-first" in next_input["prompt"]
    assert get_session_history("aura-ignore-api") is not None
    clear_all_sessions()
    clear_all_sessions()


def test_aura2tts_commits_stage_session_turn():
    clear_all_sessions()
    video = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    metadata = {"fps": 2.0, "duration": 1.0, "total_num_frames": 2}
    additional_info = {
        "aura_session_id": "aura-sync-commit",
        "deferred_multi_modal_data": {"video": [(video, metadata)]},
        "aura_system_prompt": ["system"],
    }
    # Production sync flow records and commits under the same stage request id
    # (both derive from ``source_output.request_id``).
    build_aura_input(
        transcript="你好",
        additional_info=additional_info,
        multi_modal_data={},
        request_id="req-1",
    )
    assert get_session_history("aura-sync-commit").current_rounds == 0

    aura2tts(
        [_partial_source_output("好的。", token_ids=[151644, 77091, 198, 108386])],
        prompt=[
            {
                "request_id": "0",
                "additional_information": additional_info,
            }
        ],
    )
    history = get_session_history("aura-sync-commit")
    assert history.current_rounds == 1
    assert "你好" in history.get_vllm_inputs()["prompt"]
    assert "好的。" in history.get_vllm_inputs()["prompt"]
    # Idempotent: second commit without pending is a no-op.
    from vllm_omni.model_executor.stage_input_processors.aura_session_history import (
        commit_session_turn,
    )

    commit_session_turn("aura-sync-commit", "extra")
    assert history.current_rounds == 1
    clear_all_sessions()


def test_build_aura_input_commit_rounds_increment_via_aura2tts_async_chunk():
    clear_all_sessions()
    video = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    metadata = {"fps": 2.0, "duration": 1.0, "total_num_frames": 2}
    additional_info = {
        "aura_session_id": "aura-commit-test",
        "deferred_multi_modal_data": {"video": [(video, metadata)]},
        "aura_system_prompt": ["system"],
    }
    next_input = build_aura_input(
        transcript="",
        additional_info=additional_info,
        multi_modal_data={},
        request_id="req-1",
    )
    assert next_input["additional_information"]["aura_session_id"] == "aura-commit-test"

    history = get_session_history("aura-commit-test")
    assert history is not None
    assert history.current_rounds == 0

    transfer_manager = SimpleNamespace(request_payload={})
    request = SimpleNamespace(
        request_id="req-1",
        external_req_id="req-1",
        additional_information=next_input["additional_information"],
        output_text=SILENT_TEXT,
        output_token_ids=[151669],
        is_finished=lambda: True,
    )
    aura2tts_async_chunk(transfer_manager, None, request, is_finished=True)

    history = get_session_history("aura-commit-test")
    assert history.current_rounds == 1
    assert SILENT_TEXT in history.get_vllm_inputs()["prompt"]


def test_resolve_aura_async_chunk_stage_payload_builds_prompt_from_passthrough():
    clear_all_sessions()
    video = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    metadata = {"fps": 2.0, "duration": 1.0, "total_num_frames": 2}
    payload = {
        "aura_asr_transcript": "你好",
        "additional_information": {
            "aura_session_id": "aura-resolve-test",
            "aura_system_prompt": ["system"],
            "deferred_multi_modal_data": {"video": [(video, metadata)]},
        },
    }
    request = SimpleNamespace(
        request_id="req-1",
        external_req_id="req-1",
        additional_information=None,
        omni_stage_payload=None,
    )

    class _FakeTokenizer:
        def encode(self, prompt: str) -> list[int]:
            return [1, 2, 3]

    class _FakeConfig:
        pass

    import vllm_omni.model_executor.stage_input_processors.aura_omni as aura_omni_mod

    original = aura_omni_mod.cached_tokenizer_from_config
    aura_omni_mod.cached_tokenizer_from_config = lambda _cfg: _FakeTokenizer()
    try:
        resolve_aura_async_chunk_stage_payload(payload, request, _FakeConfig())
    finally:
        aura_omni_mod.cached_tokenizer_from_config = original

    # Raw prompt text is consumed during resolve; pixels are stripped so the
    # scheduler never IPCs video via additional_information.
    assert "prompt" not in payload
    assert "multi_modal_data" not in payload
    assert payload["prompt_token_ids"] == [1, 2, 3]
    assert get_session_history("aura-resolve-test") is not None


def test_coerce_video_frames_array_handles_inhomogeneous_streaming_shapes():
    from vllm_omni.model_executor.stage_input_processors.aura_omni import (
        _coerce_video_frames_array,
        _normalize_video_tuple,
    )

    frames = [
        np.zeros((8, 8, 3), dtype=np.uint8),
        np.zeros((10, 12, 3), dtype=np.uint8),
        np.zeros((8, 8, 3), dtype=np.uint8),
    ]
    with pytest.raises(ValueError):
        np.asarray(frames, dtype=np.uint8)

    video_array = _coerce_video_frames_array(frames)
    assert video_array is not None
    assert video_array.shape == (2, 8, 8, 3)

    normalized = _normalize_video_tuple(frames, {"fps": 2.0})
    assert normalized is not None
    array, metadata = normalized
    assert array.shape[0] >= 2
    assert metadata["fps"] == 2.0


def test_pack_aura_video_ndarray_roundtrips_pixels():
    video = np.arange(2 * 4 * 4 * 3, dtype=np.uint8).reshape(2, 4, 4, 3)
    packed = pack_aura_video_ndarray(video)
    assert packed["__aura_video_ndarray__"] is True
    assert packed["shape"] == [2, 4, 4, 3]
    assert isinstance(packed["data"], (bytes, bytearray))
    restored = unpack_aura_video_ndarray(packed)
    assert restored is not None
    np.testing.assert_array_equal(restored, video)


def test_deferred_video_wire_format_survives_additional_information_msgpack():
    """Regression: nested ndarray under scalar_data must not become ['|u1', shape, buf]."""
    from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

    from vllm_omni.engine import AdditionalInformationPayload
    from vllm_omni.engine.serialization import (
        deserialize_additional_information,
        serialize_additional_information,
    )

    video = np.arange(2 * 4 * 4 * 3, dtype=np.uint8).reshape(2, 4, 4, 3)
    metadata = {"fps": 2.0, "duration": 1.0, "total_num_frames": 2}
    additional = build_aura_streaming_turn_additional_information(
        session_id="aura-wire",
        video_array=video,
        video_metadata=metadata,
        system_prompt="system",
        skip_asr=True,
        include_tts=False,
    )
    packed = additional["deferred_multi_modal_data"]["video"][0][0]
    assert isinstance(packed, dict) and packed.get("__aura_video_ndarray__")

    payload = serialize_additional_information(additional)
    encoded = MsgpackEncoder().encode(payload)
    decoded = MsgpackDecoder(AdditionalInformationPayload).decode(encoded)
    restored_info = deserialize_additional_information(decoded)

    video_tuple = video_tuple_from_deferred_multi_modal(restored_info.get("deferred_multi_modal_data"))
    assert video_tuple is not None
    arr, meta = video_tuple
    assert arr.shape == (2, 4, 4, 3)
    np.testing.assert_array_equal(arr, video)
    assert meta["fps"] == 2.0
    assert meta["total_num_frames"] == 2


def test_build_aura_input_resolves_packed_deferred_video():
    clear_all_sessions()
    video = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    video[0, 0, 0] = [1, 2, 3]
    metadata = {"fps": 2.0, "duration": 1.0, "total_num_frames": 2}
    additional_info = build_aura_streaming_turn_additional_information(
        session_id="aura-packed-resolve",
        video_array=video,
        video_metadata=metadata,
        system_prompt="system",
        skip_asr=True,
        include_tts=False,
    )
    next_input = build_aura_input(
        transcript="",
        additional_info=additional_info,
        multi_modal_data={},
        request_id="req-packed",
    )
    videos = next_input["multi_modal_data"]["video"]
    assert len(videos) == 1
    arr = videos[0][0] if isinstance(videos[0], (tuple, list)) else videos[0]
    assert hasattr(arr, "shape") and arr.shape == (2, 4, 4, 3)
    clear_all_sessions()


def _video_multimodal_data() -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    video = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    metadata = {"fps": 2.0, "duration": 1.0, "total_num_frames": 2}
    return {"video": [(video, metadata)]}, video, metadata


def _finish_async_chunk_turn(
    *,
    session_id: str,
    transcript: str,
    multi_modal_data: dict[str, Any],
    request_id: str,
    silent: bool = True,
    response_text: str = "",
) -> dict[str, Any]:
    additional_info = {
        "aura_session_id": session_id,
        "aura_system_prompt": ["system"],
    }
    next_input = build_aura_input(
        transcript=transcript,
        additional_info=additional_info,
        multi_modal_data=multi_modal_data,
        request_id=request_id,
    )
    transfer_manager = SimpleNamespace(request_payload={})
    if silent:
        content_ids = [151669]
        output_text = SILENT_TEXT
    else:
        content_ids = [108386, 77091]
        output_text = response_text or "好的，我會留意。"
    request = SimpleNamespace(
        request_id=request_id,
        external_req_id=request_id,
        additional_information=next_input["additional_information"],
        output_text=output_text,
        output_token_ids=content_ids,
        is_finished=lambda: True,
    )
    aura2tts_async_chunk(transfer_manager, None, request, is_finished=True)
    return next_input


def test_multi_turn_session_history_smoke_like_0002():
    clear_all_sessions()
    session_id = "aura-multi-turn"
    multi_modal_data, _, _ = _video_multimodal_data()

    for idx in range(3):
        _finish_async_chunk_turn(
            session_id=session_id,
            transcript="",
            multi_modal_data=multi_modal_data,
            request_id=f"req-silent-{idx}",
        )
    history = get_session_history(session_id)
    assert history is not None
    assert history.current_rounds == 3

    spoken_input = _finish_async_chunk_turn(
        session_id=session_id,
        transcript="出现《古韵》这本书的时候，提醒我。",
        multi_modal_data=multi_modal_data,
        request_id="req-spoken",
        silent=False,
        response_text="好的，我會留意，等《古韵》這本書出現時馬上提醒你。",
    )
    assert "出现《古韵》这本书的时候，提醒我。" in spoken_input["prompt"]
    history = get_session_history(session_id)
    assert history.current_rounds == 4

    for idx in range(2):
        _finish_async_chunk_turn(
            session_id=session_id,
            transcript="",
            multi_modal_data=multi_modal_data,
            request_id=f"req-silent-post-{idx}",
        )
    history = get_session_history(session_id)
    assert history.current_rounds == 6
    final_prompt = history.get_vllm_inputs()["prompt"]
    assert "出现《古韵》这本书的时候，提醒我。" in final_prompt
    assert final_prompt.count("<|silent|>") >= 4


def test_silent_placeholder_video_commits_vision_pad_text_to_history():
    clear_all_sessions()
    multi_modal_data = {"video": ["frame-0", "frame-1", "frame-2"]}
    _finish_async_chunk_turn(
        session_id="aura-placeholder-video",
        transcript="",
        multi_modal_data=multi_modal_data,
        request_id="req-placeholder",
    )
    history = get_session_history("aura-placeholder-video")
    assert history is not None
    assert history.current_rounds == 1
    prompt = history.get_vllm_inputs()["prompt"]
    assert "<|vision_start|><|video_pad|><|vision_end|>" in prompt
    assert SILENT_TEXT in prompt


def test_empty_asr_placeholder_video_prompt_includes_current_user_vision_pad():
    clear_all_sessions()
    session_id = "aura-prompt-vision-pad"
    placeholder_mm = {"video": ["frame-0", "frame-1", "frame-2"]}
    _finish_async_chunk_turn(
        session_id=session_id,
        transcript="请简单介绍下这本书讲的是什么。",
        multi_modal_data=placeholder_mm,
        request_id="req-q2",
        silent=False,
        response_text="好的，我这就简单介绍一下这本书的内容。",
    )
    next_input = build_aura_input(
        transcript="",
        additional_info={
            "aura_session_id": session_id,
            "aura_system_prompt": ["system"],
        },
        multi_modal_data=placeholder_mm,
        request_id="req-empty-video",
    )
    prompt = next_input["prompt"]
    assert prompt.endswith("<|im_start|>assistant")
    last_user_idx = prompt.rfind("<|im_start|>user")
    last_assistant_idx = prompt.rfind("<|im_start|>assistant")
    assert last_user_idx != -1 and last_user_idx < last_assistant_idx
    current_user_block = prompt[last_user_idx:last_assistant_idx]
    assert "<|vision_start|><|video_pad|><|vision_end|>" in current_user_block
    assert "请简单介绍下这本书讲的是什么。" not in current_user_block
