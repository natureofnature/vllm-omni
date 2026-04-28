from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import (
    Qwen3OmniMoeForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_resolve_voice_type_uses_request_scoped_speaker_over_instance_state():
    model = object.__new__(Qwen3OmniMoeForConditionalGeneration)
    model.default_tts_text_spk_type = "ethan"
    model.voice_type = "stale-instance-voice"

    assert model._resolve_voice_type({"speaker": ["Vivian"]}) == "vivian"
    assert model._resolve_voice_type({"speaker": "Ryan"}) == "ryan"
    assert model._resolve_voice_type({}) == "ethan"


def test_talker_preprocess_prefill_reads_speaker_from_request_payload():
    model = object.__new__(Qwen3OmniMoeForConditionalGeneration)
    model.default_tts_text_spk_type = "ethan"
    model.voice_type = "stale-instance-voice"
    model.talker = SimpleNamespace(text_projection=lambda x: x)
    model._module_device = lambda module: torch.device("cpu")
    model._talker_cache_thinker_decode_embeds = lambda info_dict, update_dict: None

    captured = {}

    def _capture_voice_type(voice_type: str) -> int:
        captured["voice_type"] = voice_type
        return 7

    model._get_text_spk_token_id = _capture_voice_type
    model._thinker_to_talker_prefill = lambda **kwargs: (
        torch.tensor([1]),
        torch.zeros((1, 4), dtype=torch.bfloat16),
        torch.zeros((0, 4), dtype=torch.bfloat16),
    )

    info_dict = {
        "speaker": ["Vivian"],
        "thinker_prefill_embeddings": torch.zeros((1, 4), dtype=torch.float32),
        "thinker_hidden_states": torch.zeros((1, 4), dtype=torch.float32),
        "thinker_sequences": [11],
        "thinker_input_ids": [11],
        "tts_bos_embed": torch.zeros((1, 1, 4), dtype=torch.float32),
        "tts_eos_embed": torch.zeros((1, 1, 4), dtype=torch.float32),
        "tts_pad_embed": torch.zeros((1, 1, 4), dtype=torch.float32),
        "num_processed_tokens": 0,
    }

    req_ids, req_embeds, update_dict = model.talker_preprocess_prefill(
        input_ids=torch.tensor([1]),
        input_embeds=torch.zeros((1, 4), dtype=torch.bfloat16),
        **info_dict,
    )

    assert captured["voice_type"] == "vivian"
    assert req_ids.shape == (1,)
    assert req_embeds.shape == (1, 4)
    assert isinstance(update_dict, dict)


def test_thinker_to_talker_prefill_short_tail_keeps_tts_eos_embed():
    model = object.__new__(Qwen3OmniMoeForConditionalGeneration)
    model.config = SimpleNamespace(
        talker_config=SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=4),
            codec_nothink_id=1,
            codec_think_bos_id=2,
            codec_think_eos_id=3,
            codec_pad_id=4,
            codec_bos_id=5,
        ),
        tts_pad_token_id=123,
    )
    model.talker = SimpleNamespace(
        text_projection=lambda x: x,
        embed_input_ids=lambda ids: torch.zeros((ids.shape[0], 4), dtype=torch.bfloat16, device=ids.device),
    )

    thinker_embed = torch.zeros((4, 4), dtype=torch.bfloat16)
    tts_pad_embed = torch.zeros((1, 4), dtype=torch.bfloat16)
    tts_bos_embed = torch.zeros((1, 4), dtype=torch.bfloat16)
    tts_eos_embed = torch.full((1, 4), 7, dtype=torch.bfloat16)

    _, _, trailing_text_hidden = model._get_talker_assistant_parts(
        im_start_index=0,
        segment_end_index=4,
        speaker_id=6,
        thinker_embed=thinker_embed,
        tts_pad_embed=tts_pad_embed,
        tts_bos_embed=tts_bos_embed,
        tts_eos_embed=tts_eos_embed,
    )

    assert torch.equal(trailing_text_hidden, tts_eos_embed)


def test_resolve_voice_type_interleaved_requests_leave_no_instance_state():
    model = object.__new__(Qwen3OmniMoeForConditionalGeneration)
    model.default_tts_text_spk_type = "ethan"

    first = model._resolve_voice_type({"speaker": ["Vivian"]})
    second = model._resolve_voice_type({"speaker": ["Ryan"]})
    fallback = model._resolve_voice_type({})

    assert first == "vivian"
    assert second == "ryan"
    assert fallback == "ethan"
    assert not hasattr(model, "voice_type")


def test_thinker2talker_full_payload_preserves_speaker_and_language():
    request = SimpleNamespace(
        prompt_token_ids=[11],
        output_token_ids=[22],
        additional_information=SimpleNamespace(
            entries={
                "speaker": SimpleNamespace(list_data=["Vivian"]),
                "language": SimpleNamespace(list_data=["en"]),
            }
        ),
    )
    pooling_output = {
        "0": torch.zeros((1, 4), dtype=torch.float32),
        "24": torch.zeros((1, 4), dtype=torch.float32),
        "tts_bos_embed": torch.zeros((1, 1, 4), dtype=torch.float32),
        "tts_eos_embed": torch.zeros((1, 1, 4), dtype=torch.float32),
        "tts_pad_embed": torch.zeros((1, 1, 4), dtype=torch.float32),
    }

    from vllm_omni.model_executor.stage_input_processors.qwen3_omni import thinker2talker_full_payload

    payload = thinker2talker_full_payload(None, pooling_output, request)

    assert payload is not None
    assert payload["speaker"] == "vivian"
    assert payload["language"] == ["en"]
    assert bool(payload["finished"].item()) is True
