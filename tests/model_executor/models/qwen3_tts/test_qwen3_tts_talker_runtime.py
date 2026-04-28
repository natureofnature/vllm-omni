# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.tts_utils import (
    QWEN3_TTS_TALKER_GPU_RESIDENT_BUFFER_KEYS,
    build_qwen3_tts_talker_multimodal_outputs,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_build_qwen3_tts_talker_multimodal_outputs_aggregates_request_payloads():
    ref_code = torch.tensor([[9, 10]], dtype=torch.long)
    span_len, mm = build_qwen3_tts_talker_multimodal_outputs(
        [
            {
                "audio_codes": torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
                "ref_code_len": 5,
                "codec_streaming": True,
            },
            {
                "audio_codes": torch.tensor([[5, 6]], dtype=torch.long),
                "ref_code_len": torch.tensor([7], dtype=torch.int32),
                "ref_code": ref_code,
                "codec_streaming": False,
            },
        ]
    )

    assert span_len == 3
    torch.testing.assert_close(mm["audio_codes"], torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.long))
    torch.testing.assert_close(mm["ref_code_len"], torch.tensor([5, 5, 7], dtype=torch.int32))
    torch.testing.assert_close(mm["codec_streaming"], torch.tensor([1, 1, 0], dtype=torch.int8))
    assert mm["ref_code"] == [ref_code]


def test_build_qwen3_tts_talker_multimodal_outputs_handles_empty_inputs():
    span_len, mm = build_qwen3_tts_talker_multimodal_outputs(None)

    assert span_len == 0
    assert mm == {}


def test_build_qwen3_tts_talker_multimodal_outputs_rejects_invalid_ref_code_len_list():
    with pytest.raises(ValueError, match="ref_code_len must be scalar or 1-element list"):
        build_qwen3_tts_talker_multimodal_outputs(
            [{"audio_codes": torch.tensor([[1, 2]], dtype=torch.long), "ref_code_len": [1, 2]}]
        )


def test_build_qwen3_tts_talker_multimodal_outputs_rejects_empty_ref_code_len_tensor():
    with pytest.raises(ValueError, match="ref_code_len is an empty tensor"):
        build_qwen3_tts_talker_multimodal_outputs(
            [{"audio_codes": torch.tensor([[1, 2]], dtype=torch.long), "ref_code_len": torch.tensor([])}]
        )


def test_build_qwen3_tts_talker_multimodal_outputs_ignores_nondict_entries():
    span_len, mm = build_qwen3_tts_talker_multimodal_outputs(
        [
            None,
            {"audio_codes": torch.tensor([[7, 8]], dtype=torch.long), "ref_code_len": 3},
            "skip-me",
        ]
    )

    assert span_len == 1
    torch.testing.assert_close(mm["audio_codes"], torch.tensor([[7, 8]], dtype=torch.long))
    torch.testing.assert_close(mm["ref_code_len"], torch.tensor([3], dtype=torch.int32))


def test_qwen3_tts_talker_gpu_resident_keys_contract():
    assert QWEN3_TTS_TALKER_GPU_RESIDENT_BUFFER_KEYS == frozenset(
        {"audio_codes", "last_talker_hidden", "tts_pad_embed", "tailing_text_hidden"}
    )
