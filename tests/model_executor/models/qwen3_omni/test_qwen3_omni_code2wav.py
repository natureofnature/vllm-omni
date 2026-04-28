from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import (
    Qwen3OmniMoeForConditionalGeneration,
)
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_code2wav import (
    Qwen3OmniMoeCode2Wav,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_chunked_decode_streaming_normalizes_short_left_context_list():
    model = object.__new__(Qwen3OmniMoeCode2Wav)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(num_quantizers=16)
    model.total_upsample = 2
    model.forward = lambda codes: torch.arange(8, dtype=torch.float32).reshape(2, 1, 4)

    outputs = model.chunked_decode_streaming(
        torch.zeros((2, 16, 2), dtype=torch.long),
        left_context_size=[1],
        seq_token_counts=[32, 16],
    )

    assert len(outputs) == 2
    assert outputs[0].shape[-1] == 2
    assert outputs[1].shape[-1] == 2


def test_code2wav_stage_defaults_missing_left_context_entries_per_request():
    model = object.__new__(Qwen3OmniMoeForConditionalGeneration)
    nn.Module.__init__(model)
    model.model_stage = "code2wav"

    captured: dict[str, object] = {}

    def _capture_generate_audio(code, voice_type, left_context_size, seq_token_counts):
        captured["code_shape"] = tuple(code.shape)
        captured["voice_type"] = voice_type
        captured["left_context_size"] = left_context_size
        captured["seq_token_counts"] = seq_token_counts
        return ["ok"]

    model.generate_audio = _capture_generate_audio

    result = model.forward(
        input_ids=torch.arange(48, dtype=torch.long),
        positions=torch.arange(48, dtype=torch.long),
        voice_type="ethan",
        model_intermediate_buffer=[{"left_context_size": 3}, {}],
        seq_token_counts=[32, 16],
    )

    assert result == ["ok"]
    assert captured["code_shape"] == (2, 16, 2)


def test_code2wav_stage_prefers_runtime_codec_payloads_for_mixed_batch():
    model = object.__new__(Qwen3OmniMoeForConditionalGeneration)
    nn.Module.__init__(model)
    model.model_stage = "code2wav"

    captured: dict[str, object] = {}

    def _capture_generate_audio(code, voice_type, left_context_size, seq_token_counts):
        captured["code"] = code.clone()
        captured["voice_type"] = voice_type
        captured["left_context_size"] = left_context_size
        captured["seq_token_counts"] = seq_token_counts
        return ["ok"]

    model.generate_audio = _capture_generate_audio

    req0_codes = torch.arange(32, dtype=torch.long)
    req1_codes = torch.arange(100, 116, dtype=torch.long)

    result = model.forward(
        input_ids=torch.arange(33, dtype=torch.long),
        positions=torch.arange(33, dtype=torch.long),
        voice_type="ethan",
        model_intermediate_buffer=[
            {"code_predictor_codes": req0_codes, "left_context_size": 3},
            {"code_predictor_codes": req1_codes, "left_context_size": 1},
        ],
        seq_token_counts=[32, 1],
    )

    assert result == ["ok"]
    assert captured["voice_type"] == "ethan"
    assert captured["left_context_size"] == [3, 1]
    assert captured["seq_token_counts"] == [32, 16]
    assert tuple(captured["code"].shape) == (2, 16, 2)
    assert torch.equal(captured["code"][0], req0_codes.reshape(16, 2))
    assert torch.equal(captured["code"][1, :, 0], req1_codes.reshape(16, 1).squeeze(-1))


def test_code2wav_stage_skips_async_entries_without_runtime_codec_payloads():
    model = object.__new__(Qwen3OmniMoeForConditionalGeneration)
    nn.Module.__init__(model)
    model.model_stage = "code2wav"
    model.vllm_config = SimpleNamespace(model_config=SimpleNamespace(async_chunk=True))

    captured: dict[str, object] = {}

    def _capture_generate_audio(code, voice_type, left_context_size, seq_token_counts):
        captured["code"] = code.clone()
        captured["voice_type"] = voice_type
        captured["left_context_size"] = left_context_size
        captured["seq_token_counts"] = seq_token_counts
        return [torch.tensor([1.0], dtype=torch.float32)]

    model.generate_audio = _capture_generate_audio

    req0_codes = torch.arange(32, dtype=torch.long)

    result = model.forward(
        input_ids=torch.arange(33, dtype=torch.long),
        positions=torch.arange(33, dtype=torch.long),
        voice_type="ethan",
        model_intermediate_buffer=[
            {"code_predictor_codes": req0_codes, "left_context_size": 3},
            {},
        ],
        seq_token_counts=[32, 1],
    )

    assert captured["voice_type"] == "ethan"
    assert captured["left_context_size"] == [3]
    assert captured["seq_token_counts"] == [32]
    assert tuple(captured["code"].shape) == (1, 16, 2)
    assert torch.equal(result[0], torch.tensor([1.0], dtype=torch.float32))
    assert isinstance(result[1], torch.Tensor)
    assert result[1].numel() == 0


def test_code2wav_stage_normalizes_serialized_left_context_sizes():
    model = object.__new__(Qwen3OmniMoeForConditionalGeneration)
    nn.Module.__init__(model)
    model.model_stage = "code2wav"

    captured: dict[str, object] = {}

    def _capture_generate_audio(code, voice_type, left_context_size, seq_token_counts):
        captured["left_context_size"] = left_context_size
        captured["seq_token_counts"] = seq_token_counts
        return ["ok", "ok"]

    model.generate_audio = _capture_generate_audio

    req0_codes = torch.arange(32, dtype=torch.long)
    req1_codes = torch.arange(16, dtype=torch.long)

    result = model.forward(
        input_ids=torch.arange(48, dtype=torch.long),
        positions=torch.arange(48, dtype=torch.long),
        voice_type="ethan",
        model_intermediate_buffer=[
            {"code_predictor_codes": req0_codes, "left_context_size": torch.tensor([3], dtype=torch.int32)},
            {"code_predictor_codes": req1_codes, "left_context_size": [1]},
        ],
        seq_token_counts=[32, 16],
    )

    assert result == ["ok", "ok"]
    assert captured["left_context_size"] == [3, 1]
    assert captured["seq_token_counts"] == [32, 16]


def test_code2wav_stage_pads_malformed_runtime_payload_without_dropping_sibling():
    model = object.__new__(Qwen3OmniMoeForConditionalGeneration)
    nn.Module.__init__(model)
    model.model_stage = "code2wav"

    captured: dict[str, object] = {}

    def _capture_generate_audio(code, voice_type, left_context_size, seq_token_counts):
        captured["code"] = code.clone()
        captured["seq_token_counts"] = seq_token_counts
        return ["first", "second"]

    model.generate_audio = _capture_generate_audio

    malformed = torch.arange(18, dtype=torch.long)
    valid = torch.arange(100, 132, dtype=torch.long)

    result = model.forward(
        input_ids=torch.arange(50, dtype=torch.long),
        positions=torch.arange(50, dtype=torch.long),
        voice_type="ethan",
        model_intermediate_buffer=[
            {"code_predictor_codes": malformed, "left_context_size": 0},
            {"code_predictor_codes": valid, "left_context_size": 2},
        ],
        seq_token_counts=[18, 32],
    )

    assert result == ["first", "second"]
    assert captured["seq_token_counts"] == [32, 32]
    assert tuple(captured["code"].shape) == (2, 16, 2)
    assert torch.equal(captured["code"][1], valid.reshape(16, 2))
    assert torch.equal(captured["code"][0].reshape(-1)[:18], malformed)
    assert torch.count_nonzero(captured["code"][0].reshape(-1)[18:]) == 0
