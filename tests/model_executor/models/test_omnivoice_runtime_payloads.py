# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.omnivoice.omnivoice import OmniVoiceModel
from vllm_omni.model_executor.models.output_templates import OmniOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _DummyGenerator:
    def __call__(self, **kwargs):
        return torch.arange(8, dtype=torch.long).reshape(1, 8, 1)


class _DummyDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, audio_tokens):
        return audio_tokens.to(dtype=torch.float32).sum(dim=1).squeeze(0)


def _make_generator_model() -> OmniVoiceModel:
    model = object.__new__(OmniVoiceModel)
    torch.nn.Module.__init__(model)
    model.model_stage = "omnivoice_generator"
    model.config = SimpleNamespace(
        llm_hidden_size=16,
        num_audio_codebook=8,
        audio_mask_id=99,
        num_step=1,
        guidance_scale=1.0,
        t_shift=1.0,
        layer_penalty_factor=1.0,
        position_temperature=1.0,
        class_temperature=1.0,
    )
    model.generator = _DummyGenerator()
    model._duration_estimator = SimpleNamespace(
        estimate_duration=lambda raw_text, prompt, rate: 1,
    )
    return model


def _make_decoder_model() -> OmniVoiceModel:
    model = object.__new__(OmniVoiceModel)
    torch.nn.Module.__init__(model)
    model.model_stage = "omnivoice_decoder"
    model.config = SimpleNamespace(
        llm_hidden_size=16,
        sample_rate=24000,
    )
    model.decoder = _DummyDecoder()
    return model


def test_forward_generator_prefers_model_intermediate_buffer():
    model = _make_generator_model()

    out = model._forward_generator(
        torch.arange(4, dtype=torch.long),
        {
            "model_intermediate_buffer": [{"raw_text": "hello from model buffer"}],
            "runtime_additional_information": [],
        },
    )

    assert isinstance(out, OmniOutput)
    audio_tokens = out.multimodal_outputs["audio_tokens"]
    assert tuple(audio_tokens.shape) == (1, 8, 1)


def test_forward_decoder_falls_back_to_legacy_runtime_additional_information():
    model = _make_decoder_model()

    out = model._forward_decoder(
        torch.arange(4, dtype=torch.long),
        {
            "runtime_additional_information": [{"audio_tokens": torch.arange(16, dtype=torch.long).reshape(1, 8, 2)}],
        },
    )

    assert isinstance(out, OmniOutput)
    assert out.multimodal_outputs["sr"] == 24000
    torch.testing.assert_close(
        out.multimodal_outputs["audio"],
        torch.tensor([56.0, 64.0]),
    )
