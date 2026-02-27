from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.sched.output import NewRequestData

from vllm_omni.core.sched.output import OmniNewRequestData
from vllm_omni.engine import AdditionalInformationEntry, AdditionalInformationPayload

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_omni_new_request_data_copies_payloads():
    prompt_embeds = torch.randn(2, 3)
    additional_information = {
        "speaker": ["test"],
        "codes": torch.tensor([1, 2], dtype=torch.int64),
    }
    request = SimpleNamespace(
        request_id="req-1",
        external_req_id="ext-1",
        prompt_token_ids=[101, 102],
        mm_features=None,
        sampling_params=None,
        pooling_params=None,
        num_computed_tokens=0,
        lora_request=None,
        prompt_embeds=prompt_embeds,
        additional_information=additional_information,
    )

    data = OmniNewRequestData.from_request(request, ([0, 1],), prefill_token_ids=[101, 102])

    assert data.prompt_embeds is prompt_embeds
    assert data.initial_model_buffer is additional_information
    assert data.prefill_token_ids == [101, 102]


def test_omni_new_request_data_allows_missing_payloads():
    request = SimpleNamespace(
        request_id="req-2",
        external_req_id="ext-2",
        prompt_token_ids=[201, 202],
        mm_features=None,
        sampling_params=None,
        pooling_params=None,
        num_computed_tokens=0,
        lora_request=None,
        prompt_embeds=None,
        additional_information=None,
    )

    data = OmniNewRequestData.from_request(request, ([0],), prefill_token_ids=None)

    assert data.prompt_embeds is None
    assert data.initial_model_buffer is None


def test_omni_new_request_data_decodes_serialized_initial_buffer():
    tensor = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    payload = AdditionalInformationPayload(
        entries={
            "codes": AdditionalInformationEntry(
                tensor_data=tensor.numpy().tobytes(),
                tensor_shape=list(tensor.shape),
                tensor_dtype="float32",
            ),
            "speaker": AdditionalInformationEntry(list_data=["test"]),
        }
    )
    request = SimpleNamespace(
        request_id="req-3",
        external_req_id="ext-3",
        prompt_token_ids=[301],
        mm_features=None,
        sampling_params=None,
        pooling_params=None,
        num_computed_tokens=0,
        lora_request=None,
        prompt_embeds=None,
        additional_information=payload,
    )

    data = OmniNewRequestData.from_request(request, ([0],), prefill_token_ids=None)

    assert torch.allclose(data.initial_model_buffer["codes"], tensor)
    assert data.initial_model_buffer["speaker"] == ["test"]


def test_omni_new_request_data_wrap_preserves_prefill_token_ids():
    prompt_embeds = torch.randn(2, 3)
    initial_model_buffer = {"speaker": ["test"]}
    scheduled = NewRequestData(
        req_id="req-4",
        prompt_token_ids=[401, 402],
        mm_features=None,
        sampling_params=None,
        pooling_params=None,
        block_ids=([0, 1],),
        num_computed_tokens=0,
        lora_request=None,
        prefill_token_ids=[401, 402],
    )
    request = SimpleNamespace(
        external_req_id="ext-4",
        prompt_embeds=prompt_embeds,
        additional_information=initial_model_buffer,
    )

    wrapped = OmniNewRequestData.from_scheduled_request_data(scheduled, request)

    assert wrapped.external_req_id == "ext-4"
    assert wrapped.prompt_embeds is prompt_embeds
    assert wrapped.prefill_token_ids == [401, 402]
    assert wrapped.initial_model_buffer is initial_model_buffer


def test_omni_new_request_data_wrap_prefers_runtime_seed_buffer():
    scheduled = NewRequestData(
        req_id="req-5",
        prompt_token_ids=[501],
        mm_features=None,
        sampling_params=None,
        pooling_params=None,
        block_ids=([0],),
        num_computed_tokens=0,
        lora_request=None,
        prefill_token_ids=[501],
    )
    request = SimpleNamespace(
        external_req_id="ext-5",
        prompt_embeds=None,
        additional_information={"speaker": ["alice"]},
        _omni_initial_model_buffer={"left_context_size": 25},
    )

    wrapped = OmniNewRequestData.from_scheduled_request_data(scheduled, request)

    assert wrapped.initial_model_buffer == {"left_context_size": 25}


def test_omni_new_request_data_from_request_prefers_runtime_seed_buffer():
    request = SimpleNamespace(
        request_id="req-6",
        external_req_id="ext-6",
        prompt_token_ids=[601],
        mm_features=None,
        sampling_params=None,
        pooling_params=None,
        num_computed_tokens=0,
        lora_request=None,
        prompt_embeds=None,
        additional_information={"speaker": ["alice"]},
        _omni_initial_model_buffer={"left_context_size": 25},
    )

    data = OmniNewRequestData.from_request(request, ([0],), prefill_token_ids=None)

    assert data.initial_model_buffer == {"left_context_size": 25}
