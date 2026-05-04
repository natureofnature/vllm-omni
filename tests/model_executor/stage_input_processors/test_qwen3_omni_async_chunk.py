from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.qwen3_omni import (
    talker2code2wav_async_chunk,
    talker2code2wav_full_payload,
    thinker2talker_async_chunk,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_explicit_finish_without_payload_returns_sentinel():
    request_id = "req-qwen3-finish"
    transfer_manager = SimpleNamespace(
        put_req_chunk=defaultdict(int, {request_id: 0}),
        request_payload={},
        thinker_decode_embed_offsets={request_id: 3},
    )
    request = SimpleNamespace(
        external_req_id=request_id,
        is_finished=lambda: False,
    )

    payload = thinker2talker_async_chunk(transfer_manager, {}, request, is_finished=True)

    assert torch.equal(payload["finished"], torch.tensor(True, dtype=torch.bool))
    assert transfer_manager.request_payload == {}
    assert transfer_manager.thinker_decode_embed_offsets == {}


def test_talker2code2wav_async_chunk_includes_code_num_quantizers():
    request_id = "req-qwen3-codec"
    frame = [101, 202, 303, 404]
    transfer_manager = SimpleNamespace(
        code_prompt_token_ids=defaultdict(list, {request_id: [frame[:] for _ in range(24)]}),
        connector=SimpleNamespace(config={"extra": {"codec_chunk_frames": 25, "codec_left_context_frames": 25}}),
    )
    request = SimpleNamespace(external_req_id=request_id, is_finished=lambda: False)

    payload = talker2code2wav_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output={"code_predictor_codes": torch.tensor([frame], dtype=torch.long)},
        request=request,
        is_finished=False,
    )

    assert payload is not None
    assert payload["left_context_size"] == 0
    assert payload["code_num_quantizers"] == len(frame)
    assert payload["finished"].item() is False
    assert len(payload["code_predictor_codes"]) == 25 * len(frame)


def test_talker2code2wav_async_chunk_final_flush_includes_code_num_quantizers():
    request_id = "req-qwen3-final"
    frame = [1, 2, 3, 4]
    transfer_manager = SimpleNamespace(
        code_prompt_token_ids=defaultdict(list, {request_id: [frame[:] for _ in range(3)]}),
        connector=SimpleNamespace(config={"extra": {"codec_chunk_frames": 25, "codec_left_context_frames": 25}}),
    )

    payload = talker2code2wav_async_chunk(
        transfer_manager, {}, SimpleNamespace(external_req_id=request_id, is_finished=lambda: True), is_finished=True
    )
    assert payload["code_num_quantizers"] == len(frame)
    assert payload["finished"].item() is True


def test_talker2code2wav_full_payload_filters_placeholder_and_terminal_special_rows():
    request = SimpleNamespace(
        output_token_ids=[
            101,
            102,
            4196,
            103,
            4198,
            -1,
        ]
    )
    pooling_output = {
        "code_predictor_codes": torch.tensor(
            [
                [11, 12, 13, 14],
                [21, 22, 23, 24],
                [1999, 1998, 1997, 1996],
                [31, 32, 33, 34],
                [2048, 2049, 2050, 2051],
                [0, 0, 0, 0],
            ],
            dtype=torch.long,
        )
    }

    payload = talker2code2wav_full_payload(
        transfer_manager=SimpleNamespace(),
        pooling_output=pooling_output,
        request=request,
    )

    assert payload is not None
    assert payload["finished"].item() is True
    assert payload["code_predictor_codes"] == [
        11,
        21,
        31,
        12,
        22,
        32,
        13,
        23,
        33,
        14,
        24,
        34,
    ]


def test_talker2code2wav_full_payload_returns_none_without_valid_codec_tokens():
    request = SimpleNamespace(output_token_ids=[4198, -1])
    pooling_output = {
        "code_predictor_codes": torch.tensor(
            [
                [41, 42, 43, 44],
                [2048, 2049, 2050, 2051],
                [51, 52, 53, 54],
            ],
            dtype=torch.long,
        )
    }

    payload = talker2code2wav_full_payload(
        transfer_manager=SimpleNamespace(),
        pooling_output=pooling_output,
        request=request,
    )

    assert payload is None


def test_talker2code2wav_full_payload_does_not_fallback_to_special_row_when_aligned():
    request = SimpleNamespace(output_token_ids=[101, 4198])
    pooling_output = {
        "code_predictor_codes": torch.tensor(
            [
                [2048, 2049, 2050, 2051],
                [61, 62, 63, 64],
            ],
            dtype=torch.long,
        )
    }

    payload = talker2code2wav_full_payload(
        transfer_manager=SimpleNamespace(),
        pooling_output=pooling_output,
        request=request,
    )

    assert payload is None
