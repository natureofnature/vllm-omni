from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.qwen3_omni import (
    talker2code2wav_async_chunk,
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
