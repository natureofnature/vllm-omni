from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.qwen3_omni import thinker2talker_async_chunk

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
