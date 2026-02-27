# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.v1.request import RequestStatus

from vllm_omni.distributed.omni_connectors.transfer_adapter.base import OmniTransferAdapterBase
from vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter import (
    OmniChunkTransferAdapter,
)
from vllm_omni.distributed.omni_connectors.utils.config import ConnectorSpec

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _req(req_id: str, status: RequestStatus, external_req_id: str | None = None):
    return SimpleNamespace(
        request_id=req_id,
        external_req_id=external_req_id or req_id,
        status=status,
        prompt_token_ids=[],
        num_computed_tokens=0,
        additional_information=None,
        is_finished=lambda: status == RequestStatus.FINISHED_STOPPED,
    )


@pytest.fixture
def build_adapter(monkeypatch):
    def _build(*, stage_id: int = 1, model_mode: str = "ar", max_num_seqs: int = 2):
        connector = MagicMock()
        connector.stage_id = stage_id
        connector.get.return_value = None
        connector.put.return_value = (True, 1, {})

        def _fake_base_init(self, config):
            self.config = config
            self._pending_load_reqs = deque()
            self._finished_load_reqs = set()
            self._cancelled_load_reqs = set()
            self._pending_save_reqs = deque()
            self.stop_event = threading.Event()
            self._recv_cond = threading.Condition()
            self._save_cond = threading.Condition()

        monkeypatch.setattr(OmniTransferAdapterBase, "__init__", _fake_base_init)
        monkeypatch.setattr(
            OmniChunkTransferAdapter,
            "create_connector",
            classmethod(lambda cls, _model_config: connector),
        )

        model_config = SimpleNamespace(worker_type=model_mode)
        scheduler_config = SimpleNamespace(max_num_seqs=max_num_seqs)
        adapter = OmniChunkTransferAdapter(
            SimpleNamespace(model_config=model_config, scheduler_config=scheduler_config)
        )
        return adapter, connector

    return _build


@pytest.mark.parametrize(
    ("raw_cfg", "expected_name", "expected_extra"),
    [
        (None, "SharedMemoryConnector", {}),
        (SimpleNamespace(name="YuanrongConnector", extra={"k": "v"}), "YuanrongConnector", {"k": "v"}),
    ],
)
def test_create_connector_config_parsing(monkeypatch, raw_cfg, expected_name, expected_extra):
    captured = {}

    def _fake_create(spec):
        captured["spec"] = spec
        return "ok"

    monkeypatch.setattr(
        "vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter"
        ".OmniConnectorFactory.create_connector",
        _fake_create,
    )

    model_config = SimpleNamespace(stage_connector_config=raw_cfg) if raw_cfg is not None else SimpleNamespace()
    connector = OmniChunkTransferAdapter.create_connector(model_config)

    assert connector == "ok"
    assert isinstance(captured["spec"], ConnectorSpec)
    assert captured["spec"].name == expected_name
    assert captured["spec"].extra == expected_extra


def test_load_poll(build_adapter):
    adapter, connector = build_adapter(stage_id=2, model_mode="ar")
    request = _req("req-1", RequestStatus.WAITING, external_req_id="external-1")

    adapter.load_async(request)
    adapter.request_ids_mapping["req-1"] = "external-1"
    payload = {"code_predictor_codes": [[1]], "hidden_states": torch.tensor([[2.0]]), "finished": True}
    connector.get.return_value = (payload, 16)
    adapter._poll_single_request(request)

    assert request.additional_information is None
    assert adapter.request_payload["external-1"] == payload
    assert adapter.get_req_chunk["req-1"] == 1
    assert "req-1" in adapter._finished_load_reqs
    assert "req-1" in adapter.finished_requests
    assert "req-1" not in adapter._pending_load_reqs


def test_save_async(build_adapter):
    adapter, _ = build_adapter(stage_id=1)
    request = _req("req-1", RequestStatus.WAITING, external_req_id="external-1")

    adapter.custom_process_next_stage_input_func = lambda **kwargs: {"x": [1], "finished": False}
    adapter.save_async(pooling_output=None, request=request)
    adapter.custom_process_next_stage_input_func = lambda **kwargs: {}
    adapter.save_async(pooling_output=None, request=request)

    task = adapter._pending_save_reqs.popleft()
    assert task["is_finished"] is False


def test_update_request_payload(build_adapter):
    adapter, _ = build_adapter()

    adapter._update_request_payload("ext", {"h": torch.tensor([[1.0]]), "codes": [1], "finished": False})
    merged = adapter._update_request_payload("ext", {"h": torch.tensor([[2.0]]), "codes": [2], "finished": True})

    assert torch.equal(merged["h"], torch.tensor([[1.0], [2.0]]))
    assert merged["codes"] == [1, 2]
    assert merged["finished"] is True


# ---------------------------------------------------------------
# Cleanup tests
# ---------------------------------------------------------------


def _populate_adapter_state(adapter, req_id="req-1", ext_id="ext-1"):
    """Fill every per-request structure so cleanup can be verified."""
    adapter.finished_requests.add(req_id)
    adapter.get_req_chunk[req_id] = 3
    adapter.requests_with_ready_chunks.add(req_id)
    adapter.request_ids_mapping[req_id] = ext_id
    adapter._pending_load_reqs.append(SimpleNamespace(request_id=req_id))
    adapter._finished_load_reqs.add(req_id)

    adapter.put_req_chunk[ext_id] = 5
    adapter.request_payload[ext_id] = {"hidden": [1, 2]}
    adapter.code_prompt_token_ids[ext_id] = [[10, 20]]


def test_cleanup_clears_all_state(build_adapter):
    """After cleanup, no per-request key should remain in any dict/set."""
    adapter, _ = build_adapter(stage_id=1)
    req_id, ext_id = "req-1", "ext-1"
    _populate_adapter_state(adapter, req_id, ext_id)

    adapter.cleanup(req_id, ext_id)

    assert req_id not in adapter.finished_requests
    assert req_id not in adapter.get_req_chunk
    assert req_id not in adapter.requests_with_ready_chunks
    assert req_id not in adapter.request_ids_mapping
    assert req_id in adapter._cancelled_load_reqs
    assert req_id not in adapter._finished_load_reqs

    assert ext_id not in adapter.put_req_chunk
    assert ext_id not in adapter.request_payload
    assert ext_id not in adapter.code_prompt_token_ids


def test_cleanup_infers_external_id(build_adapter):
    """When external_req_id is None, cleanup should look it up from the mapping."""
    adapter, _ = build_adapter(stage_id=1)
    req_id, ext_id = "req-2", "ext-2"
    _populate_adapter_state(adapter, req_id, ext_id)

    adapter.cleanup(req_id)

    assert ext_id not in adapter.put_req_chunk
    assert ext_id not in adapter.request_payload


def test_cleanup_idempotent(build_adapter):
    """Calling cleanup multiple times for the same (or nonexistent) request must not raise."""
    adapter, _ = build_adapter(stage_id=1)

    try:
        adapter.cleanup("nonexistent")
        adapter.cleanup("nonexistent")
    except Exception as e:
        pytest.fail(f"cleanup should be idempotent: {e}")

    req_id, ext_id = "req-3", "ext-3"
    _populate_adapter_state(adapter, req_id, ext_id)
    adapter.cleanup(req_id, ext_id)

    try:
        adapter.cleanup(req_id, ext_id)
    except Exception as e:
        pytest.fail(f"second cleanup should be idempotent: {e}")


def test_cleanup_request_id_reuse_not_polluted(build_adapter):
    """After cleanup, reusing the same request_id must not be treated as finished."""
    adapter, _ = build_adapter(stage_id=1)
    req_id, ext_id = "req-reuse", "ext-reuse"
    _populate_adapter_state(adapter, req_id, ext_id)

    adapter.cleanup(req_id, ext_id)

    assert req_id not in adapter.finished_requests
    assert req_id not in adapter.get_req_chunk


def test_cleanup_preserves_pending_save(build_adapter):
    """Cleanup must NOT remove _pending_save_reqs to avoid losing unsent chunks."""
    adapter, _ = build_adapter(stage_id=1)
    req_id, ext_id = "req-4", "ext-4"
    _populate_adapter_state(adapter, req_id, ext_id)

    pending_task = {"put_key": f"{ext_id}_1_0", "data": {"x": 1}}
    adapter._pending_save_reqs.append(pending_task)

    adapter.cleanup(req_id, ext_id)

    assert len(adapter._pending_save_reqs) == 1


def test_cleanup_after_poll_flow(build_adapter):
    """Cleanup after a real poll flow should clear the loaded request state."""
    adapter, connector = build_adapter(stage_id=2, model_mode="ar")
    request = _req("req-flow", RequestStatus.WAITING, external_req_id="ext-flow")

    adapter.load_async(request)
    adapter.request_ids_mapping["req-flow"] = "ext-flow"
    payload = {"hidden_states": torch.tensor([[1.0]]), "finished": True}
    connector.get.return_value = (payload, 8)
    adapter._poll_single_request(request)

    assert "req-flow" in adapter.finished_requests
    assert adapter.get_req_chunk["req-flow"] == 1
    assert "req-flow" in adapter.request_ids_mapping

    adapter.cleanup("req-flow", "ext-flow")

    assert "req-flow" not in adapter.finished_requests
    assert "req-flow" not in adapter.get_req_chunk
    assert "req-flow" not in adapter.request_ids_mapping
    assert "ext-flow" not in adapter.request_payload
