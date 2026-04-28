import json
import struct

import numpy as np
import pytest
import torch

import vllm_omni.distributed.omni_connectors.kv_transfer_manager as kv_transfer_manager_module
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.distributed.omni_connectors.kv_transfer_manager import (
    KVCacheTransferData,
    OmniKVCacheConfig,
    OmniKVTransferManager,
)
from vllm_omni.distributed.omni_connectors.utils.kv_utils import normalize_layer_kv
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.cache]


class MockConnector:
    def __init__(self):
        self.store = {}

    def put(self, from_stage, to_stage, put_key, data):
        # The manager now passes full key as put_key
        key = f"{from_stage}->{to_stage}:{put_key}"
        self.store[key] = data
        return True, len(str(data)), None  # (success, size, metadata)

    def get(self, from_stage, to_stage, get_key, metadata=None):
        # The manager now passes full key as get_key
        key = f"{from_stage}->{to_stage}:{get_key}"
        if key in self.store:
            return self.store[key], len(str(self.store[key]))
        return None


@pytest.fixture
def mock_connector():
    return MockConnector()


@pytest.fixture
def kv_config():
    return OmniKVCacheConfig(
        connector_config={"type": "mock"},
        from_stage="stage1",
        to_stage="stage2",
        stage_id="stage2",  # Acting as receiver for some tests
        need_recv_cache=True,
        need_send_cache=True,
        recv_timeout=1.0,  # Short timeout for tests
    )


@pytest.fixture
def common_constants():
    return {
        "num_layers": 2,
        "num_heads": 4,
        "head_dim": 16,
        "block_size": 8,
        "seq_len": 20,
        "req_id": "req_test_1",
    }


def _decode_stored_payload(data):
    if isinstance(data, torch.Tensor) and data.dtype == torch.uint8 and data.dim() == 1:
        return KVCacheTransferData.from_bytes(data.cpu().numpy().tobytes())

    if isinstance(data, (bytes, bytearray, memoryview)):
        return KVCacheTransferData.from_bytes(data)

    return data


def _make_serialized_payload() -> tuple[bytes, torch.Tensor]:
    key_tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    payload = KVCacheTransferData(
        request_id="req-payload",
        layer_blocks={"key_cache": [key_tensor], "value_cache": [None]},
        block_ids=[1],
        metadata={"seq_len": 3},
    ).to_bytes()
    return payload, key_tensor


def _rewrite_serialized_header(payload: bytes, mutate_header) -> bytes:
    header_len = struct.unpack(">I", payload[:4])[0]
    header = json.loads(payload[4 : 4 + header_len])
    mutate_header(header)
    new_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return struct.pack(">I", len(new_header)) + new_header + payload[4 + header_len :]


def test_manager_extraction(kv_config, mock_connector, common_constants):
    """Test extraction and sending logic in OmniKVTransferManager."""
    num_layers = common_constants["num_layers"]
    block_size = common_constants["block_size"]
    num_heads = common_constants["num_heads"]
    head_dim = common_constants["head_dim"]
    seq_len = common_constants["seq_len"]
    req_id = common_constants["req_id"]

    num_blocks = 10
    kv_caches = []
    for _ in range(num_layers):
        k_cache = torch.randn(num_blocks, block_size, num_heads, head_dim)
        v_cache = torch.randn(num_blocks, block_size, num_heads, head_dim)
        # Stack K and V to create [2, num_blocks, block_size, n_heads, head_dim]
        layer_cache = torch.stack([k_cache, v_cache], dim=0)
        kv_caches.append(layer_cache)

    block_ids = [1, 3, 5]
    finished_reqs = {req_id: {"block_ids": block_ids, "seq_len": seq_len}}

    manager = OmniKVTransferManager(kv_config)
    # Mock the connector factory or injection
    manager._connector = mock_connector

    processed = manager.handle_finished_requests_kv_transfer(finished_reqs, kv_caches, block_size, "float32")

    assert req_id in processed

    # Check if data was put into connector
    # Manager builds full key: omni_{from}_to_{to}_kv_cache_{req_id}
    full_request_id = f"omni_stage1_to_stage2_kv_cache_{req_id}"
    expected_key = f"stage1->stage2:{full_request_id}"
    assert expected_key in mock_connector.store

    data = _decode_stored_payload(mock_connector.store[expected_key])
    assert data["request_id"] == req_id
    assert "layer_blocks" in data
    assert len(data["layer_blocks"]["key_cache"]) == num_layers

    # Verify shape of extracted tensor: [seq_len, heads, dim]
    # Note: Manager detaches and moves to CPU
    expected_shape = (seq_len, num_heads, head_dim)
    assert data["layer_blocks"]["key_cache"][0].shape == expected_shape


def test_from_bytes_rejects_out_of_bounds_header_len():
    payload, _ = _make_serialized_payload()
    bad_payload = struct.pack(">I", len(payload)) + payload[4:]

    with pytest.raises(ValueError, match="header_len"):
        KVCacheTransferData.from_bytes(bad_payload)

    with pytest.raises(ValueError, match="header_len"):
        KVCacheTransferData.from_bytes_gpu(torch.tensor(list(bad_payload), dtype=torch.uint8))


def test_from_bytes_rejects_out_of_bounds_tensor_span():
    payload, _ = _make_serialized_payload()
    bad_payload = _rewrite_serialized_header(payload, lambda header: header["td"][0].update({"o": 4096}))

    with pytest.raises(ValueError, match="tensor span"):
        KVCacheTransferData.from_bytes(bad_payload)

    with pytest.raises(ValueError, match="tensor span"):
        KVCacheTransferData.from_bytes_gpu(torch.tensor(list(bad_payload), dtype=torch.uint8))


def test_from_bytes_rejects_unsupported_dtype():
    payload, _ = _make_serialized_payload()
    bad_payload = _rewrite_serialized_header(payload, lambda header: header["td"][0].update({"d": "cuda"}))

    with pytest.raises(ValueError, match="Unsupported dtype"):
        KVCacheTransferData.from_bytes(bad_payload)

    with pytest.raises(ValueError, match="Unsupported dtype"):
        KVCacheTransferData.from_bytes_gpu(torch.tensor(list(bad_payload), dtype=torch.uint8))


def test_from_bytes_uses_explicit_layer_index_descriptor():
    payload, key_tensor = _make_serialized_payload()
    payload_with_explicit_index = _rewrite_serialized_header(
        payload,
        lambda header: header["td"][0].update({"n": "key_cache_extra_suffix", "i": 0}),
    )

    data = KVCacheTransferData.from_bytes(payload_with_explicit_index)

    assert torch.equal(data["layer_blocks"]["key_cache"][0], key_tensor)


def test_update_sender_info_uses_configured_source_stage():
    config = OmniKVCacheConfig(
        connector_config={"type": "mock"},
        stage_id=2,
        engine_input_source=[1],
        need_recv_cache=True,
    )
    manager = OmniKVTransferManager(config)

    manager.update_sender_info(
        {
            0: {"host": "10.0.0.1", "zmq_port": 50151},
            1: {"host": "10.0.0.2", "zmq_port": 50152},
        }
    )

    assert manager.config.connector_config["sender_host"] == "10.0.0.2"
    assert manager.config.connector_config["sender_zmq_port"] == 50152


def test_clone_received_payload_tensors_breaks_buffer_alias():
    payload, key_tensor = _make_serialized_payload()
    raw = np.frombuffer(bytearray(payload), dtype=np.uint8)
    data = KVCacheTransferData.from_bytes(memoryview(raw))

    OmniKVTransferManager._clone_received_payload_tensors(data)
    raw[:] = 0

    assert torch.equal(data["layer_blocks"]["key_cache"][0], key_tensor)


def test_receive_kv_cache_uses_exponential_backoff(monkeypatch):
    config = OmniKVCacheConfig(
        connector_config={"type": "mock"},
        from_stage="sender",
        stage_id="receiver",
        need_recv_cache=True,
        recv_timeout=0.3,
    )
    manager = OmniKVTransferManager(config)

    class _NeverReadyConnector:
        def get(self, **kwargs):
            del kwargs
            return None

    manager._connector = _NeverReadyConnector()

    now = {"value": 0.0}
    sleep_intervals = []

    monkeypatch.setattr(kv_transfer_manager_module.time, "time", lambda: now["value"])

    def _fake_sleep(interval: float) -> None:
        sleep_intervals.append(interval)
        now["value"] += interval

    monkeypatch.setattr(kv_transfer_manager_module.time, "sleep", _fake_sleep)

    data, size = manager.receive_kv_cache_for_request("req-backoff")

    assert (data, size) == (None, 0)
    assert sleep_intervals == pytest.approx([0.01, 0.02, 0.04, 0.08, 0.16])


def test_manager_extraction_tuple_layout(kv_config, mock_connector, common_constants):
    """Test extraction with tuple layout."""
    num_layers = common_constants["num_layers"]
    block_size = common_constants["block_size"]
    num_heads = common_constants["num_heads"]
    head_dim = common_constants["head_dim"]
    seq_len = common_constants["seq_len"]
    req_id = common_constants["req_id"]

    num_blocks = 10
    kv_caches = []
    for _ in range(num_layers):
        k_cache = torch.randn(num_blocks, block_size, num_heads, head_dim)
        v_cache = torch.randn(num_blocks, block_size, num_heads, head_dim)
        kv_caches.append((k_cache, v_cache))

    block_ids = [1, 3, 5]
    finished_reqs = {req_id: {"block_ids": block_ids, "seq_len": seq_len}}

    manager = OmniKVTransferManager(kv_config)
    manager._connector = mock_connector

    processed = manager.handle_finished_requests_kv_transfer(finished_reqs, kv_caches, block_size, "float32")
    assert req_id in processed

    full_request_id = f"omni_stage1_to_stage2_kv_cache_{req_id}"
    expected_key = f"stage1->stage2:{full_request_id}"
    assert expected_key in mock_connector.store

    data = _decode_stored_payload(mock_connector.store[expected_key])
    expected_shape = (seq_len, num_heads, head_dim)
    for idx in range(len(kv_caches)):
        assert data["layer_blocks"]["key_cache"][idx].shape == expected_shape
        assert data["layer_blocks"]["value_cache"][idx].shape == expected_shape


def test_manager_extraction_mismatched_kv_block_counts(kv_config, mock_connector, common_constants):
    """Mismatched key/value block counts should not crash extraction."""
    block_size = common_constants["block_size"]
    num_heads = common_constants["num_heads"]
    head_dim = common_constants["head_dim"]
    req_id = common_constants["req_id"]

    key_blocks = torch.randn(3, block_size, num_heads, head_dim)
    value_blocks = torch.randn(2, block_size, num_heads, head_dim)
    kv_caches = [(key_blocks, value_blocks)]

    finished_reqs = {req_id: {"block_ids": [0, 1, 2], "seq_len": 32}}

    manager = OmniKVTransferManager(kv_config)
    manager._connector = mock_connector

    processed = manager.handle_finished_requests_kv_transfer(finished_reqs, kv_caches, block_size, "float32")
    assert req_id in processed

    full_request_id = f"omni_stage1_to_stage2_kv_cache_{req_id}"
    expected_key = f"stage1->stage2:{full_request_id}"
    assert expected_key in mock_connector.store

    data = _decode_stored_payload(mock_connector.store[expected_key])
    expected_shape = (2 * block_size, num_heads, head_dim)
    assert data["layer_blocks"]["key_cache"][0].shape == expected_shape
    assert data["layer_blocks"]["value_cache"][0].shape == expected_shape


@pytest.mark.parametrize(
    "invalid_case",
    ["invalid_stacked_shape", "invalid_tuple_length", "non_tensor_entries"],
)
def test_normalize_layer_kv_rejects_invalid_inputs(kv_config, common_constants, invalid_case):
    """_normalize_layer_kv should reject malformed KV representations."""
    block_size = common_constants["block_size"]
    num_heads = common_constants["num_heads"]
    head_dim = common_constants["head_dim"]
    req_id = common_constants["req_id"]

    if invalid_case == "invalid_stacked_shape":
        layer_kv = torch.randn(3, block_size, num_heads, head_dim)
    elif invalid_case == "invalid_tuple_length":
        layer_kv = (
            torch.randn(2, block_size, num_heads, head_dim),
            torch.randn(2, block_size, num_heads, head_dim),
            torch.randn(2, block_size, num_heads, head_dim),
        )
    else:
        layer_kv = (torch.randn(2, block_size, num_heads, head_dim), "not-a-tensor")

    normalized = normalize_layer_kv(layer_kv, req_id=req_id, layer_idx=0)
    assert normalized is None


def test_manager_reception(kv_config, mock_connector, common_constants):
    """Test raw reception plus explicit request application logic."""
    num_layers = common_constants["num_layers"]
    block_size = common_constants["block_size"]
    num_heads = common_constants["num_heads"]
    head_dim = common_constants["head_dim"]
    seq_len = common_constants["seq_len"]
    req_id = common_constants["req_id"]

    expected_shape = (seq_len, num_heads, head_dim)
    key_cache = [torch.randn(expected_shape) for _ in range(num_layers)]
    value_cache = [torch.randn(expected_shape) for _ in range(num_layers)]

    layer_blocks = {"key_cache": key_cache, "value_cache": value_cache}
    metadata = {
        "block_size": block_size,
        "num_layers": num_layers,
        "dtype": "float32",
        "seq_len": seq_len,
    }

    data_to_receive = {
        "request_id": req_id,
        "layer_blocks": layer_blocks,
        "metadata": metadata,
        "block_ids": [],
    }

    manager = OmniKVTransferManager(kv_config)
    manager._connector = mock_connector

    full_request_id = f"omni_stage1_to_stage2_kv_cache_{req_id}"
    store_key = f"stage1->stage2:{full_request_id}"
    mock_connector.store[store_key] = data_to_receive

    req = OmniDiffusionRequest(
        prompts=["test_recv"],
        sampling_params=OmniDiffusionSamplingParams(),
        request_ids=[req_id],
    )

    data, size = manager.receive_kv_cache_for_request(req_id, target_device=torch.device("cpu"))
    assert size > 0
    assert data is not None

    manager.apply_kv_cache_to_request(req, data)

    assert hasattr(req, "past_key_values")
    assert hasattr(req, "kv_metadata")
    assert len(req.past_key_values.key_cache) == num_layers
    assert torch.allclose(req.past_key_values.key_cache[0], key_cache[0])
    assert req.kv_metadata["seq_len"] == seq_len


def test_rank_aware_manager_uses_send_key_builder(kv_config, mock_connector, common_constants):
    manager = OmniKVTransferManager(kv_config)
    manager._connector = mock_connector
    manager.kv_send_key_builder = lambda request_id, from_stage, to_stage: [
        f"{request_id}_{from_stage}_0_0_0",
        f"{request_id}_{from_stage}_0_0_1",
    ]

    num_layers = common_constants["num_layers"]
    block_size = common_constants["block_size"]
    num_heads = common_constants["num_heads"]
    head_dim = common_constants["head_dim"]
    seq_len = common_constants["seq_len"]
    req_id = common_constants["req_id"]

    kv_caches = []
    for _ in range(num_layers):
        layer = torch.randn(2, 4, block_size, num_heads, head_dim)
        kv_caches.append(layer)

    finished_reqs = {req_id: {"block_ids": [0, 1], "seq_len": seq_len}}
    manager.handle_finished_requests_kv_transfer(finished_reqs, kv_caches, block_size, "float32")

    assert f"stage1->stage2:{req_id}_stage1_0_0_0" in mock_connector.store
    assert f"stage1->stage2:{req_id}_stage1_0_0_1" in mock_connector.store


def test_rank_aware_manager_merges_received_payloads(kv_config, mock_connector, common_constants):
    manager = OmniKVTransferManager(kv_config)
    manager._connector = mock_connector
    manager.kv_recv_key_builder = lambda request_id, from_stage, to_stage: [
        f"{request_id}_{from_stage}_0_0_0",
        f"{request_id}_{from_stage}_0_1_0",
    ]
    manager.kv_payload_merger = lambda payloads: {
        "request_id": payloads[0]["request_id"],
        "layer_blocks": {
            "key_cache": [
                torch.cat(
                    [payloads[0]["layer_blocks"]["key_cache"][0], payloads[1]["layer_blocks"]["key_cache"][0]], dim=1
                )
            ],
            "value_cache": [
                torch.cat(
                    [payloads[0]["layer_blocks"]["value_cache"][0], payloads[1]["layer_blocks"]["value_cache"][0]],
                    dim=1,
                )
            ],
        },
        "metadata": {"seq_len": common_constants["seq_len"]},
        "block_ids": [],
    }

    req_id = common_constants["req_id"]
    key0 = f"stage1->stage2:{req_id}_stage1_0_0_0"
    key1 = f"stage1->stage2:{req_id}_stage1_0_1_0"
    mock_connector.store[key0] = {
        "request_id": req_id,
        "layer_blocks": {
            "key_cache": [torch.ones(2, 1, 3)],
            "value_cache": [torch.ones(2, 1, 3)],
        },
        "metadata": {"seq_len": common_constants["seq_len"]},
        "block_ids": [],
    }
    mock_connector.store[key1] = {
        "request_id": req_id,
        "layer_blocks": {
            "key_cache": [torch.full((2, 1, 3), 2.0)],
            "value_cache": [torch.full((2, 1, 3), 2.0)],
        },
        "metadata": {"seq_len": common_constants["seq_len"]},
        "block_ids": [],
    }

    data, size = manager.receive_kv_cache_for_request(req_id)
    assert size > 0
    assert data is not None
    assert tuple(data["layer_blocks"]["key_cache"][0].shape) == (2, 2, 3)


def test_rank_aware_manager_slices_received_payloads(kv_config, mock_connector, common_constants):
    manager = OmniKVTransferManager(kv_config)
    manager._connector = mock_connector
    manager.kv_recv_key_builder = lambda request_id, from_stage, to_stage: [
        f"{request_id}_{from_stage}_0_0_0",
    ]

    def _slice(payload):
        return {
            "request_id": payload["request_id"],
            "layer_blocks": {
                "key_cache": [payload["layer_blocks"]["key_cache"][0][:, :1, :]],
                "value_cache": [payload["layer_blocks"]["value_cache"][0][:, :1, :]],
            },
            "metadata": {**payload["metadata"], "sliced": True},
            "block_ids": payload["block_ids"],
        }

    manager.kv_payload_slicer = _slice

    req_id = common_constants["req_id"]
    key = f"stage1->stage2:{req_id}_stage1_0_0_0"
    mock_connector.store[key] = {
        "request_id": req_id,
        "layer_blocks": {
            "key_cache": [torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)],
            "value_cache": [torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)],
        },
        "metadata": {"seq_len": common_constants["seq_len"]},
        "block_ids": [],
    }

    data, size = manager.receive_kv_cache_for_request(req_id)
    assert size > 0
    assert data is not None
    assert data["metadata"]["sliced"] is True
    assert tuple(data["layer_blocks"]["key_cache"][0].shape) == (2, 1, 3)
