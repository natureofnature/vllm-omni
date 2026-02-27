import pytest
import torch

from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.distributed.omni_connectors.kv_transfer_manager import (
    OmniKVCacheConfig,
    OmniKVTransferManager,
)
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

    data = mock_connector.store[expected_key]
    assert data["request_id"] == req_id
    assert "layer_blocks" in data
    assert len(data["layer_blocks"]["key_cache"]) == num_layers

    # Verify shape of extracted tensor: [seq_len, heads, dim]
    # Note: Manager detaches and moves to CPU
    expected_shape = (seq_len, num_heads, head_dim)
    assert data["layer_blocks"]["key_cache"][0].shape == expected_shape


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

    data = mock_connector.store[expected_key]
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

    data = mock_connector.store[expected_key]
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

    manager = OmniKVTransferManager(kv_config)
    normalized = manager._normalize_layer_kv(layer_kv, req_id=req_id, layer_idx=0)
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


def test_integration_flow(common_constants):
    """Simulate extraction -> connector -> reception."""
    num_layers = common_constants["num_layers"]
    block_size = common_constants["block_size"]
    num_heads = common_constants["num_heads"]
    head_dim = common_constants["head_dim"]
    req_id = common_constants["req_id"]

    sender_config = OmniKVCacheConfig(
        connector_config={"type": "mock"}, from_stage="sender", to_stage="receiver", need_send_cache=True
    )
    sender_manager = OmniKVTransferManager(sender_config)
    connector = MockConnector()
    sender_manager._connector = connector  # Shared connector

    # Create Data
    num_blocks = 5
    kv_caches = []
    for _ in range(num_layers):
        layer = torch.randn(2, num_blocks, block_size, num_heads, head_dim)
        kv_caches.append(layer)

    finished_reqs = {req_id: {"block_ids": [0, 1], "seq_len": 10}}

    # Send
    sender_manager.handle_finished_requests_kv_transfer(finished_reqs, kv_caches, block_size, "float32")

    receiver_config = OmniKVCacheConfig(
        connector_config={"type": "mock"},
        from_stage="sender",
        stage_id="receiver",
        need_recv_cache=True,
        recv_timeout=1.0,
    )
    receiver_manager = OmniKVTransferManager(receiver_config)
    # Share the same mock connector instance
    receiver_manager._connector = connector

    req = OmniDiffusionRequest(
        prompts=["test_integ"],
        sampling_params=OmniDiffusionSamplingParams(),
        request_ids=[req_id],
    )

    # Receive
    data, size = receiver_manager.receive_kv_cache_for_request(req_id)

    # Verify
    assert size > 0
    assert data is not None
    receiver_manager.apply_kv_cache_to_request(req, data)
    assert req.past_key_values is not None
    assert req.kv_metadata["seq_len"] == 10


def test_manager_extraction_no_connector(kv_config, common_constants):
    """Test extraction when connector is unavailable (should still return IDs)."""
    block_size = common_constants["block_size"]
    req_id = common_constants["req_id"]

    manager = OmniKVTransferManager(kv_config)
    # Force connector to be None
    manager._connector = None
    manager.config.connector_config = None
    finished_reqs = {req_id: {"block_ids": [1, 2], "seq_len": 10}}

    processed = manager.handle_finished_requests_kv_transfer(
        finished_reqs, kv_caches=[], block_size=block_size, cache_dtype="float32"
    )

    assert req_id in processed


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
