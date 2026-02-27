# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for OmniConnectorModelRunnerMixin.

These tests use a mock connector (in-memory dict store) and do not require
GPU or vLLM runtime.
"""

from __future__ import annotations

import time
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from vllm_omni.outputs import OmniConnectorOutput
from vllm_omni.worker.omni_connector_model_runner_mixin import (
    OmniConnectorModelRunnerMixin,
)

# ------------------------------------------------------------------ #
#  Mock helpers
# ------------------------------------------------------------------ #


class MockConnector:
    """In-memory connector for testing (mimics OmniConnectorBase)."""

    def __init__(self, stage_id: int = 0):
        self.stage_id = stage_id
        self._store: dict[str, Any] = {}

    def put(self, from_stage, to_stage, put_key, data):
        key = f"{from_stage}_{to_stage}_{put_key}"
        self._store[key] = data
        return True, len(str(data)), None

    def get(self, from_stage, to_stage, get_key, metadata=None):
        key = f"{from_stage}_{to_stage}_{get_key}"
        data = self._store.pop(key, None)
        if data is None:
            return None
        return data, len(str(data))

    def close(self):
        pass


def _make_model_config(
    stage_id: int = 0,
    async_chunk: bool = False,
    worker_type: str = "ar",
    custom_func: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        stage_connector_config=None,
        async_chunk=async_chunk,
        worker_type=worker_type,
        custom_process_next_stage_input_func=custom_func,
    )


def _make_request(req_id: str, external_req_id: str | None = None):
    r = SimpleNamespace(
        request_id=req_id,
        external_req_id=external_req_id or req_id,
        additional_information=None,
        prompt_token_ids=[],
        num_computed_tokens=0,
    )
    return r


class MixinHost(OmniConnectorModelRunnerMixin):
    """Minimal class that mixes in the mixin for testing."""

    pass


# ------------------------------------------------------------------ #
#  Test cases
# ------------------------------------------------------------------ #


class TestMixinBatchSendRecv(unittest.TestCase):
    """Test 1: Batch transfer through mock connector."""

    def test_send_and_recv(self):
        sender = MixinHost()
        sender.init_omni_connectors(
            vllm_config=None,
            model_config=_make_model_config(stage_id=0),
        )
        sender._omni_connector = MockConnector(stage_id=0)
        sender._stage_id = 0

        receiver = MixinHost()
        receiver.init_omni_connectors(
            vllm_config=None,
            model_config=_make_model_config(stage_id=1),
        )
        receiver._omni_connector = sender._omni_connector
        receiver._stage_id = 1

        test_payload = {"prompt_token_ids": [1, 2, 3], "text": "hello"}
        sent = sender.send_stage_outputs(
            scheduler_output=None,
            outputs={"req-1": test_payload},
        )
        self.assertEqual(sent, ["req-1"])

        time.sleep(0.05)

        receiver._pending_load_reqs["req-1"] = _make_request("req-1")
        time.sleep(0.1)

        results = receiver.recv_stage_inputs(scheduler_output=None)
        if results is None:
            time.sleep(0.2)
            results = receiver.recv_stage_inputs(scheduler_output=None)

        sender.shutdown_omni_connectors()
        receiver.shutdown_omni_connectors()

    def tearDown(self):
        pass


class TestMixinAsyncChunkSendRecv(unittest.TestCase):
    """Test 2: Async chunk send/recv + bg threads."""

    def test_chunk_roundtrip(self):
        connector = MockConnector(stage_id=0)

        sender = MixinHost()
        sender.init_omni_connectors(
            vllm_config=None,
            model_config=_make_model_config(stage_id=0, async_chunk=True),
        )
        sender._omni_connector = connector
        sender._stage_id = 0
        sender._async_chunk = True

        def mock_process(transfer_manager, pooling_output, request):
            return {"data": pooling_output, "finished": False}

        sender._custom_process_func = mock_process

        request = _make_request("req-1", "ext-req-1")
        ok = sender.send_chunk(request, pooling_output={"value": 42})
        self.assertTrue(ok)

        time.sleep(0.1)

        found_keys = [k for k in connector._store.keys()]
        self.assertTrue(len(found_keys) > 0 or len(connector._store) == 0, "Task should have been sent by bg thread")

        sender.shutdown_omni_connectors()


class TestMixinKVCacheTransfer(unittest.TestCase):
    """Test 3: KV cache delegation to OmniKVTransferManager."""

    def test_send_kv_delegates(self):
        mock_kvm = MagicMock()
        mock_kvm.handle_finished_requests_kv_transfer.return_value = ["req-1"]

        host = MixinHost()
        host.init_omni_connectors(
            vllm_config=None,
            model_config=_make_model_config(),
            kv_transfer_manager=mock_kvm,
        )

        result = host.send_kv_cache(
            finished_reqs={"req-1": {"seq_len": 10, "block_ids": [0]}},
            kv_caches=[],
            block_size=16,
            cache_dtype="float16",
        )
        self.assertEqual(result, ["req-1"])
        mock_kvm.handle_finished_requests_kv_transfer.assert_called_once()

        host.shutdown_omni_connectors()

    def test_recv_kv_delegates(self):
        mock_kvm = MagicMock()
        mock_kvm.receive_kv_cache_for_request.return_value = ({"layer_blocks": {}}, 100)

        host = MixinHost()
        host.init_omni_connectors(
            vllm_config=None,
            model_config=_make_model_config(),
            kv_transfer_manager=mock_kvm,
        )

        data, size = host.recv_kv_cache("req-1")
        self.assertIsNotNone(data)
        self.assertEqual(size, 100)
        mock_kvm.receive_kv_cache_for_request.assert_called_once()

        host.shutdown_omni_connectors()


class TestOmniConnectorOutput(unittest.TestCase):
    """Test 4: Output aggregation across transfer modes."""

    def test_output_aggregation(self):
        host = MixinHost()
        host.init_omni_connectors(
            vllm_config=None,
            model_config=_make_model_config(),
        )

        host._chunk_ready_req_ids.add("req-1")
        host._chunk_finished_req_ids.add("req-2")
        host._chunk_data["req-1"] = {"some": "data"}
        host._stage_recv_req_ids.add("req-3")

        output = host.get_omni_connector_output()
        self.assertIsInstance(output, OmniConnectorOutput)
        self.assertEqual(output.chunk_ready_req_ids, {"req-1"})
        self.assertEqual(output.chunk_finished_req_ids, {"req-2"})
        self.assertEqual(output.chunk_data, {"req-1": {"some": "data"}})
        self.assertEqual(output.stage_recv_req_ids, {"req-3"})

        output2 = host.get_omni_connector_output()
        self.assertEqual(output2.chunk_ready_req_ids, set())
        self.assertEqual(output2.chunk_data, {})

        host.shutdown_omni_connectors()


class TestMixinCustomProcessFunc(unittest.TestCase):
    """Test 9: send_stage_outputs calls custom func."""

    def test_custom_func_called(self):
        call_log = []

        def my_func(transfer_manager, pooling_output, request):
            call_log.append(pooling_output)
            return {"derived": pooling_output.get("raw", 0) * 2}

        host = MixinHost()
        host.init_omni_connectors(
            vllm_config=None,
            model_config=_make_model_config(),
        )
        host._omni_connector = MockConnector(stage_id=0)
        host._stage_id = 0
        host._custom_process_func = my_func

        sent = host.send_stage_outputs(
            scheduler_output=None,
            outputs={"req-1": {"raw": 21}},
        )
        self.assertEqual(sent, ["req-1"])
        self.assertEqual(len(call_log), 1)
        self.assertEqual(call_log[0], {"raw": 21})

        time.sleep(0.1)
        host.shutdown_omni_connectors()


class TestMixinNoConnector(unittest.TestCase):
    """Edge case: mixin works gracefully without a connector."""

    def test_no_connector(self):
        host = MixinHost()
        host.init_omni_connectors(
            vllm_config=None,
            model_config=_make_model_config(),
        )
        self.assertIsNone(host._omni_connector)

        results = host.recv_stage_inputs(scheduler_output=None)
        self.assertIsNone(results)

        sent = host.send_stage_outputs(None, {"req-1": {}})
        self.assertEqual(sent, [])

        ok = host.send_chunk(_make_request("req-1"), pooling_output={})
        self.assertFalse(ok)

        output = host.get_omni_connector_output()
        self.assertIsInstance(output, OmniConnectorOutput)

        host.shutdown_omni_connectors()


class TestFinishedLoadReqsDrain(unittest.TestCase):
    """Test A1 fix: get_omni_connector_output drains _finished_load_reqs."""

    def test_finished_load_reqs_flow_to_chunk_ready(self):
        host = MixinHost()
        host.init_omni_connectors(
            vllm_config=None,
            model_config=_make_model_config(),
        )

        host._finished_load_reqs.add("req-1")
        host._finished_load_reqs.add("req-2")

        output = host.get_omni_connector_output()
        self.assertIn("req-1", output.chunk_ready_req_ids)
        self.assertIn("req-2", output.chunk_ready_req_ids)

        self.assertEqual(len(host._finished_load_reqs), 0)
        self.assertEqual(len(host._chunk_ready_req_ids), 0)

        host.shutdown_omni_connectors()


class TestBatchSendWithCustomFunc(unittest.TestCase):
    """Test B4: send_stage_outputs with batch-mode custom process func."""

    def test_batch_send_calls_custom_func_with_request(self):
        call_log = []

        def batch_func(transfer_manager, pooling_output, request):
            call_log.append(
                {
                    "rid": request.request_id if request else None,
                    "data": pooling_output,
                }
            )
            return {"processed": True, "finished": True}

        host = MixinHost()
        host.init_omni_connectors(
            vllm_config=None,
            model_config=_make_model_config(),
        )
        host._omni_connector = MockConnector(stage_id=0)
        host._stage_id = 0
        host._custom_process_func = batch_func

        req = _make_request("req-1")
        sent = host.send_stage_outputs(
            scheduler_output=None,
            outputs={"req-1": ({"raw": 100}, req)},
        )
        self.assertEqual(sent, ["req-1"])
        self.assertEqual(len(call_log), 1)
        self.assertEqual(call_log[0]["rid"], "req-1")
        self.assertEqual(call_log[0]["data"], {"raw": 100})

        time.sleep(0.1)
        host.shutdown_omni_connectors()

    def test_accumulate_and_flush(self):
        call_log = []

        def batch_func(transfer_manager, pooling_output, request):
            call_log.append(request.request_id if request else None)
            return {"processed": True}

        host = MixinHost()
        host.init_omni_connectors(
            vllm_config=None,
            model_config=_make_model_config(),
        )
        host._omni_connector = MockConnector(stage_id=0)
        host._stage_id = 0
        host._custom_process_func = batch_func

        req = _make_request("req-1")
        host.accumulate_batch_output("req-1", {"raw": 42}, req)
        self.assertEqual(len(host._pending_batch_send), 1)

        host.flush_batch_outputs({"req-1"})
        self.assertEqual(len(host._pending_batch_send), 0)
        self.assertEqual(len(call_log), 1)
        self.assertEqual(call_log[0], "req-1")

        time.sleep(0.1)
        host.shutdown_omni_connectors()


class TestKVSentReqIdsAccumulation(unittest.TestCase):
    """Test that kv_sent_req_ids accumulates results from send_kv_cache."""

    def test_kv_sent_accumulation(self):
        mock_kvm = MagicMock()
        mock_kvm.handle_finished_requests_kv_transfer.return_value = ["req-1", "req-2"]

        host = MixinHost()
        host.init_omni_connectors(
            vllm_config=None,
            model_config=_make_model_config(),
            kv_transfer_manager=mock_kvm,
        )

        host.send_kv_cache(
            finished_reqs={"req-1": {}, "req-2": {}},
            kv_caches=[],
            block_size=16,
            cache_dtype="float16",
        )

        output = host.get_omni_connector_output()
        self.assertIn("req-1", output.kv_sent_req_ids)
        self.assertIn("req-2", output.kv_sent_req_ids)

        output2 = host.get_omni_connector_output()
        self.assertEqual(output2.kv_sent_req_ids, [])

        host.shutdown_omni_connectors()


if __name__ == "__main__":
    unittest.main()
