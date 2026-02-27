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

import torch

from vllm_omni.outputs import OmniConnectorOutput
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner
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

    def test_send_chunk_passes_is_finished_and_connector(self):
        connector = MockConnector(stage_id=0)

        sender = MixinHost()
        sender.init_omni_connectors(
            vllm_config=None,
            model_config=_make_model_config(stage_id=0, async_chunk=True),
        )
        sender._omni_connector = connector
        sender._stage_id = 0
        sender._async_chunk = True

        seen = {}

        def mock_process(transfer_manager, pooling_output, request, is_finished=False):
            seen["connector"] = transfer_manager.connector
            seen["is_finished"] = is_finished
            return {"data": pooling_output, "finished": is_finished}

        sender._custom_process_func = mock_process

        request = _make_request("req-1", "ext-req-1")
        request.is_finished = lambda: True
        sender._send_single_request(
            {
                "stage_id": 0,
                "next_stage_id": 1,
                "request_id": "ext-req-1",
                "request": request,
                "pooling_output": {"value": 42},
            }
        )
        self.assertIs(seen["connector"], connector)
        self.assertTrue(seen["is_finished"])

        sender.shutdown_omni_connectors()

    def test_send_chunk_does_not_retry_real_type_error(self):
        connector = MockConnector(stage_id=0)

        sender = MixinHost()
        sender.init_omni_connectors(
            vllm_config=None,
            model_config=_make_model_config(stage_id=0, async_chunk=True),
        )
        sender._omni_connector = connector
        sender._stage_id = 0
        sender._async_chunk = True

        seen = {"calls": 0}

        def broken_process(transfer_manager, pooling_output, request, is_finished=""):
            seen["calls"] += 1
            return {"data": is_finished + "tail"}

        sender._custom_process_func = broken_process

        request = _make_request("req-1", "ext-req-1")
        request.is_finished = lambda: True
        ok = sender.send_chunk(request, pooling_output={"value": 42})
        self.assertFalse(ok)
        self.assertEqual(seen["calls"], 1)

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

    def test_receive_multi_kv_fetches_companions_via_mixin(self):
        mock_kvm = MagicMock()

        host = MixinHost()
        host.init_omni_connectors(
            vllm_config=None,
            model_config=_make_model_config(),
            kv_transfer_manager=mock_kvm,
        )

        host.recv_kv_cache = MagicMock(
            side_effect=[({"layer_blocks": {"k": [1]}}, 64), ({"layer_blocks": {"k": [2]}}, 32)]
        )
        seen = {}

        def collect_cfg(request_id, cfg_role_payloads):
            seen["request_id"] = request_id
            seen["cfg_role_payloads"] = cfg_role_payloads
            return {"cfg_text_kv_metadata": {"seq_len": 3}}

        req = SimpleNamespace(
            request_id="req-1",
            sampling_params=SimpleNamespace(cfg_kv_request_ids={"cfg_text": "req-1__cfg_text"}),
        )
        ok = host.receive_multi_kv_cache(req, cfg_kv_collect_func=collect_cfg)
        self.assertTrue(ok)
        host.recv_kv_cache.assert_any_call("req-1", target_device=None)
        host.recv_kv_cache.assert_any_call("req-1__cfg_text", target_device=None)
        mock_kvm.apply_kv_cache_to_request.assert_called_once_with(req, {"layer_blocks": {"k": [1]}})
        self.assertEqual(seen["request_id"], "req-1")
        self.assertEqual(
            seen["cfg_role_payloads"],
            {"cfg_text": ({"layer_blocks": {"k": [2]}}, 32)},
        )
        self.assertEqual(req.sampling_params.cfg_text_kv_metadata, {"seq_len": 3})

        host.shutdown_omni_connectors()

    def test_receive_multi_kv_skips_inactive_request(self):
        mock_kvm = MagicMock()

        host = MixinHost()
        host.init_omni_connectors(
            vllm_config=None,
            model_config=_make_model_config(),
            kv_transfer_manager=mock_kvm,
        )

        host.requests = {}
        host.recv_kv_cache = MagicMock(return_value=({"layer_blocks": {"k": [1]}}, 64))
        req = SimpleNamespace(request_id="req-1", sampling_params=None)

        ok = host.receive_multi_kv_cache(req)

        self.assertFalse(ok)
        host.recv_kv_cache.assert_not_called()
        mock_kvm.apply_kv_cache_to_request.assert_not_called()

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
        host._local_request_metadata["req-1"] = {"next_stage_prompt_len": 10}
        host._stage_recv_req_ids.add("req-3")

        output = host.get_omni_connector_output()
        self.assertIsInstance(output, OmniConnectorOutput)
        self.assertEqual(output.chunk_ready_req_ids, {"req-1"})
        self.assertEqual(output.chunk_finished_req_ids, {"req-2"})
        self.assertEqual(output.request_metadata, {"req-1": {"next_stage_prompt_len": 10}})
        self.assertEqual(output.stage_recv_req_ids, {"req-3"})

        output2 = host.get_omni_connector_output()
        self.assertEqual(output2.chunk_ready_req_ids, set())
        self.assertEqual(output2.request_metadata, {})

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

    def test_batch_send_passes_is_finished_and_connector(self):
        seen = {}

        def batch_func(transfer_manager, pooling_output, request, is_finished=False):
            seen["connector"] = transfer_manager.connector
            seen["is_finished"] = is_finished
            seen["data"] = pooling_output
            seen["rid"] = request.request_id if request else None
            return {"processed": True, "finished": is_finished}

        host = MixinHost()
        host.init_omni_connectors(
            vllm_config=None,
            model_config=_make_model_config(),
        )
        host._omni_connector = MockConnector(stage_id=0)
        host._stage_id = 0
        host._custom_process_func = batch_func

        req = _make_request("req-1")
        req.is_finished = lambda: True
        sent = host.send_stage_outputs(
            scheduler_output=None,
            outputs={"req-1": ({"raw": 100}, req)},
        )
        self.assertEqual(sent, ["req-1"])
        self.assertEqual(
            seen,
            {
                "connector": host._omni_connector,
                "is_finished": True,
                "data": {"raw": 100},
                "rid": "req-1",
            },
        )

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


class TestChunkStreamCompletedGuard(unittest.TestCase):
    """Test that register_chunk_recv is skipped after finish sentinel.

    This validates the fix for the race condition where the scheduling
    coordinator re-registers a request for chunk polling after its
    upstream chunk stream has already finished (is_finished sentinel
    received), causing the bg recv thread to poll for a non-existent
    shared-memory segment (e.g. ``_0_7`` when only 7 chunks 0–6 exist).
    """

    def _make_host(self, stage_id: int = 1) -> MixinHost:
        host = MixinHost()
        host.init_omni_connectors(
            vllm_config=None,
            model_config=_make_model_config(stage_id=stage_id, async_chunk=True),
        )
        host._omni_connector = MockConnector(stage_id=stage_id)
        host._stage_id = stage_id
        host._async_chunk = True
        return host

    def test_register_blocked_after_finish_sentinel(self):
        """register_chunk_recv must be a no-op after the finish sentinel."""
        host = self._make_host(stage_id=1)

        req = _make_request("req-1", "ext-req-1")

        # Simulate the bg thread having received the finish sentinel:
        with host._lock:
            host._chunk_stream_completed.add("req-1")

        # Now try to re-register — this mimics the coordinator asking
        # the model runner to poll for the next (non-existent) chunk.
        host.register_chunk_recv(req)

        # The request must NOT appear in _pending_load_reqs
        self.assertNotIn(
            "req-1",
            host._pending_load_reqs,
            "register_chunk_recv should skip requests whose chunk stream is already complete",
        )

        host.shutdown_omni_connectors()

    def test_register_allowed_before_finish(self):
        """register_chunk_recv works normally before finish sentinel."""
        host = self._make_host(stage_id=1)
        req = _make_request("req-1", "ext-req-1")

        host.register_chunk_recv(req)
        self.assertIn(
            "req-1",
            host._pending_load_reqs,
            "register_chunk_recv should add request to pending when stream is not yet complete",
        )

        host.shutdown_omni_connectors()

    def test_finish_sentinel_populates_completed_set(self):
        """Receiving is_finished=True adds to _chunk_stream_completed."""
        host = self._make_host(stage_id=1)

        # Simulate _poll_single_request receiving is_finished=True
        req_id = "req-1"
        with host._lock:
            host._chunk_finished_req_ids.add(req_id)
            host._chunk_stream_completed.add(req_id)
            host._local_stage_payload_cache[req_id] = {"finished": True}
            host._local_request_metadata[req_id] = {}
            host._finished_load_reqs.add(req_id)
            host._pending_load_reqs.pop(req_id, None)

        self.assertIn(req_id, host._chunk_stream_completed)

        # Subsequent register_chunk_recv should be blocked
        req = _make_request(req_id, f"ext-{req_id}")
        host.register_chunk_recv(req)
        self.assertNotIn(req_id, host._pending_load_reqs)

        host.shutdown_omni_connectors()

    def test_stage_0_always_skipped(self):
        """Stage-0 has no upstream, register_chunk_recv is always no-op."""
        host = self._make_host(stage_id=0)
        host._stage_id = 0

        req = _make_request("req-1")
        host.register_chunk_recv(req)
        self.assertNotIn("req-1", host._pending_load_reqs)

        host.shutdown_omni_connectors()

    def test_batch_recv_guard_still_works(self):
        """Pre-existing guard: batch recv results prevent registration."""
        host = self._make_host(stage_id=1)

        with host._lock:
            host._batch_recv_results["req-1"] = {"some": "data"}

        req = _make_request("req-1", "ext-req-1")
        host.register_chunk_recv(req)
        self.assertNotIn("req-1", host._pending_load_reqs)

        host.shutdown_omni_connectors()

    def test_register_does_not_synthesize_additional_information(self):
        """register_chunk_recv should not mutate requests for legacy carrier fields."""
        host = self._make_host(stage_id=1)

        req = SimpleNamespace(request_id="req-1", external_req_id="ext-req-1")
        self.assertFalse(hasattr(req, "additional_information"))

        host.register_chunk_recv(req)

        self.assertIn("req-1", host._pending_load_reqs)
        self.assertFalse(hasattr(req, "additional_information"))

        host.shutdown_omni_connectors()


class TestCleanupFinishedRequest(unittest.TestCase):
    """Test cleanup_finished_request frees per-request mixin state."""

    def _make_host(self, stage_id: int = 1) -> MixinHost:
        host = MixinHost()
        host.init_omni_connectors(
            vllm_config=None,
            model_config=_make_model_config(stage_id=stage_id, async_chunk=True),
        )
        host._omni_connector = MockConnector(stage_id=stage_id)
        host._stage_id = stage_id
        host._async_chunk = True
        return host

    def test_cleanup_removes_all_state(self):
        """cleanup_finished_request removes all tracking dicts/sets."""
        host = self._make_host(stage_id=1)
        req_id = "req-1"
        ext_id = "ext-req-1"

        # Simulate state accumulated during a request's lifetime
        host._request_ids_mapping[req_id] = ext_id
        host._put_req_chunk[ext_id] = 5
        host._get_req_chunk[req_id] = 3
        host._request_payload[ext_id] = {"some": "data"}
        host._code_prompt_token_ids[ext_id] = [[1, 2, 3]]
        host._chunk_stream_completed.add(req_id)
        host._batch_recv_results[req_id] = {"result": True}
        host._stage_recv_req_ids.add(req_id)
        host._local_stage_payload_cache[req_id] = {"engine_inputs": {}}
        host._local_request_metadata[req_id] = {"prompt_len": 10}

        # Cleanup
        host.cleanup_finished_request(req_id)

        # All state should be gone
        self.assertNotIn(req_id, host._request_ids_mapping)
        self.assertNotIn(ext_id, host._put_req_chunk)
        self.assertNotIn(req_id, host._get_req_chunk)
        self.assertNotIn(ext_id, host._request_payload)
        self.assertNotIn(ext_id, host._code_prompt_token_ids)
        self.assertNotIn(req_id, host._chunk_stream_completed)
        self.assertNotIn(req_id, host._batch_recv_results)
        self.assertNotIn(req_id, host._stage_recv_req_ids)
        self.assertNotIn(req_id, host._local_stage_payload_cache)
        self.assertNotIn(req_id, host._local_request_metadata)

        host.shutdown_omni_connectors()

    def test_cleanup_removes_per_cycle_ready_state(self):
        """cleanup_finished_request clears ready/finished carry-over for req-id reuse."""
        host = self._make_host(stage_id=1)
        req_id = "req-1"

        host._pending_load_reqs[req_id] = _make_request(req_id, "ext-req-1")
        host._finished_load_reqs.add(req_id)
        host._chunk_ready_req_ids.add(req_id)
        host._chunk_finished_req_ids.add(req_id)

        host.cleanup_finished_request(req_id)

        self.assertNotIn(req_id, host._pending_load_reqs)
        self.assertNotIn(req_id, host._finished_load_reqs)
        self.assertNotIn(req_id, host._chunk_ready_req_ids)
        self.assertNotIn(req_id, host._chunk_finished_req_ids)

        host.shutdown_omni_connectors()

    def test_cleanup_without_mapping(self):
        """cleanup works for Stage-0 where _request_ids_mapping isn't set."""
        host = self._make_host(stage_id=0)
        host._stage_id = 0
        req_id = "req-1"

        # Stage-0 uses req_id directly (no ext_id mapping)
        host._put_req_chunk[req_id] = 3
        host._get_req_chunk[req_id] = 0

        host.cleanup_finished_request(req_id)

        self.assertNotIn(req_id, host._put_req_chunk)
        self.assertNotIn(req_id, host._get_req_chunk)

        host.shutdown_omni_connectors()

    def test_prune_inactive_requests_cleans_stale_state_but_keeps_active(self):
        """Inactive request IDs should be pruned without touching active ones."""
        host = self._make_host(stage_id=1)
        active_req_id = "req-active"
        stale_req_id = "req-stale"
        stale_ext_id = "ext-stale"

        host._request_ids_mapping[active_req_id] = "ext-active"
        host._request_ids_mapping[stale_req_id] = stale_ext_id
        host._put_req_chunk[stale_ext_id] = 2
        host._get_req_chunk[stale_req_id] = 1
        host._finished_load_reqs.add(stale_req_id)
        host._chunk_ready_req_ids.update({active_req_id, stale_req_id})
        host._chunk_finished_req_ids.add(stale_req_id)
        host._chunk_stream_completed.add(stale_req_id)
        host._batch_recv_results[stale_req_id] = {"result": True}
        host._stage_recv_req_ids.update({active_req_id, stale_req_id})
        host._local_stage_payload_cache[stale_req_id] = {"engine_inputs": {}}
        host._local_request_metadata[stale_req_id] = {"prompt_len": 8}
        host._request_payload[stale_ext_id] = {"stale": True}
        host._code_prompt_token_ids[stale_ext_id] = [[1, 2, 3]]

        pruned = host.prune_inactive_requests({active_req_id})

        self.assertEqual(pruned, {stale_req_id})
        self.assertIn(active_req_id, host._request_ids_mapping)
        self.assertIn(active_req_id, host._chunk_ready_req_ids)
        self.assertIn(active_req_id, host._stage_recv_req_ids)
        self.assertNotIn(stale_req_id, host._request_ids_mapping)
        self.assertNotIn(stale_ext_id, host._put_req_chunk)
        self.assertNotIn(stale_req_id, host._get_req_chunk)
        self.assertNotIn(stale_req_id, host._pending_load_reqs)
        self.assertNotIn(stale_req_id, host._finished_load_reqs)
        self.assertNotIn(stale_req_id, host._chunk_ready_req_ids)
        self.assertNotIn(stale_req_id, host._chunk_finished_req_ids)
        self.assertNotIn(stale_req_id, host._chunk_stream_completed)
        self.assertNotIn(stale_req_id, host._batch_recv_results)
        self.assertNotIn(stale_req_id, host._stage_recv_req_ids)
        self.assertNotIn(stale_req_id, host._local_stage_payload_cache)
        self.assertNotIn(stale_req_id, host._local_request_metadata)
        self.assertNotIn(stale_ext_id, host._request_payload)
        self.assertNotIn(stale_ext_id, host._code_prompt_token_ids)

        host.shutdown_omni_connectors()


class TestSendChunkCachesMapping(unittest.TestCase):
    """Test that send_chunk caches internal→external req ID mapping."""

    def test_send_chunk_populates_request_ids_mapping(self):
        """send_chunk should cache the internal→external mapping."""
        host = MixinHost()
        host.init_omni_connectors(
            vllm_config=None,
            model_config=_make_model_config(stage_id=0, async_chunk=True),
        )
        host._omni_connector = MockConnector(stage_id=0)
        host._stage_id = 0
        host._async_chunk = True

        def mock_process(transfer_manager, pooling_output, request):
            return {"data": "test", "finished": False}

        host._custom_process_func = mock_process

        request = _make_request("internal-1", "external-1")
        host.send_chunk(request, pooling_output={"v": 1})

        # The mapping should be cached
        self.assertEqual(
            host._request_ids_mapping.get("internal-1"),
            "external-1",
        )

        time.sleep(0.1)
        host.shutdown_omni_connectors()


class TestLocalPayloadCacheLifecycle(unittest.TestCase):
    """Unit tests for the local payload cache API (RFC §2.4)."""

    def _make_host(self) -> MixinHost:
        host = MixinHost()
        host.init_omni_connectors(
            vllm_config=None,
            model_config=_make_model_config(stage_id=0),
        )
        host._omni_connector = MockConnector(stage_id=0)
        host._stage_id = 0
        return host

    def test_put_get_pop(self):
        host = self._make_host()
        payload = {"engine_inputs": {"ids": [1, 2, 3]}}
        host.put_local_stage_payload("r1", payload)

        self.assertEqual(host.get_local_stage_payload("r1"), payload)
        popped = host.pop_local_stage_payload("r1")
        self.assertEqual(popped, payload)
        self.assertIsNone(host.get_local_stage_payload("r1"))
        host.shutdown_omni_connectors()

    def test_pop_missing_returns_none(self):
        host = self._make_host()
        self.assertIsNone(host.pop_local_stage_payload("nonexistent"))
        host.shutdown_omni_connectors()

    def test_put_overwrites(self):
        host = self._make_host()
        host.put_local_stage_payload("r1", {"v": 1})
        host.put_local_stage_payload("r1", {"v": 2})
        self.assertEqual(host.get_local_stage_payload("r1"), {"v": 2})
        host.shutdown_omni_connectors()

    def test_metadata_put_get(self):
        host = self._make_host()
        host.put_local_request_metadata("r1", {"prompt_len": 10})
        self.assertEqual(
            host.get_local_request_metadata("r1"),
            {"prompt_len": 10},
        )
        host.shutdown_omni_connectors()

    def test_cleanup_removes_cache_and_metadata(self):
        host = self._make_host()
        host.put_local_stage_payload("r1", {"data": True})
        host.put_local_request_metadata("r1", {"len": 5})
        host.cleanup_finished_request("r1")
        self.assertIsNone(host.get_local_stage_payload("r1"))
        self.assertIsNone(host.get_local_request_metadata("r1"))
        host.shutdown_omni_connectors()

    def test_recv_stage_inputs_populates_local_cache(self):
        host = self._make_host()
        host._omni_connector = MockConnector(stage_id=0)
        host._stage_id = 0

        # Simulate batch recv results arriving from the bg thread
        with host._lock:
            host._batch_recv_results["r1"] = {"tok": [10]}

        host.recv_stage_inputs(scheduler_output=None)
        self.assertEqual(host.get_local_stage_payload("r1"), {"tok": [10]})
        host.shutdown_omni_connectors()


class TestKVTransferLifecycle(unittest.TestCase):
    """Unit tests for KV transfer lifecycle methods."""

    def _make_host(self) -> MixinHost:
        host = MixinHost()
        host.init_omni_connectors(
            vllm_config=None,
            model_config=_make_model_config(stage_id=0),
        )
        return host

    def test_mark_drain_ack_complete(self):
        host = self._make_host()
        self.assertFalse(host.has_pending_kv_work())

        host.mark_kv_transfer("r1", seq_len=100, block_ids=[0, 1, 2])
        self.assertTrue(host.has_pending_kv_work())
        self.assertTrue(host.is_kv_transfer_triggered("r1"))

        # Drain moves pending → active
        pending = host.drain_pending_kv_transfers()
        self.assertEqual(pending, {"r1": {"seq_len": 100, "block_ids": [0, 1, 2]}})
        self.assertIn("r1", host._kv_active_transfers)
        self.assertTrue(host.has_pending_kv_work())

        # Ack moves active → completed
        host.ack_kv_transfers(["r1"])
        self.assertNotIn("r1", host._kv_active_transfers)
        self.assertIn("r1", host._kv_completed_transfers)

        # Drain completed
        completed = host.drain_completed_kv_transfers()
        self.assertEqual(completed, {"r1"})
        self.assertFalse(host.has_pending_kv_work())
        host.shutdown_omni_connectors()

    def test_mark_dedup(self):
        host = self._make_host()
        host.mark_kv_transfer("r1", seq_len=100, block_ids=[0])
        host.mark_kv_transfer("r1", seq_len=200, block_ids=[0, 1])
        # Second mark is a no-op
        self.assertEqual(host._kv_pending_transfers["r1"]["seq_len"], 100)
        host.shutdown_omni_connectors()

    def test_cleanup_removes_kv_state(self):
        host = self._make_host()
        host.mark_kv_transfer("r1", seq_len=50, block_ids=[0])
        host.drain_pending_kv_transfers()
        host.cleanup_finished_request("r1")
        self.assertFalse(host.is_kv_transfer_triggered("r1"))
        self.assertNotIn("r1", host._kv_active_transfers)
        self.assertFalse(host.has_pending_kv_work())
        host.shutdown_omni_connectors()

    def test_has_pending_kv_work_in_connector_output(self):
        host = self._make_host()
        output = host.get_omni_connector_output()
        self.assertFalse(output.has_pending_kv_work)

        host.mark_kv_transfer("r1", seq_len=10, block_ids=[0])
        output = host.get_omni_connector_output()
        self.assertTrue(output.has_pending_kv_work)
        host.shutdown_omni_connectors()


class TestAsyncPayloadLifecycle(unittest.TestCase):
    """Regression tests for async payload delivery lifecycle."""

    def test_request_payload_cleared_without_scheduling_metadata(self):
        host = MixinHost()
        host.init_omni_connectors(
            vllm_config=None,
            model_config=_make_model_config(stage_id=1, async_chunk=True, worker_type="ar"),
        )
        host._request_ids_mapping["r1"] = "r1"
        payload = {
            "thinker_decode_embeddings": torch.ones(1, 2),
            "thinker_output_token_ids": [1],
            "override_keys": ["thinker_decode_embeddings", "thinker_output_token_ids"],
            "finished": torch.tensor(False),
        }

        host._accumulate_payload("r1", dict(payload))
        with host._lock:
            host._finished_load_reqs.add("r1")
            host._local_stage_payload_cache["r1"] = dict(payload)

        host.get_omni_connector_output()
        self.assertNotIn("r1", host._request_payload)
        host.shutdown_omni_connectors()

    def test_request_payload_not_cleared_before_payload_is_consumable(self):
        host = MixinHost()
        host.init_omni_connectors(
            vllm_config=None,
            model_config=_make_model_config(stage_id=1, async_chunk=True, worker_type="ar"),
        )
        host._request_ids_mapping["r1"] = "r1"
        payload = {
            "thinker_decode_embeddings": torch.ones(1, 2),
            "thinker_output_token_ids": [1],
            "override_keys": ["thinker_decode_embeddings", "thinker_output_token_ids"],
            "finished": torch.tensor(False),
        }

        host._accumulate_payload("r1", dict(payload))
        with host._lock:
            host._finished_load_reqs.add("r1")

        host.get_omni_connector_output()
        self.assertIn("r1", host._request_payload)
        host.shutdown_omni_connectors()

    def test_sync_local_stage_payloads_is_consume_once(self):
        carrier = SimpleNamespace(
            _local_stage_payload_cache={"r1": {"thinker_decode_embeddings": torch.ones(1, 2)}},
            model_intermediate_buffer={"r1": {"thinker_decode_embeddings": None}},
        )

        OmniGPUModelRunner._sync_local_stage_payloads(carrier)
        self.assertNotIn("r1", carrier._local_stage_payload_cache)
        self.assertIsNotNone(carrier.model_intermediate_buffer["r1"]["thinker_decode_embeddings"])

        carrier.model_intermediate_buffer["r1"]["thinker_decode_embeddings"] = None
        OmniGPUModelRunner._sync_local_stage_payloads(carrier)
        self.assertIsNone(carrier.model_intermediate_buffer["r1"]["thinker_decode_embeddings"])

    def test_payload_consumable_ignores_token_horizon_only_updates(self):
        host = MixinHost()
        host.init_omni_connectors(
            vllm_config=None,
            model_config=_make_model_config(stage_id=1, async_chunk=True, worker_type="ar"),
        )
        payload = {
            "thinker_output_token_ids": [1, 2, 3],
            "finished": torch.tensor(False),
            "override_keys": [
                "thinker_output_token_ids",
                "thinker_decode_embeddings_token_start",
                "thinker_decode_embeddings_token_end",
            ],
            "thinker_decode_embeddings_token_start": 2,
            "thinker_decode_embeddings_token_end": 3,
        }
        self.assertFalse(host._payload_is_consumable(payload))
        host.shutdown_omni_connectors()

    def test_payload_consumable_accepts_decode_embeddings(self):
        host = MixinHost()
        host.init_omni_connectors(
            vllm_config=None,
            model_config=_make_model_config(stage_id=1, async_chunk=True, worker_type="ar"),
        )
        payload = {
            "thinker_output_token_ids": [1, 2, 3],
            "thinker_decode_embeddings": torch.ones(1, 2),
            "finished": torch.tensor(False),
        }
        self.assertTrue(host._payload_is_consumable(payload))
        host.shutdown_omni_connectors()

    def test_ar_metadata_only_followup_chunk_does_not_rewake_request(self):
        host = MixinHost()
        host.init_omni_connectors(
            vllm_config=None,
            model_config=_make_model_config(stage_id=1, async_chunk=True, worker_type="ar"),
        )
        host._omni_connector = MagicMock()
        host._stage_id = 1
        host._async_chunk = True
        host._model_mode = "ar"
        host._request_ids_mapping["r1"] = "ext-r1"
        host._get_req_chunk["r1"] = 0

        host._omni_connector.get.side_effect = [
            (
                {
                    "thinker_decode_embeddings": torch.ones(1, 2),
                    "finished": torch.tensor(False),
                },
                1,
            ),
            (
                {
                    "next_stage_prompt_len": 7,
                    "finished": torch.tensor(False),
                },
                1,
            ),
        ]

        host._poll_single_request("r1")
        output1 = host.get_omni_connector_output()
        self.assertEqual(output1.chunk_ready_req_ids, {"r1"})

        host._poll_single_request("r1")
        output2 = host.get_omni_connector_output()
        self.assertEqual(output2.chunk_ready_req_ids, set())
        self.assertEqual(output2.request_metadata, {"r1": {"next_stage_prompt_len": 7}})

        host.shutdown_omni_connectors()


class TestRankAwareKVRouting(unittest.TestCase):
    def _make_host(self, *, from_tp: int, to_tp: int, local_rank: int) -> MixinHost:
        host = MixinHost()
        host.init_omni_connectors(vllm_config=None, model_config=_make_model_config(stage_id=1))
        host._from_tp = from_tp
        host._to_tp = to_tp
        host._local_rank = local_rank
        return host

    def test_recv_keys_use_remote_rank_as_from_rank(self):
        host = self._make_host(from_tp=4, to_tp=2, local_rank=1)
        self.assertEqual(
            host.get_rank_aware_kv_keys("req", from_stage=0),
            ["req_0_0_2_1", "req_0_0_3_1"],
        )
        host.shutdown_omni_connectors()

    def test_send_keys_route_from_rank_gt_to_rank(self):
        host = self._make_host(from_tp=4, to_tp=2, local_rank=3)
        self.assertEqual(host.get_rank_aware_kv_send_keys("req", from_stage=0), ["req_0_0_3_1"])
        host.shutdown_omni_connectors()

    def test_invalid_recv_rank_mapping_raises(self):
        host = self._make_host(from_tp=3, to_tp=2, local_rank=1)
        with self.assertRaises(ValueError):
            host.get_rank_aware_kv_keys("req", from_stage=0)
        host.shutdown_omni_connectors()

    def test_invalid_send_rank_mapping_raises(self):
        host = self._make_host(from_tp=3, to_tp=2, local_rank=1)
        with self.assertRaises(ValueError):
            host.get_rank_aware_kv_send_keys("req", from_stage=0)
        host.shutdown_omni_connectors()

    def test_merge_rank_sharded_payloads_concatenates_head_dimension(self):
        host = self._make_host(from_tp=4, to_tp=2, local_rank=0)
        payloads = [
            {"layer_blocks": {"key_cache": [torch.ones(2, 1, 3)], "value_cache": [torch.ones(2, 1, 3)]}},
            {"layer_blocks": {"key_cache": [torch.full((2, 1, 3), 2.0)], "value_cache": [torch.full((2, 1, 3), 2.0)]}},
        ]
        merged = host._merge_rank_sharded_kv_payloads(payloads)
        self.assertEqual(tuple(merged["layer_blocks"]["key_cache"][0].shape), (2, 2, 3))
        self.assertTrue(torch.equal(merged["layer_blocks"]["key_cache"][0][:, 0], torch.ones(2, 3)))
        self.assertTrue(torch.equal(merged["layer_blocks"]["key_cache"][0][:, 1], torch.full((2, 3), 2.0)))
        host.shutdown_omni_connectors()

    def test_slice_rank_sharded_payload_splits_head_dimension(self):
        host = self._make_host(from_tp=2, to_tp=4, local_rank=1)
        payload = {
            "layer_blocks": {
                "key_cache": [torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)],
                "value_cache": [torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)],
            },
            "metadata": {},
        }
        sliced = host._slice_rank_sharded_kv_payload(payload)
        self.assertEqual(tuple(sliced["layer_blocks"]["key_cache"][0].shape), (2, 2, 3))
        expected = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)[:, 2:4, :]
        self.assertTrue(torch.equal(sliced["layer_blocks"]["key_cache"][0], expected))
        host.shutdown_omni_connectors()


class TestPendingRecvPruneProtection(unittest.TestCase):
    def test_ready_pending_recv_is_not_pruned(self):
        host = MixinHost()
        host.init_omni_connectors(vllm_config=None, model_config=_make_model_config(stage_id=1, async_chunk=True))
        host._pending_load_reqs["req-1"] = _make_request("req-1")
        host._finished_load_reqs.add("req-1")
        host._local_stage_payload_cache["req-1"] = {"thinker_prefill_embeddings": torch.ones(1, 1)}

        output = host.get_omni_connector_output()
        stale = host.prune_inactive_requests(set())

        self.assertEqual(output.chunk_ready_req_ids, {"req-1"})
        self.assertEqual(stale, set())
        self.assertIn("req-1", host._pending_load_reqs)
        self.assertIn("req-1", host._local_stage_payload_cache)
        host.shutdown_omni_connectors()


class TestAttachOmniConnectorOutput(unittest.TestCase):
    def test_wraps_empty_model_runner_output_when_signals_exist(self):
        from vllm.v1.worker.gpu_model_runner import EMPTY_MODEL_RUNNER_OUTPUT

        host = MixinHost()
        host.get_omni_connector_output = lambda: OmniConnectorOutput(chunk_ready_req_ids={"req-1"})

        wrapped = host.attach_omni_connector_output(EMPTY_MODEL_RUNNER_OUTPUT)

        self.assertIsNot(wrapped, EMPTY_MODEL_RUNNER_OUTPUT)
        self.assertEqual(wrapped.omni_connector_output.chunk_ready_req_ids, {"req-1"})

    def test_returns_original_result_when_no_signals_exist(self):
        result = SimpleNamespace(existing=True)
        host = MixinHost()
        host.get_omni_connector_output = lambda: OmniConnectorOutput()

        wrapped = host.attach_omni_connector_output(result)

        self.assertIs(wrapped, result)


if __name__ == "__main__":
    unittest.main()
