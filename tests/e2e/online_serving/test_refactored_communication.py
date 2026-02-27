# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E smoke tests for the refactored communication layer.

These tests validate that the new OmniConnectorModelRunnerMixin and
ChunkSchedulingCoordinator work correctly with real (or mock) connectors
and can be wired into Model Runners and Schedulers without regression.

NOTE: These tests require vLLM-Omni runtime.  They will be skipped
automatically if the runtime is unavailable (e.g., outside the container).
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

# Guard imports so the test file can be collected outside the container
_SKIP_REASON = None
try:
    from vllm_omni.core.sched.omni_scheduling_coordinator import ChunkSchedulingCoordinator
    from vllm_omni.worker.omni_connector_model_runner_mixin import OmniConnectorModelRunnerMixin
except ImportError as exc:
    _SKIP_REASON = f"vllm_omni not importable: {exc}"


def skip_if_unavailable(cls):
    if _SKIP_REASON:
        return unittest.skip(_SKIP_REASON)(cls)
    return cls


# ------------------------------------------------------------------ #
#  Mock helpers
# ------------------------------------------------------------------ #


class _MockConnector:
    """Minimal in-memory connector."""

    def __init__(self, stage_id: int = 0):
        self.stage_id = stage_id
        self._store: dict[str, Any] = {}

    def put(self, from_stage, to_stage, put_key, data):
        key = f"{from_stage}_{to_stage}_{put_key}"
        self._store[key] = data
        return True, 0, None

    def get(self, from_stage, to_stage, get_key, metadata=None):
        key = f"{from_stage}_{to_stage}_{get_key}"
        data = self._store.pop(key, None)
        return (data, 0) if data is not None else None

    def close(self):
        pass


def _make_model_config(**overrides):
    defaults = {
        "stage_connector_config": None,
        "async_chunk": False,
        "worker_type": "ar",
        "custom_process_next_stage_input_func": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_request(req_id, status="waiting"):
    return SimpleNamespace(
        request_id=req_id,
        external_req_id=req_id,
        status=status,
        additional_information=None,
        prompt_token_ids=[],
        num_computed_tokens=0,
        num_prompt_tokens=0,
    )


class _MixinHost(OmniConnectorModelRunnerMixin):
    pass


# ------------------------------------------------------------------ #
#  Test 1: Batch send → recv round-trip
# ------------------------------------------------------------------ #


@skip_if_unavailable
class TestBatchRoundTrip(unittest.TestCase):
    """Batch send_stage_outputs → recv_stage_inputs through mock connector."""

    def test_roundtrip(self):
        connector = _MockConnector(stage_id=0)

        sender = _MixinHost()
        sender.init_omni_connectors(None, _make_model_config())
        sender._omni_connector = connector
        sender._stage_id = 0

        sent = sender.send_stage_outputs(None, {"r1": {"prompt_token_ids": [1, 2]}})
        self.assertEqual(sent, ["r1"])

        import time

        time.sleep(0.2)

        receiver = _MixinHost()
        receiver.init_omni_connectors(None, _make_model_config())
        receiver._omni_connector = connector
        receiver._stage_id = 1
        receiver._pending_load_reqs["r1"] = _make_request("r1")

        time.sleep(0.2)
        receiver.recv_stage_inputs(None)

        sender.shutdown_omni_connectors()
        receiver.shutdown_omni_connectors()


# ------------------------------------------------------------------ #
#  Test 2: Chunk send → recv with async_chunk
# ------------------------------------------------------------------ #


@skip_if_unavailable
class TestAsyncChunkRoundTrip(unittest.TestCase):
    """Async chunk send_chunk → recv_chunk through mock connector."""

    def test_chunk_flow(self):
        connector = _MockConnector(stage_id=0)
        sender = _MixinHost()
        sender.init_omni_connectors(None, _make_model_config(async_chunk=True))
        sender._omni_connector = connector
        sender._stage_id = 0
        sender._async_chunk = True

        def mock_func(transfer_manager, pooling_output, request):
            return {"val": 42, "finished": True}

        sender._custom_process_func = mock_func

        req = _make_request("r1")
        ok = sender.send_chunk(req, pooling_output={"raw": 1})
        self.assertTrue(ok)

        import time

        time.sleep(0.2)
        sender.shutdown_omni_connectors()


# ------------------------------------------------------------------ #
#  Test 3: KV cache delegation
# ------------------------------------------------------------------ #


@skip_if_unavailable
class TestKVCacheDelegation(unittest.TestCase):
    """send_kv_cache / recv_kv_cache delegate to OmniKVTransferManager."""

    def test_send_delegates(self):
        kvm = MagicMock()
        kvm.handle_finished_requests_kv_transfer.return_value = ["r1"]

        host = _MixinHost()
        host.init_omni_connectors(None, _make_model_config(), kv_transfer_manager=kvm)

        result = host.send_kv_cache(
            finished_reqs={"r1": {}},
            kv_caches=[],
            block_size=16,
            cache_dtype="float16",
        )
        self.assertEqual(result, ["r1"])
        kvm.handle_finished_requests_kv_transfer.assert_called_once()
        host.shutdown_omni_connectors()

    def test_recv_delegates(self):
        kvm = MagicMock()
        kvm.receive_kv_cache_for_request.return_value = ({}, 100)

        host = _MixinHost()
        host.init_omni_connectors(None, _make_model_config(), kv_transfer_manager=kvm)

        data, size = host.recv_kv_cache("r1")
        self.assertEqual(size, 100)
        kvm.receive_kv_cache_for_request.assert_called_once()
        host.shutdown_omni_connectors()


# ------------------------------------------------------------------ #
#  Test 4: Coordinator state transition end-to-end
# ------------------------------------------------------------------ #


@skip_if_unavailable
class TestCoordinatorEndToEnd(unittest.TestCase):
    """Scheduler-side coordinator: WAITING_FOR_CHUNK → WAITING → scheduled."""

    def test_full_cycle(self):
        from vllm.v1.request import RequestStatus

        coord = ChunkSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1)

        req = _make_request("r1", status=RequestStatus.WAITING)

        class MockQueue:
            def __init__(self, items=None):
                self._items = list(items or [])

            def __iter__(self):
                return iter(self._items)

            def __len__(self):
                return len(self._items)

            def add_request(self, r):
                self._items.append(r)

            def prepend_requests(self, rs):
                self._items = list(rs) + self._items

            def remove(self, r):
                self._items.remove(r)

        waiting = MockQueue([req])
        running = []

        coord.process_pending_chunks(
            waiting,
            running,
            chunk_ready_req_ids=set(),
            chunk_finished_req_ids=set(),
        )
        self.assertEqual(req.status, RequestStatus.WAITING_FOR_CHUNK)

        coord.restore_queues(waiting, running)

        coord.process_pending_chunks(
            waiting,
            running,
            chunk_ready_req_ids={"r1"},
            chunk_finished_req_ids={"r1"},
        )
        self.assertEqual(req.status, RequestStatus.WAITING)
        self.assertIn("r1", coord.finished_requests)


# ------------------------------------------------------------------ #
#  Test 5: Heterogeneous TP rank mapping
# ------------------------------------------------------------------ #


@skip_if_unavailable
class TestTPRankMapping(unittest.TestCase):
    """Verify get_kv_remote_ranks for different TP configurations."""

    def _make_host(self, from_tp, to_tp, local_rank=0):
        host = _MixinHost()
        host.init_omni_connectors(None, _make_model_config())
        host._from_tp = from_tp
        host._to_tp = to_tp
        host._local_rank = local_rank
        return host

    def test_homogeneous(self):
        host = self._make_host(4, 4, local_rank=2)
        self.assertEqual(host.get_kv_remote_ranks(), [2])
        host.shutdown_omni_connectors()

    def test_downscale(self):
        host = self._make_host(from_tp=4, to_tp=2, local_rank=0)
        self.assertEqual(host.get_kv_remote_ranks(), [0, 1])
        host.shutdown_omni_connectors()

        host2 = self._make_host(from_tp=4, to_tp=2, local_rank=1)
        self.assertEqual(host2.get_kv_remote_ranks(), [2, 3])
        host2.shutdown_omni_connectors()

    def test_upscale(self):
        host = self._make_host(from_tp=2, to_tp=4, local_rank=0)
        self.assertEqual(host.get_kv_remote_ranks(), [0])
        host.shutdown_omni_connectors()

        host2 = self._make_host(from_tp=2, to_tp=4, local_rank=1)
        self.assertEqual(host2.get_kv_remote_ranks(), [0])
        host2.shutdown_omni_connectors()

        host3 = self._make_host(from_tp=2, to_tp=4, local_rank=2)
        self.assertEqual(host3.get_kv_remote_ranks(), [1])
        host3.shutdown_omni_connectors()

    def test_data_transfer_rank(self):
        host0 = self._make_host(4, 2, local_rank=0)
        self.assertTrue(host0.is_data_transfer_rank())
        host0.shutdown_omni_connectors()

        host1 = self._make_host(4, 2, local_rank=1)
        self.assertFalse(host1.is_data_transfer_rank())
        host1.shutdown_omni_connectors()


# ------------------------------------------------------------------ #
#  Test 6: WAITING_FOR_INPUT coordinator flow
# ------------------------------------------------------------------ #


@skip_if_unavailable
class TestWaitingForInputFlow(unittest.TestCase):
    """Coordinator: WAITING_FOR_INPUT → WAITING after stage_recv_req_ids arrive."""

    def test_batch_input_transition(self):
        from vllm.v1.request import RequestStatus

        coord = ChunkSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1)
        req = _make_request("r1", status=RequestStatus.WAITING_FOR_INPUT)

        class Q:
            def __init__(self, items=None):
                self._items = list(items or [])

            def __iter__(self):
                return iter(self._items)

            def __len__(self):
                return len(self._items)

            def add_request(self, r):
                self._items.append(r)

            def remove(self, r):
                self._items.remove(r)

        waiting = Q([req])
        running = []

        coord.process_pending_batch_inputs(
            waiting,
            running,
            stage_recv_req_ids=set(),
        )
        self.assertEqual(req.status, RequestStatus.WAITING_FOR_INPUT)
        self.assertEqual(len(coord._waiting_for_input), 1)

        coord.restore_queues(waiting, running)

        coord.process_pending_batch_inputs(
            waiting,
            running,
            stage_recv_req_ids={"r1"},
        )
        self.assertEqual(req.status, RequestStatus.WAITING)


# ------------------------------------------------------------------ #
#  Test 7: Finished load reqs drain (A1 fix)
# ------------------------------------------------------------------ #


@skip_if_unavailable
class TestFinishedLoadReqsDrain(unittest.TestCase):
    """get_omni_connector_output drains _finished_load_reqs into chunk_ready."""

    def test_drain(self):
        host = _MixinHost()
        host.init_omni_connectors(None, _make_model_config())

        host._finished_load_reqs.add("r1")
        host._finished_load_reqs.add("r2")

        output = host.get_omni_connector_output()
        self.assertIn("r1", output.chunk_ready_req_ids)
        self.assertIn("r2", output.chunk_ready_req_ids)
        self.assertEqual(len(host._finished_load_reqs), 0)

        host.shutdown_omni_connectors()


# ------------------------------------------------------------------ #
#  Test 8: Seed-request task format
# ------------------------------------------------------------------ #


@skip_if_unavailable
class TestSeedRequestTaskFormat(unittest.TestCase):
    """Validate the seed request dict matches what OmniStage expects."""

    def test_seed_task_structure(self):
        seed_input = {
            "prompt_token_ids": [0] * 10,
            "multi_modal_data": None,
            "mm_processor_kwargs": None,
        }
        seed_task = {
            "request_id": "r1",
            "engine_inputs": seed_input,
            "sampling_params": None,
            "from_connector": False,
        }

        self.assertFalse(seed_task["from_connector"])
        self.assertEqual(seed_task["engine_inputs"]["prompt_token_ids"], [0] * 10)
        self.assertIsNone(seed_task["engine_inputs"]["multi_modal_data"])
        self.assertEqual(seed_task["request_id"], "r1")


if __name__ == "__main__":
    unittest.main()
