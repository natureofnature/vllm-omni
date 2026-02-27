# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for OmniSchedulingCoordinator (formerly ChunkSchedulingCoordinator).

These tests use mock request objects and mock queues.  They do not require
GPU, vLLM runtime, or any connector.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from vllm_omni.core.sched.omni_scheduling_coordinator import (
    ChunkSchedulingCoordinator,
    OmniSchedulingCoordinator,
)

# ------------------------------------------------------------------ #
#  Mock helpers
# ------------------------------------------------------------------ #


class _RequestStatus:
    WAITING = "waiting"
    RUNNING = "running"
    WAITING_FOR_CHUNK = "waiting_for_chunk"
    WAITING_FOR_INPUT = "waiting_for_input"
    FINISHED_STOPPED = "finished_stopped"


# Patch RequestStatus for tests that don't import vllm
try:
    from vllm.v1.request import RequestStatus
except ImportError:
    RequestStatus = _RequestStatus  # type: ignore[misc,assignment]


def _make_request(req_id: str, status: str = "waiting") -> SimpleNamespace:
    return SimpleNamespace(
        request_id=req_id,
        external_req_id=req_id,
        status=status,
        additional_information=None,
        prompt_token_ids=[],
        num_computed_tokens=0,
    )


class MockQueue:
    """Simplified queue that mimics the Scheduler waiting queue interface."""

    def __init__(self, items: list | None = None):
        self._items: list = list(items or [])

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)

    def __contains__(self, item):
        return item in self._items

    def add_request(self, request):
        self._items.append(request)

    def prepend_requests(self, requests):
        self._items = list(requests) + self._items

    def remove(self, request):
        self._items.remove(request)


# ------------------------------------------------------------------ #
#  Tests
# ------------------------------------------------------------------ #


class TestChunkCoordinatorStateTransition(unittest.TestCase):
    """Test 5: process_pending_chunks transitions WAITING_FOR_CHUNK → target."""

    def test_ready_request_transitions_to_waiting(self):
        coord = ChunkSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1)

        req = _make_request("r1", status=RequestStatus.WAITING_FOR_CHUNK)
        waiting = MockQueue([req])
        running: list = []

        coord.process_pending_chunks(
            waiting,
            running,
            chunk_ready_req_ids={"r1"},
            chunk_finished_req_ids=set(),
        )

        self.assertEqual(req.status, RequestStatus.WAITING)
        self.assertIn("r1", coord.requests_with_ready_chunks)

    def test_non_ready_stays_waiting_for_chunk(self):
        coord = ChunkSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1)

        req = _make_request("r1", status=RequestStatus.WAITING_FOR_CHUNK)
        waiting = MockQueue([req])
        running: list = []

        coord.process_pending_chunks(
            waiting,
            running,
            chunk_ready_req_ids=set(),
            chunk_finished_req_ids=set(),
        )

        self.assertEqual(req.status, RequestStatus.WAITING_FOR_CHUNK)

    def test_stage_0_is_noop(self):
        coord = ChunkSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=0)
        req = _make_request("r1")
        waiting = MockQueue([req])
        running: list = []

        coord.process_pending_chunks(
            waiting,
            running,
            chunk_ready_req_ids={"r1"},
            chunk_finished_req_ids=set(),
        )
        self.assertNotEqual(req.status, RequestStatus.WAITING_FOR_CHUNK)


class TestChunkCoordinatorRestoreQueues(unittest.TestCase):
    """Test 6: restore_queues returns waiting-for-chunk requests."""

    def test_restore(self):
        coord = ChunkSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1)

        r1 = _make_request("r1")
        r2 = _make_request("r2")
        coord._waiting_for_chunk_waiting.append(r1)
        coord._waiting_for_chunk_running.append(r2)

        waiting = MockQueue()
        running: list = []

        coord.restore_queues(waiting, running)

        self.assertIn(r1, waiting)
        self.assertIn(r2, running)
        self.assertEqual(len(coord._waiting_for_chunk_waiting), 0)
        self.assertEqual(len(coord._waiting_for_chunk_running), 0)


class TestChunkCoordinatorFinishedSignal(unittest.TestCase):
    """Test 8: chunk_finished_req_ids → finished_requests."""

    def test_finished_signal(self):
        coord = ChunkSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1)

        req = _make_request("r1", status=RequestStatus.WAITING_FOR_CHUNK)
        waiting = MockQueue([req])
        running: list = []

        coord.process_pending_chunks(
            waiting,
            running,
            chunk_ready_req_ids={"r1"},
            chunk_finished_req_ids={"r1"},
        )

        self.assertIn("r1", coord.finished_requests)


class TestChunkCoordinatorUpdateRequestData(unittest.TestCase):
    """Test update_request_data applies chunk payloads to requests."""

    def test_ar_mode(self):
        coord = ChunkSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1)

        req = _make_request("r1")
        requests = {"r1": req}

        chunk_data = {"r1": {"thinker_embeddings": "tensor_data", "finished": False}}

        coord.update_request_data(requests, chunk_data, model_mode="ar")

        self.assertEqual(req.additional_information, chunk_data["r1"])

    def test_generation_mode(self):
        coord = ChunkSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1)

        req = _make_request("r1")
        req.prompt_token_ids = [0, 0, 0]
        requests = {"r1": req}

        chunk_data = {"r1": {"code_predictor_codes": [10, 20, 30]}}

        coord.update_request_data(requests, chunk_data, model_mode="generation")

        self.assertEqual(req.prompt_token_ids, [10, 20, 30])
        self.assertEqual(req.num_computed_tokens, 0)


class TestChunkCoordinatorPostprocess(unittest.TestCase):
    """Test postprocess_scheduler_output clears ready chunks."""

    def test_clear_ready(self):
        coord = ChunkSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1)
        coord.requests_with_ready_chunks = {"r1", "r2"}

        new_req = SimpleNamespace(req_id="r1")
        cached_reqs = SimpleNamespace(req_ids=["r2"])
        scheduler_output = SimpleNamespace(
            scheduled_new_reqs=[new_req],
            scheduled_cached_reqs=cached_reqs,
        )

        coord.postprocess_scheduler_output(scheduler_output)

        self.assertEqual(coord.requests_with_ready_chunks, set())


class TestBackwardCompatAlias(unittest.TestCase):
    """Verify ChunkSchedulingCoordinator is an alias for OmniSchedulingCoordinator."""

    def test_alias(self):
        self.assertIs(ChunkSchedulingCoordinator, OmniSchedulingCoordinator)


class TestWaitingForInputTransition(unittest.TestCase):
    """Test B8: process_pending_batch_inputs transitions WAITING_FOR_INPUT."""

    def test_transition_on_recv(self):
        coord = OmniSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1)

        req = _make_request("r1", status=RequestStatus.WAITING_FOR_INPUT)
        waiting = MockQueue([req])
        running: list = []

        coord.process_pending_batch_inputs(
            waiting,
            running,
            stage_recv_req_ids={"r1"},
        )

        self.assertEqual(req.status, RequestStatus.WAITING)

    def test_stays_waiting_for_input_if_not_received(self):
        coord = OmniSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1)

        req = _make_request("r1", status=RequestStatus.WAITING_FOR_INPUT)
        waiting = MockQueue([req])
        running: list = []

        coord.process_pending_batch_inputs(
            waiting,
            running,
            stage_recv_req_ids=set(),
        )

        self.assertEqual(req.status, RequestStatus.WAITING_FOR_INPUT)
        self.assertEqual(len(coord._waiting_for_input), 1)

    def test_stage_0_is_noop(self):
        coord = OmniSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=0)

        req = _make_request("r1", status=RequestStatus.WAITING_FOR_INPUT)
        waiting = MockQueue([req])
        running: list = []

        coord.process_pending_batch_inputs(
            waiting,
            running,
            stage_recv_req_ids={"r1"},
        )
        self.assertEqual(req.status, RequestStatus.WAITING_FOR_INPUT)

    def test_restore_queues_includes_waiting_for_input(self):
        coord = OmniSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1)

        r1 = _make_request("r1")
        coord._waiting_for_input.append(r1)

        waiting = MockQueue()
        running: list = []

        coord.restore_queues(waiting, running)

        self.assertIn(r1, waiting)
        self.assertEqual(len(coord._waiting_for_input), 0)

    def test_batch_mode_auto_transitions_waiting_to_waiting_for_input(self):
        """In batch mode (not async_chunk), fresh WAITING requests on
        non-Stage-0 should be transitioned to WAITING_FOR_INPUT."""
        coord = OmniSchedulingCoordinator(
            scheduler_max_num_seqs=10,
            stage_id=1,
            async_chunk=False,
        )

        req = _make_request("r1", status=RequestStatus.WAITING)
        waiting = MockQueue([req])
        running: list = []

        coord.process_pending_batch_inputs(
            waiting,
            running,
            stage_recv_req_ids=set(),
        )

        self.assertEqual(req.status, RequestStatus.WAITING_FOR_INPUT)
        self.assertEqual(len(coord._waiting_for_input), 1)
        self.assertEqual(len(coord.pending_input_registrations), 1)

    def test_async_chunk_mode_does_not_auto_transition(self):
        """In async_chunk mode, fresh WAITING requests should NOT be
        transitioned to WAITING_FOR_INPUT."""
        coord = OmniSchedulingCoordinator(
            scheduler_max_num_seqs=10,
            stage_id=1,
            async_chunk=True,
        )

        req = _make_request("r1", status=RequestStatus.WAITING)
        waiting = MockQueue([req])
        running: list = []

        coord.process_pending_batch_inputs(
            waiting,
            running,
            stage_recv_req_ids=set(),
        )

        self.assertEqual(req.status, RequestStatus.WAITING)

    def test_pending_input_registrations(self):
        coord = OmniSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1)

        req = _make_request("r1", status=RequestStatus.WAITING_FOR_INPUT)
        waiting = MockQueue([req])
        running: list = []

        coord.process_pending_batch_inputs(
            waiting,
            running,
            stage_recv_req_ids=set(),
        )

        self.assertEqual(len(coord.pending_input_registrations), 1)
        self.assertEqual(coord.pending_input_registrations[0].request_id, "r1")


if __name__ == "__main__":
    unittest.main()
